#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.dag_builder import *
from codegen.mir_emitter import *
from codegen.passes import *
from codegen.spec import *
from codegen.dag_printer import *
from codegen.scheduler import *


class InstructionSelection(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc: MachineFunction):
        self.func = mfunc.func_info.func
        self.mfunc = mfunc
        self.calling_conv = mfunc.target_info.get_calling_conv()
        self.selector = mfunc.target_info.get_instruction_selector()
        self.legalizer = mfunc.target_info.get_legalizer()
        self.data_layout = self.func.module.data_layout
        self.target_lowering = mfunc.target_info.get_lowering()
        self.func_info = mfunc.func_info
        self.reg_info = mfunc.target_info.get_register_info()

        self.init()
        self.combine()
        # for bb, dag in self.dags.items():
        #     mbb = self.mbb_map[bb]
        #     print_dag(id(mbb), dag, "./data/dag", self.func.name)
        self.legalize_type()
        self.legalize()
        self.select()
        self.schedule()

    def need_export_for_successors(self, inst: Instruction):
        for user in inst.uses:
            if isinstance(user, PHINode):
                continue

            if user.block != inst.block:
                return True

        return False

    def init(self):
        self.dags = {}
        self.mbbs = []
        self.mbb_map = {}
        self.dag_builders = {}
        self.func_info = FunctionLoweringInfo(
            self.mfunc.func_info.func, self.mfunc.target_info.get_calling_conv())

        if len(self.func.bbs) == 0:
            return

        for bb in self.func.bbs:
            # Prepare basic block for machine instructions.
            mbb = MachineBasicBlock()
            self.mfunc.append_bb(mbb)
            self.mbb_map[bb] = mbb

            dag_builder = DagBuilder(
                self.mfunc, self.mbb_map, self.data_layout, self.func_info)

            self.dag_builders[bb] = dag_builder

        dag_builder = self.dag_builders[self.func.bbs[0]]

        self.target_lowering.lower_arguments(
            self.mfunc.func_info.func, dag_builder)

        for bb in self.func.bbs:
            for inst in bb.insts:
                if isinstance(inst, AllocaInst) and inst.is_static_alloca():
                    if len(inst.uses) == 0:
                        continue

                    if inst in self.func_info.frame_map:
                        continue

                    self.create_frame_idx(inst, False)

                elif self.need_export_for_successors(inst):
                    assert(inst not in self.func_info.reg_value_map)

                    value_types = compute_value_types(
                        inst.ty, self.data_layout)

                    reg_info = self.reg_info

                    regs = []
                    for ty in value_types:
                        reg_vt = reg_info.get_register_type(ty)
                        reg_count = reg_info.get_register_count(ty)

                        for idx in range(reg_count):
                            vreg = self.target_lowering.get_machine_vreg(
                                reg_vt)
                            regs.append(
                                self.mfunc.reg_info.create_virtual_register(vreg))

                    if inst not in self.func_info.reg_value_map:
                        self.func_info.reg_value_map[inst] = regs

        for bb in self.func.bbs:
            dag_builder = self.dag_builders[bb]
            for inst in bb.insts:
                dag_builder.visit(inst)

            dag = dag_builder.g
            self.dags[bb] = dag

        for bb in self.func.bbs:
            dag = self.dags[bb]
            dag.remove_unreachable_nodes()

    def compute_alloca_size(self, value):
        size, align = self.data_layout.get_type_size_in_bits(value.alloca_ty)

        size = int(int((size + align - 1) / align) * align)

        return int(size / 8), int(align / 8)

    def create_frame_idx(self, value, is_fixed, offset=0):
        size, align = self.compute_alloca_size(value)

        align = max(align, value.align)

        if is_fixed:
            frame_idx = self.mfunc.frame.create_fixed_stack_object(
                size, offset)
        else:
            frame_idx = self.mfunc.frame.create_stack_object(size, align)

        self.func_info.frame_map[value] = frame_idx

    def _bfs_transform(self, node: DagNode, dag, action, visited):
        assert(isinstance(node, DagNode))

        if node in visited:
            return visited[node]

        new_nodes = []
        for op in set(node.operands):
            new_nodes.append(self._bfs_transform(
                op.node, dag, action, visited))

        for op, new_op in zip(node.operands, new_nodes):
            new_op
            op.node = new_op

        visited[node] = action(node, dag)

        return visited[node]

    def _bfs(self, node: DagNode, action, visited):
        assert(isinstance(node, DagNode))

        if node in visited:
            return

        visited.add(node)

        for op in set(node.operands):
            self._bfs(op.node, action, visited)

        action(node)

    def _bfs_pre(self, node: DagNode, action, visited):
        assert(isinstance(node, DagNode))

        if node in visited:
            return

        visited.add(node)

        action(node)

        for op in set(node.operands):
            self._bfs(op.node, action, visited)

    def do_legalize(self):
        def iter_all_nodes(dag):
            def iter_all_nodes_bfs(node: DagNode, visited):
                assert(isinstance(node, DagNode))

                if node in visited:
                    return

                visited.add(node)

                for op in set(node.operands):
                    yield from iter_all_nodes_bfs(op.node, visited)

                yield node

            return iter_all_nodes_bfs(dag.root.node, set())

        def create_legalize_node(dag, results):
            def legalize_node(node):
                results[node] = self.target_lowering.lower(node, dag)

            return legalize_node

        for bb in self.func.bbs:
            dag = self.dags[bb]

            results = {}

            while True:
                changed = False

                for node in reversed(list(iter_all_nodes(dag))):
                    if node in results:
                        continue

                    changed = True
                    results[node] = self.target_lowering.lower(node, dag)

                if not changed:
                    break

                operands = set()

                def collect_operands(node):
                    for operand in node.operands:
                        operands.add(operand)

                self._bfs(dag.root.node, collect_operands, set())

                operands.add(dag.root)

                assignments = set()
                for operand in operands:
                    if operand.node in results:
                        assignments.add((operand, results[operand.node]))

                for old_node, new_node in results.items():
                    if old_node is not new_node:
                        for op in old_node.uses:
                            new_node.uses.add(op)

                    changed |= old_node is not new_node

                for operand, new_node in assignments:
                    operand.node = new_node

            dag.remove_unreachable_nodes()

        return changed

    def legalize(self):
        self.do_legalize()

    def do_legalize_type(self):
        def create_legalize_type_node(dag, results):
            def legalize_node_result(node):
                new_node = self.legalizer.legalize_node_result(
                    node, dag, results)

                if new_node:
                    results[node] = new_node
                else:
                    results[node] = node

            return legalize_node_result

        changed = False

        for bb in self.func.bbs:
            dag = self.dags[bb]

            results = {}
            self._bfs(dag.root.node, create_legalize_type_node(
                dag, results), set())

            nodes = set()

            def collect_nodes(node):
                nodes.add(node)

            self._bfs(dag.root.node, collect_nodes, set())

            legalized = {}
            for node in nodes:
                for idx in range(len(node.operands)):
                    new_node = self.legalizer.legalize_node_operand(
                        node, idx, dag, results)

                    if new_node:
                        legalized[node] = new_node

            assignments = set()
            for node in nodes:
                for operand in node.operands:
                    if operand.node in legalized:
                        assignments.add((operand, legalized[operand.node]))

            if dag.root.node in legalized:
                assignments.add((dag.root, legalized[dag.root.node]))

            for operand, new_node in assignments:
                old_node = operand.node

                if old_node is not new_node:
                    for op in old_node.uses:
                        new_node.uses.add(op)

                operand.node = new_node

                changed |= old_node is not new_node

            dag.remove_unreachable_nodes()

        return changed

    def legalize_type(self):
        while self.do_legalize_type():
            pass

    def combine_node(self, node: DagNode, dag: Dag):
        if node.opcode == VirtualDagOps.MERGE_VALUES:
            ops = node.operands

            for use in node.uses:
                new_op = node.operands[use.ref.index]
                use.ref = new_op
                new_op.node.uses.add(use)

            dag.remove_node(node)

            return DagValue(node, 0)

        if node.opcode == VirtualDagOps.INSERT_VECTOR_ELT:
            in_vec = node.operands[0]
            in_val = node.operands[1]
            idx = node.operands[2]

            if in_vec.node.opcode == VirtualDagOps.BUILD_VECTOR and isinstance(idx.node, ConstantDagNode) and len(in_vec.node.uses) == 1:
                ops = list(in_vec.node.operands)
                assert(ops[idx.node.value.value].ty == in_val.ty)
                ops[idx.node.value.value] = in_val

                return DagValue(dag.add_node(VirtualDagOps.BUILD_VECTOR, in_vec.node.value_types, *ops), 0)

        return None

    def combine(self):
        def node_collector(dag, nodes):
            def func(node):
                nodes.append(node)

            return func

        for bb in self.func.bbs:
            dag = self.dags[bb]

            nodes = []
            self._bfs(dag.root.node, node_collector(dag, nodes), set())

            while len(nodes) > 0:
                node = nodes.pop()

                value = self.combine_node(node, dag)

                if value is None or value.node == node:
                    continue

                for user in node.uses:
                    value.node.uses.add(user)

                if value.node not in nodes:
                    nodes.append(value.node)

                for user in node.uses:
                    if user not in nodes:
                        nodes.append(user.source)

                    user.ref.node = value.node

                dag.remove_unreachable_nodes()

    def select(self):
        class ISelNode:
            def __init__(self, node):
                self.node = node
                self.compute_hash()

            def compute_hash(self):
                node = self.node

                operands = tuple((id(opnd.node), opnd.index)
                                 for opnd in node.operands)
                self._hash_val = hash((node.opcode, operands))

            def __hash__(self):
                return self._hash_val

            def __eq__(self, other):
                if not isinstance(other, ISelNode):
                    return False

                this_node = self.node
                other_node = other.node

                for opnd1, opnd2 in zip(this_node.operands, other_node.operands):
                    if opnd1.node is not opnd2.node:
                        return False
                    if opnd1.index != opnd2.index:
                        return False
                return this_node.opcode == other_node.opcode

        def create_select_node(dag, results):
            def select_node(node):
                if isinstance(node, MachineDagNode):
                    return

                selected = self.selector.select(node, dag)
                if selected is not node:
                    results[node.number] = (node, selected)
            return select_node

        for bb in self.func.bbs:
            dag = self.dags[bb]

            results = {}
            self._bfs(dag.root.node, create_select_node(
                dag, results), set())

            operands = list()

            def collect_operands(node):
                for operand in node.operands:
                    operands.append(operand)

            operands.append(dag.root)

            self._bfs_pre(dag.root.node, collect_operands, set())

            changed = True
            while changed:
                changed = False

                assignments = set()
                for operand in operands:
                    if operand.node.number in results:
                        _, result = results[operand.node.number]
                        assignments.add((operand, result))

                for old_node, new_node in results.values():
                    if old_node is not new_node:
                        for op in old_node.uses:
                            new_node.uses.add(op)

                for operand, new_node in assignments:
                    if operand.node != new_node:
                        changed = True
                    operand.node = new_node

            dag.remove_unreachable_nodes()

    def schedule(self):
        vr_map = {}

        mbb = self.mbb_map[self.func.bbs[0]]
        for reg, vreg in self.mfunc.reg_info.live_ins:
            minst = MachineInstruction(TargetDagOps.COPY)
            minst.add_reg(vreg, RegState.Define)
            minst.add_reg(reg, RegState.Non)

            mbb.append_inst(minst)

        scheduler = ListScheduler()

        for bb in self.func.bbs:
            dag = self.dags[bb]
            mbb = self.mbb_map[bb]
            sched_dag = ScheduleDag(self.mfunc, dag)
            sched_dag.build()

            emitter = MachineInstrEmitter(mbb, vr_map)

            # print_sunit_dag("sunit-" + str(id(mbb)), sched_dag,
            #                 "./data/dag", self.func.name)

            nodes = list(scheduler.schedule(sched_dag))
            for node in scheduler.schedule(sched_dag):
                emitter.emit(node, dag)
