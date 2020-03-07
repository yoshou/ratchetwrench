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

        self.init()
        self.combine()
        self.legalize_type()
        self.legalize()
        # for bb, dag in self.dags.items():
        #     mbb = self.mbb_map[bb]
        #     print_dag(id(mbb), dag, "./data/dag", self.func.name)
        self.select()
        self.schedule()

    def need_export_for_successors(self, inst: Instruction):
        for user in inst.uses:
            if user.block != inst.block:
                return True

        return False

    def init(self):
        self.dags = {}
        self.mbbs = []
        self.mbb_map = {}
        self.dag_builders = {}

        if len(self.func.bbs) == 0:
            return

        for bb in self.func.bbs:
            # Prepare basic block for machine instructions.
            mbb = MachineBasicBlock(self.mfunc)
            self.mfunc.bbs.append(mbb)
            self.mbb_map[bb] = mbb

            dag_builder = DagBuilder(
                self.mfunc, self.mbb_map, self.data_layout)

            self.dag_builders[bb] = dag_builder

        dag_builder = self.dag_builders[self.func.bbs[0]]

        self.target_lowering.lower_arguments(
            self.mfunc.func_info.func, dag_builder)

        for bb in self.func.bbs:
            for inst in bb.insts:
                if isinstance(inst, AllocaInst) and inst.is_static_alloca():
                    if len(inst.uses) == 0:
                        continue
                    self.create_frame_idx(inst, False)

                elif self.need_export_for_successors(inst):
                    assert(inst not in self.func_info.reg_value_map)

                    value_types = compute_value_types(
                        inst.ty, self.data_layout)

                    if len(value_types) > 1:
                        raise NotImplementedError()

                    regs = []
                    for ty in value_types:
                        vreg = self.target_lowering.get_machine_vreg(ty)
                        regs.append(vreg)

                    self.func_info.reg_value_map[inst] = self.mfunc.reg_info.create_virtual_register(
                        regs[0])

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

        return int(size / 8), int(align / 8)

    def create_frame_idx(self, value, is_fixed, offset=0):
        if value in self.func_info.frame_map:
            return

        size, align = self.compute_alloca_size(value)

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

    def _bfs2(self, node: DagNode, action, visited):
        assert(isinstance(node, DagNode))

        if node in visited:
            return

        visited.add(node)

        for op in set(node.operands):
            self._bfs(op.node, action, visited)

        action(node)

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

    def promote_integer_result_setcc(self, node, dag, legalized):
        setcc_ty = MachineValueType(ValueType.I8)

        return dag.add_node(node.opcode, [setcc_ty], *node.operands)

    # def promote_integer_result_setcc(self, node, dag, legalized):
    #     setcc_ty = MachineValueType(ValueType.I32)

    #     return dag.add_node(node.opcode, [setcc_ty], *node.operands)

    def get_legalized_op(self, operand, legalized):
        if operand.node in legalized:
            return DagValue(legalized[operand.node], operand.index)

        return operand

    def promote_integer_result_bin(self, node, dag, legalized):
        lhs = self.get_legalized_op(node.operands[0], legalized)
        rhs = self.get_legalized_op(node.operands[1], legalized)

        return dag.add_node(node.opcode, [lhs.ty], lhs, rhs)

    def promote_integer_result(self, node, dag, legalized):
        if node.opcode == VirtualDagOps.SETCC:
            return self.promote_integer_result_setcc(node, dag, legalized)
        elif node.opcode in [VirtualDagOps.AND, VirtualDagOps.OR]:
            return self.promote_integer_result_bin(node, dag, legalized)
        elif node.opcode in [VirtualDagOps.LOAD]:
            return dag.add_node(node.opcode, [MachineValueType(ValueType.I32)], *node.operands)
        else:
            raise ValueError("No method to promote.")

    def legalize_node_type(self, node: DagNode, dag: Dag, legalized):
        for vt in node.value_types:
            if vt.value_type == ValueType.I1:
                return self.promote_integer_result(node, dag, legalized)

        return node

    def do_legalize_type(self):
        def create_legalize_type_node(dag, results):
            def legalize_type_node(node):
                results[node] = self.legalizer.legalize_node_type(
                    node, dag, results)

            return legalize_type_node

        changed = False

        for bb in self.func.bbs:
            dag = self.dags[bb]

            results = {}
            self._bfs(dag.root.node, create_legalize_type_node(
                dag, results), set())

            operands = set()

            def collect_operands(node):
                for operand in node.operands:
                    operands.add(operand)

            self._bfs(dag.root.node, collect_operands, set())

            operands.add(dag.root)

            assignments = set()
            for operand in operands:
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

                for op in node.uses:
                    value.node.uses.add(op)

                if value.node not in nodes:
                    nodes.append(value.node)

                for user in node.uses:
                    if user not in nodes:
                        nodes.append(user)

            dag.remove_unreachable_nodes()

    def select(self):
        def create_select_node(dag, results):
            def select_node(node):
                results[node] = self.selector.select(node, dag)
            return select_node

        for bb in self.func.bbs:
            dag = self.dags[bb]

            results = {}
            self._bfs2(dag.root.node, create_select_node(dag, results), set())

            operands = set()

            def collect_operands(node):
                for operand in node.operands:
                    operands.add(operand)

            self._bfs(dag.root.node, collect_operands, set())

            # for node in results.values():
            #     for operand in node.operands:
            #         operands.add(operand)

            operands.add(dag.root)

            assignments = set()
            for operand in operands:
                if operand.node in results:
                    assignments.add((operand, results[operand.node]))

            for old_node, new_node in results.items():
                if old_node is not new_node:
                    for op in old_node.uses:
                        new_node.uses.add(op)

            for operand, new_node in assignments:
                operand.node = new_node

            dag.remove_unreachable_nodes()

    def schedule(self):

        def bfs(node: DagNode, action, visited):
            if node in visited:
                return

            visited.add(node)

            for op in node.operands:
                bfs(op.node, action, visited)

            action(node)

        def topological_sort_schedule(root):
            lst = []
            bfs(root, lambda node: lst.append(node), set())
            return lst

        def glued_node_iter(node):
            glue_ty = MachineValueType(ValueType.GLUE)
            yield node
            while len(node.operands) > 0 and node.operands[-1].ty == glue_ty:
                node = node.operands[-1].node
                yield node

        def bfs2(node, action, visited):
            if node in visited:
                return

            visited.add(node)

            for succ in node.preds:
                bfs2(succ.node, action, visited)

            action(node)

        def topological_sort_schedule2(nodes):
            lst = []
            visited = set()
            for node in nodes:
                bfs2(node, lambda n: lst.append(n), visited)
            return lst

        vr_map = {}

        mbb = self.mbb_map[self.func.bbs[0]]
        for reg, vreg in self.mfunc.reg_info.live_ins:
            minst = MachineInstruction(TargetDagOps.COPY)
            minst.add_reg(vreg, RegState.Define)
            minst.add_reg(reg, RegState.Non)

            mbb.append_inst(minst)

        for bb in self.func.bbs:
            dag = self.dags[bb]
            mbb = self.mbb_map[bb]
            sched_dag = ScheduleDag(self.mfunc, dag)
            # sched_dag.build()

            emitter = MachineInstrEmitter(mbb, vr_map)
            nodes = topological_sort_schedule(dag.root.node)

            sched_node_map = {}

            # Create nodes
            work_list = [dag.root.node]
            visited = set()

            while len(work_list) > 0:
                node = work_list.pop()

                if node in visited:
                    continue

                for operand in node.operands:
                    work_list.append(operand.node)

                visited.add(node)

                sched_node = sched_dag.create_sched_node(node)
                sched_node_map[node] = sched_node

                for glued_node in list(glued_node_iter(node))[1:]:
                    sched_node_map[glued_node] = sched_node

                    for operand in glued_node.operands:
                        work_list.append(operand.node)
                    visited.add(glued_node)

            # Create edges
            for sched_node in sched_node_map.values():
                for glued_node in glued_node_iter(sched_node.node):
                    for operand in glued_node.operands:
                        op_sched_node = sched_node_map[operand.node]

                        sched_edge = ScheduleEdge(op_sched_node)
                        sched_node.add_pred(sched_edge)

            # Run scheduler
            def schedule_nodes(sched_nodes):
                for sched_node in reversed(topological_sort_schedule2(sched_node_map.values())):
                    for glued_node in glued_node_iter(sched_node.node):
                        yield glued_node

            for node in reversed(list(schedule_nodes(sched_node_map.values()))):
                emitter.emit(node, dag)
