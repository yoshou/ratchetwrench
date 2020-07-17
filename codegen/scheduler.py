#!/usr/bin/env python
# -*- coding: utf-8 -*-


class ScheduleUnit:
    def __init__(self, node, id):
        self.node = node
        self.id = id
        self.succs = []
        self.preds = []
        self._height = -1
        self.latency = 0

    def compute_height(self):
        # Calculates the maximal path from the node to the entry.

        if self._height >= 0:
            return

        max_height = 0
        for succ in self.succs:
            succ.node.compute_height()

            max_height = max(max_height, succ.node._height + succ.node.latency)

        self._height = max_height

    def get_height(self):
        if self._height == -1:
            self.compute_height()

        return self._height

    def set_height(self, value):
        self._height = value

    def add_pred(self, edge):
        self.preds.append(edge)
        edge.node.succs.append(ScheduleEdge(self, edge.kind))

    def add_succ(self, edge):
        self.succs.append(edge)
        edge.node.preds.append(ScheduleEdge(self, edge.kind))


from enum import Enum, auto


class DependencyKind(Enum):
    Data = auto()
    Order = auto()


class ScheduleEdge:
    def __init__(self, node, kind):
        self.node = node
        self.kind = kind


def glued_node_iter(node):
    from codegen.types import MachineValueType, ValueType

    glue_ty = MachineValueType(ValueType.GLUE)
    yield node
    while len(node.operands) > 0 and node.operands[-1].ty == glue_ty:
        node = node.operands[-1].node
        yield node


def is_passive_node(node):
    from codegen.dag import ConstantDagNode, ConstantFPDagNode, RegisterDagNode, GlobalAddressDagNode, BasicBlockDagNode
    from codegen.dag import FrameIndexDagNode, ConstantPoolDagNode, ExternalSymbolDagNode
    from codegen.dag import VirtualDagOps

    if isinstance(node, (ConstantDagNode, ConstantFPDagNode, RegisterDagNode, GlobalAddressDagNode, BasicBlockDagNode,
                         FrameIndexDagNode, ConstantPoolDagNode, ExternalSymbolDagNode)):
        return True

    if node.opcode == VirtualDagOps.ENTRY:
        return True

    return False


class ScheduleDag:
    def __init__(self, mfunc, dag):
        self.mfunc = mfunc
        self.dag = dag
        self.nodes = []

    def create_sched_node(self, node):
        sched_node = ScheduleUnit(node, len(self.nodes))

        from codegen.spec import MachineInstructionDef

        if isinstance(node.opcode, MachineInstructionDef):
            sched = node.opcode.sched
            if sched:
                sched_node.latency = sched.latency

        self.nodes.append(sched_node)
        return sched_node

    def build(self):
        dag = self.dag

        sched_node_map = {}

        # Create nodes
        work_list = [dag.root.node]
        visited = set()

        while len(work_list) > 0:
            node = work_list.pop()

            if node in visited:
                continue

            if is_passive_node(node):
                continue

            sched_node = self.create_sched_node(node)

            # Assign same schedule node to all glued nodes.
            for glued_node in list(glued_node_iter(node)):
                sched_node_map[glued_node] = sched_node

                for operand in glued_node.operands:
                    work_list.append(operand.node)
                visited.add(glued_node)

        from codegen.types import ValueType

        # Create edges
        for node, sched_node in sched_node_map.items():
            if node != sched_node.node:
                continue

            for glued_node in glued_node_iter(sched_node.node):
                for operand in glued_node.operands:
                    if is_passive_node(operand.node):
                        continue

                    op_sched_node = sched_node_map[operand.node]

                    if sched_node == op_sched_node:
                        continue

                    kind = DependencyKind.Data

                    if operand.ty.value_type == ValueType.OTHER:
                        kind = DependencyKind.Order

                    sched_edge = ScheduleEdge(op_sched_node, kind)

                    sched_node.add_succ(sched_edge)

        self.sched_node_map = sched_node_map


def bfs(node, action, visited):
    if node in visited:
        return

    visited.add(node)

    for succ in node.succs:
        bfs(succ.node, action, visited)

    action(node)


def topological_sort(nodes):
    lst = []
    visited = set()
    for node in nodes:
        bfs(node, lambda n: lst.append(n), visited)

    lst.reverse()
    return lst


class TopologicalSortScheduler:

    def schedule(self, sched_dag):
        def schedule_nodes(sched_nodes):
            temp = topological_sort(set(sched_nodes))
            for sched_node in topological_sort(sched_nodes):
                for glued_node in glued_node_iter(sched_node.node):
                    yield glued_node

        return reversed(list(schedule_nodes(sched_dag.sched_node_map.values())))


class ScheduingPriorityQueueEntry:
    def __init__(self, value, comp_func):
        self.value = value
        self.comp_func = comp_func

    def __lt__(self, other):
        return self.comp_func(self.value, other.value)


class ScheduingPriorityQueue:
    def __init__(self, comp_func):
        from queue import PriorityQueue

        self.comp_func = comp_func
        self._queue = PriorityQueue()

    def put(self, value):
        self._queue.put(ScheduingPriorityQueueEntry(value, self.comp_func))

    def get(self):
        entry = self._queue.get()
        return entry.value

    def empty(self):
        return self._queue.empty()


def calc_sethi_ullman_number(sunit, numbers):
    work_list = [sunit]

    if numbers[sunit] != 0:
        return

    while work_list:
        sunit = work_list[-1]

        sethi_ullman_num = 0
        extra = 0
        all_deps_known = True

        for succ in sunit.succs:
            if succ.kind != DependencyKind.Data:
                continue

            succ_sunit = succ.node

            if numbers[succ_sunit] == 0:
                work_list.append(succ_sunit)
                all_deps_known = False

            if numbers[succ_sunit] > sethi_ullman_num:
                extra = 0
            else:
                extra += 1

            sethi_ullman_num = max(sethi_ullman_num, numbers[succ_sunit])

        if not all_deps_known:
            continue

        sethi_ullman_num += extra
        if sethi_ullman_num == 0:
            sethi_ullman_num = 1
        numbers[sunit] = sethi_ullman_num

        work_list.pop(-1)


def calc_sethi_ullman_number_bottom(sunit, numbers):
    work_list = [sunit]

    if numbers[sunit] != 0:
        return

    while work_list:
        sunit = work_list[-1]

        sethi_ullman_num = 0
        extra = 0
        all_deps_known = True

        for pred in sunit.preds:
            if pred.kind != DependencyKind.Data:
                continue

            pred_sunit = pred.node

            if numbers[pred_sunit] == 0:
                work_list.append(pred_sunit)
                all_deps_known = False

            if numbers[pred_sunit] > sethi_ullman_num:
                extra = 0
            else:
                extra += 1

            sethi_ullman_num = max(sethi_ullman_num, numbers[pred_sunit])

        if not all_deps_known:
            continue

        sethi_ullman_num += extra
        if sethi_ullman_num == 0:
            sethi_ullman_num = 1
        numbers[sunit] = sethi_ullman_num

        work_list.pop(-1)


def calc_sethi_ullman_numbers(sunits, numbers):
    for sunit in sunits:
        numbers[sunit] = 0

    for sunit in sunits:
        calc_sethi_ullman_number(sunit, numbers)


def calc_sethi_ullman_numbers_bottom(sunits, numbers):
    for sunit in sunits:
        numbers[sunit] = 0

    for sunit in sunits:
        calc_sethi_ullman_number_bottom(sunit, numbers)


class ListScheduler:
    def __init__(self):
        pass

    def schedule_bottom(self, sched_dag):

        sunits = set(sched_dag.sched_node_map.values())

        numbers = {}
        calc_sethi_ullman_numbers_bottom(list(sunits), numbers)

        def get_closest(sunit):
            max_height = 0
            for succ in sunit.succs:
                if succ.kind != DependencyKind.Data:
                    continue

                max_height = max(max_height, succ.node.get_height())

            return max_height

        def get_num_scratches(sunit):
            scratches = 0
            for pred in sunit.preds:
                if pred.kind != DependencyKind.Data:
                    continue

                scratches += 1

            return scratches

        def comp_sunit(a, b):
            if numbers[a] != numbers[b]:
                return numbers[a] > numbers[b]

            a_height = get_closest(a)
            b_height = get_closest(b)

            if a_height != b_height:
                return a_height < b_height

            a_scratch = get_num_scratches(a)
            b_scratch = get_num_scratches(b)

            # How many registers becomes live when the node is scheduled.
            if a_scratch != b_scratch:
                return a_scratch > b_scratch

            return a.id > b.id

        available_queue = ScheduingPriorityQueue(comp_sunit)

        preds = {}
        for sunit in sunits:
            preds[sunit] = set([pred.node for pred in sunit.preds])

        for sunit in sunits:
            if len(preds[sunit]) == 0:
                available_queue.put(sunit)

        lst = []

        while not available_queue.empty():
            sunit = available_queue.get()

            lst.append(sunit)

            sunit.set_height(len(lst))

            for succ_sunit in set([succ.node for succ in sunit.succs]):
                preds[succ_sunit].remove(sunit)

                if len(preds[succ_sunit]) == 0:
                    available_queue.put(succ_sunit)

        assert(len(sunits) == len(lst))

        def list_insts(units):
            for sched_node in units:
                for glued_node in glued_node_iter(sched_node.node):
                    yield glued_node

        return reversed(list(list_insts(lst)))

    def schedule_top(self, sched_dag):

        sunits = set(sched_dag.sched_node_map.values())

        numbers = {}
        calc_sethi_ullman_numbers(list(sunits), numbers)

        def get_closest(sunit):
            max_height = 0
            for pred in sunit.preds:
                if pred.kind != DependencyKind.Data:
                    continue

                max_height = max(max_height, pred.node.get_height())

            return max_height

        def get_num_scratches(sunit):
            scratches = 0
            for pred in sunit.succs:
                if pred.kind != DependencyKind.Data:
                    continue

                scratches += 1

            return scratches

        def comp_sunit(a, b):
            if numbers[a] == numbers[b]:
                # a_height = get_closest(a)
                # b_height = get_closest(b)

                # if a_height != b_height:
                #     return a_height < b_height

                # a_scratch = get_num_scratches(a)
                # b_scratch = get_num_scratches(b)

                # if a_scratch != b_scratch:
                #     return a_scratch > b_scratch

                return a.id > b.id

            return numbers[a] > numbers[b]

        available_queue = ScheduingPriorityQueue(comp_sunit)

        succs = {}
        for sunit in sunits:
            succs[sunit] = set([succ.node for succ in sunit.succs])

        for sunit in sunits:
            if len(succs[sunit]) == 0:
                available_queue.put(sunit)

        lst = []

        while not available_queue.empty():
            sunit = available_queue.get()
            temp = numbers[sunit]

            lst.append(sunit)

            sunit.set_height(len(lst))

            for pred_sunit in set([pred.node for pred in sunit.preds]):
                succs[pred_sunit].remove(sunit)

                if len(succs[pred_sunit]) == 0:
                    available_queue.put(pred_sunit)

        assert(len(sunits) == len(lst))

        lst.reverse()

        def list_insts(units):
            for sched_node in units:
                for glued_node in glued_node_iter(sched_node.node):
                    yield glued_node

        return reversed(list(list_insts(lst)))

    def schedule(self, sched_dag):
        return self.schedule_bottom(sched_dag)
