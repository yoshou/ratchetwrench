from rachetwrench.ir.values import *


class DomTreeNode:
    def __init__(self, value, idx):
        self.value = value
        self.idx = idx
        self.doms = set([self])
        self.is_changed = True
        self.parent = None
        self.children = []


def get_value_successors(value):
    if isinstance(value, User):
        return [operand for operand in value.operands if operand]

    return []


def get_value_predecessors(value):
    return value.uses


class DominatorTree:
    def __init__(self):
        self.nodes = []
        self.value_to_node = {}

    def is_nodes_changed(self):
        for node in self.nodes:
            if node.is_changed:
                return True

        return False

    def find_roots(self, func):
        roots = []

        for bb in func.bbs:
            for inst in bb.insts:
                if len(inst.uses) == 0:
                    roots.append(inst)

        return roots

    def dfs(self, roots):
        nodes = []
        visited = set()

        def dfs_rec(node, visited, nodes):
            if node in visited:
                return

            assert(isinstance(node, Value))
            nodes.append(node)

            for succ in get_value_successors(node):
                dfs_rec(succ, visited, nodes)

        for root in roots:
            dfs_rec(root, visited, nodes)

        return nodes

    def build(self, func: Function):
        self.roots = self.find_roots(func)

        for idx, value in enumerate(self.dfs(self.roots)):
            node = DomTreeNode(value, idx)
            self.nodes.append(node)
            self.value_to_node[value] = node

        self.roots = [self.value_to_node[node] for node in self.roots]

        # https://en.wikipedia.org/wiki/Dominator_(graph_theory)
        # // dominator of the start node is the start itself
        # Dom(n0) = {n0}
        # // for all other nodes, set all nodes as the dominators
        # for each n in N - {n0}
        #     Dom(n) = N;
        # // iteratively eliminate nodes that are not dominators
        # while changes in any Dom(n)
        #     for each n in N - {n0}:
        #         Dom(n) = {n} union with intersection over Dom(p) for all p in pred(n)

        for node in self.roots:
            node.is_changed = False

        iters = 0
        while self.is_nodes_changed():
            iters += 1
            for node in self.nodes:
                if node in self.roots:
                    continue

                preds = [self.value_to_node[value]
                         for value in get_value_predecessors(node.value) if value in self.value_to_node]

                if len(preds) > 0:
                    intersection = preds[0].doms
                    for pred in preds[1:]:
                        intersection = intersection & pred.doms
                else:
                    intersection = set()

                old_doms = set(node.doms)
                node.doms = node.doms | intersection

                node.is_changed = node.doms != old_doms

        for node in sorted(self.nodes, key=lambda x: len(x.doms)):
            for dom in list(node.doms):
                if dom != node:
                    if not dom.parent:
                        dom.parent = node

        for node in self.nodes:
            if node.parent:
                node.parent.children.append(node)

    def dominate(self, value_a, value_b):
        node_a = self.value_to_node[value_a]
        node_b = self.value_to_node[value_b]

        return node_b in node_a.doms


from rachetwrench.codegen.passes import FunctionPass


class DominatorTreePass(FunctionPass):

    def process_function(self, func):
        self.dom_tree = DominatorTree()
        self.dom_tree.build(func)
