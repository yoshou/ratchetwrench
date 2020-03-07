#!/usr/bin/env python
# -*- coding: utf-8 -*-

class ScheduleUnit:
    def __init__(self, node, id):
        self.node = node
        self.id = id
        self.succs = []
        self.preds = []

    def add_pred(self, edge):
        self.preds.append(edge)
        edge.node.succs.append(ScheduleEdge(self))

class ScheduleEdge:
    def __init__(self, node):
        self.node = node

class ScheduleDag:
    def __init__(self, mfunc, dag):
        self.mfunc = mfunc
        self.dag = dag
        self.nodes = []

    def create_sched_node(self, node):
        sched_node = ScheduleUnit(node, len(self.nodes))
        self.nodes.append(sched_node)
        return sched_node

    def build(self):

        def bfs(node, action, visited):
            if node in visited:
                return

            visited.add(node)

            for op in node.operands:
                bfs(op.node, action, visited)

            action(node)

        def all_nodes(root):
            lst = []
            bfs(root, lambda node: lst.append(node), set())
            return lst

        for node in all_nodes(self.mfunc.root):
            pass