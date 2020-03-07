#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import html
from enum import Enum
from collections import namedtuple
import pygraphviz as pgv

from ir.values import *
from ir.printer import print_inst, SlotTracker


class CFGNode:
    def __init__(self, block):
        self.block = block

    @property
    def successors(self):
        if len(self.block.insts) == 0:
            return []

        term = self.block.insts[-1]

        if isinstance(term, BranchInst):
            return [
                CFGNodeEdge(self, CFGNode(term.then_target),
                            CFGNodeEdgeType.COND_THEN),
                CFGNodeEdge(self, CFGNode(term.else_target),
                            CFGNodeEdgeType.COND_ELSE)
            ]

        if isinstance(term, JumpInst):
            return [
                CFGNodeEdge(self, CFGNode(term.goto_target),
                            CFGNodeEdgeType.JUMP),
            ]

        if isinstance(term, ReturnInst):
            return []

        if isinstance(term, BranchPredInst):
            return [
                CFGNodeEdge(self, CFGNode(term.then_target),
                            CFGNodeEdgeType.COND_THEN),
                CFGNodeEdge(self, CFGNode(term.else_target),
                            CFGNodeEdgeType.COND_ELSE)
            ]

        return []

    @property
    def insts(self):
        return self.block.insts

    def __hash__(self):
        return hash(self.block)

    def __eq__(self, other):
        if not isinstance(other, (CFGNode, BasicBlock)):
            return False

        if isinstance(other, BasicBlock):
            return self.block == other

        return self.block == other.block

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.block)


class CFGNodeEdgeType(Enum):
    COND_THEN = "then"
    COND_ELSE = "else"
    JUMP = "jump"
    NORMAL = ""


class CFGNodeEdge:
    def __init__(self, node_from, node_to, edge_type):
        self.node_from = node_from
        self.node_to = node_to
        self.edge_type = edge_type


class CFGFunc:
    def __init__(self, name, return_ty, args, blocks):
        self.blocks = blocks
        self.name = name
        self.return_ty = return_ty
        self.args = args

    @property
    def start(self):
        return self.blocks[0]


# def eliminate_empty_node(head):
#     class GraphToList:
#         def __init__(self):
#             self.nodes = []

#         def visit(self, node):
#             self.nodes.append(node)

#     func = GraphToList()
#     cfg_traverse_depth(head, func.visit, set())

#     for node in func.nodes:
#         if node.node_type == CFGNodeType.basic_block and len(node.insts) == 0:
#             for edge in node.outgoing:
#                 if edge.edge_type == CFGNodeEdgeType.NORMAL:
#                     # eliminate
#                     node_to = edge.node_to
#                     node_to.incoming.remove(edge)

#                     for incoming_edge in node.incoming:
#                         incoming_edge.node_to = node_to
#                         node_to.incoming.append(incoming_edge)


def build_cfg(module):
    cfg_funcs = []
    for func in module.funcs:
        blocks = [CFGNode(bb) for bb in func.bbs]
        cfg_funcs.append(CFGFunc(func.name, func.return_ty, func.args, blocks))

    return cfg_funcs


def cfg_traverse_depth(node, action, visited):
    if node.block in visited:
        return

    visited.add(node.block)

    action(node)

    # print([type(inst) for inst in node.block.insts])
    for link in node.successors:
        cfg_traverse_depth(link.node_to, action, visited)


def gen_block_label(block, slot_id_map):

    if len(block.insts) == 0:
        return '<<table border="0" cellborder="1" cellspacing="0"><tr><td align="left">end</td></tr></table>>'

    text = "".join([
        f'<tr><td align="left">{html.escape(print_inst(inst, slot_id_map))}</td></tr>' for inst in block.insts])

    return f'<<table border="0" cellborder="1" cellspacing="0">{text}</table>>'


def gen_edge_label(edge):
    if edge.edge_type == CFGNodeEdgeType.JUMP:
        return ""
    return edge.edge_type.value


def print_cfg(funcs, out_dir, font_size=10, font_name="Ricty Diminished"):

    class GraphToList:
        def __init__(self):
            self.nodes = []

        def visit(self, node):
            self.nodes.append(node)

    for func in funcs:
        g = pgv.AGraph(directed=True)

        lst = GraphToList()
        cfg_traverse_depth(func.start, lst.visit, set())

        slot_id_map = SlotTracker()
        slot_id_map.track(func)

        for node in lst.nodes:
            g.add_node(node, label=gen_block_label(node, slot_id_map),
                       shape="plaintext", fontsize=font_size, fontname=font_name)

        for node in lst.nodes:
            for edge in node.successors:
                g.add_edge(edge.node_from, edge.node_to,
                           label=gen_edge_label(edge), fontsize=font_size, fontname=font_name)

        g.draw(os.path.join(
            out_dir, f'{func.name}.png'), prog='dot')
