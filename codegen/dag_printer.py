#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pygraphviz as pgv
from codegen.dag_builder import *


def print_dag_node(idx, node, g, font_size, font_name):
    rows = []

    if len(node.operands) > 0:
        inputs = "|".join([f"<i{i}>{i}" for i in range(len(node.operands))])
        rows.append(f"{{{inputs}}}")

    opcode = node.get_label().replace("<", "\<").replace(">", "\>")
    rows.append(f"{{{opcode}}}")
    rows.append(f"{{t{idx}}}")

    if len(node.value_types) > 0:
        outputs = "|".join(
            [f"<o{str(i)}>{str(ty)}" for i, ty in enumerate(node.value_types)])
        rows.append(f"{{{outputs}}}")

    label = f"{{{'|'.join(rows)}}}"

    g.add_node(node, label=label, shape='record', style='rounded',
               fontsize=font_size)


import uuid


def print_dag(i, dag: Dag, out_dir, funcname, font_size=14, font_name="Ricty Diminished"):
    g = pgv.AGraph(strict=False, directed=True, rankdir='BT')

    visited = set()

    def dfs(node: DagNode, depth=0):
        if node is None:
            return

        if node in visited:
            return

        for i, operand in enumerate(node.operands):

            if isinstance(operand, DagValue):
                dfs(operand.node, depth + 1)
                if operand.ty.value_type == ValueType.OTHER:
                    g.add_edge(node, operand.node, key=f"ch", headport=f"o{operand.index}", tailport=f"i{i}",
                               fontsize=font_size, fontname=font_name, style="dashed", color="blue")
                elif operand.ty.value_type == ValueType.GLUE:
                    g.add_edge(node, operand.node, key=f"glue", headport=f"o{operand.index}", tailport=f"i{i}",
                               fontsize=font_size, fontname=font_name, style="solid", color="red")
                else:
                    g.add_edge(node, operand.node, key=f"value", headport=f"o{operand.index}", tailport=f"i{i}",
                               fontsize=font_size, fontname=font_name, style="solid", color="black")
            else:
                print(node.opcode, operand)
                raise NotImplementedError

        idx = len(visited)

        print_dag_node(idx, node, g, font_size, font_name)

        visited.add(node)

    dfs(dag.root.node)

    root_id = uuid.uuid4()
    g.add_node(root_id, label="Root", fontsize=font_size, fontname=font_name)
    g.add_edge(root_id, dag.root.node)

    g.draw(os.path.join(
        out_dir, f'{funcname}_dag{i}.png'), prog='dot')
