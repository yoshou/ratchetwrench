#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast.node import *


def print_ast(ast):
    depth = [0]

    def enter_node(node, depth):
        print(('****'*(depth[0])), type(node).__name__)

        depth[0] += 1

    def exit_node(node, depth):
        depth[0] -= 1

    traverse_depth(ast, enter_func=enter_node,
                   exit_func=exit_node, args=(depth,))

    return ast


def print_ast_expr(node):
    if isinstance(node, (Ident, IdentExpr)):
        return node.val
    if isinstance(node, (TypedIdentExpr)):
        return node.val.name
    if isinstance(node, (IntegerConstantExpr, FloatingConstantExpr)):
        return node.val
    if isinstance(node, (BinaryOp, TypedBinaryOp)):
        return f"{print_ast_expr(node.lhs)} {node.op} {print_ast_expr(node.rhs)}"
    if isinstance(node, (PostOp, TypedPostOp)):
        return f"{print_ast_expr(node.expr)}{node.op}"
    if isinstance(node, (UnaryOp, TypedUnaryOp)):
        return f"{node.op}{print_ast_expr(node.expr)}"
    if isinstance(node, (AccessorOp, TypedAccessorOp)):
        return f"{print_ast_expr(node.obj)}.{node.field.val}"
    if isinstance(node, (FunctionCall, TypedFunctionCall)):
        args = [print_ast_expr(arg) for arg in node.params]
        return f"{node.ident.val}({', '.join(args)})"

    print(node)
