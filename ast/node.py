#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections


def flatten(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, collections.Iterable) and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


class Node:
    pass


types = []


def define_node(name, fields, extends_opt=None):
    if name in types:
        raise RuntimeError(f"{name} is already defined.")

    extends = [Node]
    if extends_opt:
        extends.extend(extends_opt)

    extends = tuple(extends)

    types.append(name)

    def init_(self, *args):
        self.vals = list(args)
        self.fields = list(fields)

        assert(len(self.vals) == len(self.fields))

    def dict_(self):
        return {field: val for field, val in zip(fields, self.vals)}

    def getattr_(self, name):
        if name in fields:
            idx = fields.index(name)
            return self.vals[idx]
        else:
            raise AttributeError(
                f'attribute "{name}" is not found in the type "{self.__class__.__name__}".')

    def len_(self):
        return len(fields)

    def getitem_(self, idx):
        return self.vals[idx]

    def setitem_(self, idx, val):
        self.vals[idx] = val

    def str_(self):
        kvs = ", ".join(["{0}={1}".format(k, repr(v))
                         for k, v in zip(fields, self.vals)])
        return "{0}({1})".format(name, kvs)

    def eq_(self, other):
        if isinstance(self, type(other)):
            return self.vals == other.vals
        return False

    def hash_(self):
        return hash(tuple(flatten(self.vals)))

    return type(name, extends,
                {
                    "__getattr__": getattr_,
                    "__init__": init_,
                    "__len__": len_,
                    "__getitem__": getitem_,
                    "__setitem__": setitem_,
                    "__str__": str_,
                    "__repr__": str_,
                    "__eq__": eq_,
                    "__hash__": hash_,
                })


Ident = define_node('Ident', ('val',))
Type = define_node('Type', ('specifier', 'array_specifier'))
FullType = define_node(
    'FullType', ('qualifiers', 'specifier', 'array_specifier'))
StructSpecifier = define_node(
    'StructSpecifier', ('ident', 'decls', 'is_union'))
StructDeclaration = define_node('StructDeclaration', ('type', 'declarators'))
StructDeclarator = define_node('StructDeclarator', ('ident', 'arrspec'))

ArraySpecifier = define_node('ArraySpecifier', ('sizes',))

InitializerList = define_node('InitializerList', ('exprs',))

# declarations
FunctionParam = define_node('FunctionParam', ('type', 'ident', 'qual'))
FunctionProto = define_node('FunctionProto', ('type', 'ident', 'params'))
Function = define_node('Function', ('proto', 'stmts'))

Variable = define_node(
    'Variable', ('type', 'idents'))

EnumDecl = define_node('EnumDecl', ('ident', 'const_decls'))
EnumConstantDecl = define_node('EnumConstantDecl', ('ident', 'value'))

# expressions
FunctionCall = define_node('FunctionCall', ('ident', 'params'))
BinaryOp = define_node('BinaryOp', ('op', 'lhs', 'rhs'))
UnaryOp = define_node('UnaryOp', ('op', 'expr'))
PostOp = define_node('PostOp', ('op', 'expr'))
AccessorOp = define_node('AccessorOp', ('obj', 'field'))
ArrayIndexerOp = define_node('ArrayIndexerOp', ('arr', 'idx'))
IdentExpr = define_node('IdentExpr', ('val',))
IntegerConstantExpr = define_node('IntegerConstantExpr', ('val', 'type'))
FloatingConstantExpr = define_node('FloatingConstantExpr', ('val', 'type'))
CastExpr = define_node('CastExpr', ('expr', 'type'))
ConditionalExpr = define_node(
    'ConditionalExpr', ('cond_expr', 'true_expr', 'false_expr'))
CommaOp = define_node('CommaOp', ('exprs',))
SizeOfExpr = define_node('SizeOfExpr', ('expr', 'type'))
StringLiteralExpr = define_node('StringLiteralExpr', ('val', 'type'))

# statements
CompoundStmt = define_node('CompoundStmt', ('stmts',))
ExprStmt = define_node('ExprStmt', ('expr',))
IfStmt = define_node('IfStmt', ('cond', 'then_stmt', 'else_stmt'))
ForStmt = define_node('ForStmt', ('init', 'cond', 'loop', 'stmt'))
WhileStmt = define_node('WhileStmt', ('cond', 'stmt'))
DoWhileStmt = define_node('DoWhileStmt', ('cond', 'stmt'))
ReturnStmt = define_node('ReturnStmt', ('expr',))
ContinueStmt = define_node('ContinueStmt', ())
BreakStmt = define_node('BreakStmt', ())

SwitchStmt = define_node('SwitchStmt', ('cond', 'stmts'))
CaseLabel = define_node('CaseLabel', ('expr',))
CaseLabelStmt = define_node('CaseLabelStmt', ('expr', 'stmt'))

# for semantic analysis
TypedBinaryOp = define_node('TypedBinaryOp', ('op', 'lhs', 'rhs', 'type'))
TypedUnaryOp = define_node('TypedUnaryOp', ('op', 'expr', 'type'))
TypedConditionalExpr = define_node(
    'TypedConditionalExpr', ('cond_expr', 'true_expr', 'false_expr', 'type'))
TypedPostOp = define_node('TypedPostOp', ('op', 'expr', 'type'))
TypedIdentExpr = define_node('TypedIdentExpr', ('val', 'type'))
TypedAccessorOp = define_node('TypedAccessorOp', ('obj', 'field', 'type'))
TypedArrayIndexerOp = define_node(
    'TypedArrayIndexerOp', ('arr', 'idx', 'type'))
TypedFunctionCall = define_node(
    'TypedFunctionCall', ('ident', 'params', 'type'))
TypedCommaOp = define_node('TypedCommaOp', ('exprs', 'type'))
TypedSizeOfExpr = define_node(
    'TypedSizeOfExpr', ('expr', 'sized_type', 'type'))

TypedInitializerList = define_node('TypedInitializerList', ('exprs', 'type'))


TypedFunctionParam = define_node('TypedFunctionParam', ('type', 'ident'))
TypedFunctionProto = define_node(
    'TypedFunctionProto', ('type', 'ident', 'params', 'specs'))
TypedFunction = define_node('TypedFunction', ('proto', 'params', 'stmts'))

TypedVariable = define_node(
    'TypedVariable', ('type', 'idents', 'storage_class'))


def ident_func(node, *data):
    return node


def traverse_depth(node, enter_func=ident_func, exit_func=ident_func, args=[], depth=0):
    is_node = isinstance(node, (list, tuple, Node))

    enter_func(node, *args)

    if is_node:
        for i in range(len(node)):
            traverse_depth(node[i], enter_func, exit_func, args, depth + 1)

    exit_func(node, *args)


def traverse_depth_update(node, enter_func=ident_func, exit_func=ident_func, data=[], depth=0):
    is_node = isinstance(node, (list, Node))
    if is_node:
        node = enter_func(node, depth, data)

        for i in range(len(node)):
            node[i] = traverse_depth_update(
                node[i], enter_func, exit_func, data, depth + 1)

        node = exit_func(node, depth, data)
    else:
        node = enter_func(node, depth, data)
        node = exit_func(node, depth, data)

    return node
