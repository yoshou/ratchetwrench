#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import traceback
import struct
from enum import Enum, auto

from c.lex import *
from ast.node import *

from c.parse_utils import *


assignment_operator = [
    "=", "*=", "/=", "%=", "+=", "-=", "<=", ">=", "&=", "^=", "|="
]


def parse_primary_expression(tokens, pos, ctx):
    save_pos = []

    # identifier
    if isinstance(tokens[pos], Identifier):
        return (pos + 1, IdentExpr(str(tokens[pos])))

    # constant
    if isinstance(tokens[pos], FloatingConstant):
        t = Type('float', None)
        val = float(str(tokens[pos]))
        return (pos + 1, FloatingConstantExpr(val, t))

    if isinstance(tokens[pos], IntegerConstant):
        val = str(tokens[pos])
        if val.startswith('0x'):
            val = val[2:]

            if len(val) <= 8:
                val = struct.unpack('>i', bytes.fromhex(val))[0]
                t = Type('int', None)
            else:
                raise NotImplementedError()

        else:
            val = int(val)
            t = Type('int', None)
        return (pos + 1, IntegerConstantExpr(val, t))

    # string-literal
    if isinstance(tokens[pos], StringLiteral):
        raise NotImplementedError()

    # ( expression )
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == ")":
                return (pos + 1, expr)
    pos = save_pos.pop()

    # generic-selection
    save_pos.append(pos)
    if str(tokens[pos]) == "_Generic":
        raise NotImplementedError()
    pos = save_pos.pop()

    return (pos, None)


def parse_argument_expression_list(tokens, pos, ctx):
    # assignment-expression
    # expression , assignment-expression
    return parse_list(tokens, pos, ctx, parse_assignment_expression)


def parse_postfix_expression_tail(tokens, pos, ctx, lhs):
    save_pos = []

    # [ expression ]
    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        pos += 1

        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == "]":
                pos += 1
                raise NotImplementedError()
    pos = save_pos.pop()

    # ( argument-expression-list_opt )
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, arg_expr_list) = parse_argument_expression_list(tokens, pos, ctx)

        if str(tokens[pos]) == ")":
            pos += 1
            return (pos, arg_expr_list)
    pos = save_pos.pop()

    # postfix-expression . identifier
    # postfix-expression -> identifier
    save_pos.append(pos)
    if str(tokens[pos]) in [".", "->"]:
        pos += 1
        if isinstance(tokens[pos], Identifier):
            return (pos + 1, AccessorOp(lhs, Ident(str(tokens[pos]))))
    pos = save_pos.pop()

    # ++
    # --
    save_pos.append(pos)
    if str(tokens[pos]) in ["++", "--"]:
        op = str(tokens[pos])
        pos += 1
        return (pos, PostOp(op, lhs))
    pos = save_pos.pop()

    return (pos, None)


def parse_postfix_expression_head(tokens, pos, ctx):
    save_pos = []

    # ( type-name ) { initializer-list }
    # ( type-name ) { initializer-list , }
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, typename) = parse_type_name(tokens, pos, ctx)
        if typename:
            if str(tokens[pos]) == ")":
                pos += 1
                if str(tokens[pos]) == "{":
                    pos += 1
                    (pos, init_list) = parse_initializer_list(tokens, pos, ctx)

                    if str(tokens[pos]) == ",":
                        pos += 1

                    if init_list:
                        if str(tokens[pos]) == "}":
                            pos += 1
                            raise NotImplementedError()
    pos = save_pos.pop()

    # primary-expression
    (pos, expr) = parse_primary_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    return (pos, None)


def parse_postfix_expression(tokens, pos, ctx):
    # primary_expression
    # postfix-expression [ expression ]
    # postfix-expression ( argument-expression-list_opt )
    # postfix-expression . identifier
    # postfix-expression -> identifier
    # postfix-expression ++
    # postfix-expression --
    # ( type-name ) { initializer-list }
    # ( type-name ) { initializer-list , }
    (pos, expr) = parse_postfix_expression_head(tokens, pos, ctx)
    if expr:
        while True:
            (pos, tail) = parse_postfix_expression_tail(tokens, pos, ctx, expr)
            if not tail:
                break

            expr = tail
        return (pos, expr)

    return (pos, None)


def gen_parse_operator_generic(ops, parse_operand):
    def func(tokens, pos, ctx):
        save_pos = []

        save_pos.append(pos)
        (pos, lhs) = parse_operand(tokens, pos, ctx)

        if lhs:
            while str(tokens[pos]) in ops:
                save_pos.append(pos)
                op = str(tokens[pos])
                pos += 1

                (pos, rhs) = parse_operand(tokens, pos, ctx)

                if not rhs:
                    pos = save_pos.pop()
                    break

                lhs = BinaryOp(op, lhs, rhs)

            return (pos, lhs)

        pos = save_pos.pop()

        return (pos, None)

    return func


unary_operator = ['&', '*', '+', '-', '~', '!']


def parse_unary_expression(tokens, pos, ctx):
    save_pos = []

    # ++ unary_expression
    # -- unary_expression
    save_pos.append(pos)
    if str(tokens[pos]) in ["--", "++"]:
        op = str(tokens[pos])
        pos += 1
        (pos, expr) = parse_unary_expression(tokens, pos, ctx)
        if expr:
            return (pos, UnaryOp(op, expr))
    pos = save_pos.pop()

    # postfix_expression
    (pos, expr) = parse_postfix_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    # unary-operator cast-expression
    save_pos.append(pos)
    if str(tokens[pos]) in unary_operator:
        op = str(tokens[pos])
        pos += 1
        (pos, expr) = parse_cast_expression(tokens, pos, ctx)
        if expr:
            return (pos, UnaryOp(op, expr))
    pos = save_pos.pop()

    if str(tokens[pos]) == "sizeof":
        raise NotImplementedError()

    if str(tokens[pos]) == "_Alignof":
        raise NotImplementedError()

    return (pos, None)


def parse_cast_expression(tokens, pos, ctx):
    save_pos = []

    # unary-expression
    (pos, unary_expr) = parse_unary_expression(tokens, pos, ctx)
    if unary_expr:
        return (pos, unary_expr)

    # ( type-name ) cast-expression
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, typename) = parse_type_name(tokens, pos, ctx)
        if typename:
            if str(tokens[pos]) == ")":
                pos += 1
                (pos, expr) = parse_cast_expression(tokens, pos, ctx)
                if expr:
                    return (pos, CastExpr(expr, typename))
    pos = save_pos.pop()

    return (pos, None)


parse_multiplicative_expression = gen_parse_operator_generic(
    ["*", "/", "%"], parse_cast_expression)

parse_additive_expression = gen_parse_operator_generic(
    ["+", "-"], parse_multiplicative_expression)

parse_shift_expression = gen_parse_operator_generic(
    ["<<", ">>"], parse_additive_expression)

parse_relational_expression = gen_parse_operator_generic(
    ["<", ">", "<=", ">="], parse_shift_expression)

parse_equality_expression = gen_parse_operator_generic(
    ["==", "!="], parse_relational_expression)

parse_and_expression = gen_parse_operator_generic(
    ["&"], parse_equality_expression)

parse_exclusive_or_expression = gen_parse_operator_generic(
    ["^"], parse_and_expression)

parse_inclusive_or_expression = gen_parse_operator_generic(
    ["|"], parse_exclusive_or_expression)

parse_logical_and_expression = gen_parse_operator_generic(
    ["&&"], parse_inclusive_or_expression)

parse_logical_or_expression = gen_parse_operator_generic(
    ["||"], parse_logical_and_expression)


def parse_conditional_expression(tokens, pos, ctx):
    save_pos = []

    # logical-OR-expression
    # logical-OR-expression ? expression : conditional-expression
    (pos, cond_expr) = parse_logical_or_expression(tokens, pos, ctx)
    if cond_expr:
        save_pos.append(pos)
        if str(tokens[pos]) == "?":
            pos += 1

            (pos, true_expr) = parse_expression(tokens, pos, ctx)
            if true_expr:
                if str(tokens[pos]) == ":":
                    pos += 1

                    (pos, false_expr) = parse_conditional_expression(
                        tokens, pos, ctx)
                    if false_expr:
                        return (pos, ConditionalExpr(cond_expr, true_expr, false_expr))
        pos = save_pos.pop()

        return (pos, cond_expr)

    return (pos, None)


def parse_assignment_expression(tokens, pos, ctx):
    save_pos = []

    # unary-expression assignment-operator assignment-expression
    save_pos.append(pos)
    (pos, lhs) = parse_unary_expression(tokens, pos, ctx)
    if lhs:
        if str(tokens[pos]) in assignment_operator:
            op = str(tokens[pos])
            pos += 1
            (pos, rhs) = parse_assignment_expression(tokens, pos, ctx)
            if rhs:
                return (pos, BinaryOp(op, lhs, rhs))
    pos = save_pos.pop()

    # conditional-expression
    (pos, expr) = parse_conditional_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    return (pos, None)


def parse_expression(tokens, pos, ctx):
    # assignment-expression
    # expression , assignment-expression
    return parse_list(tokens, pos, ctx, parse_assignment_expression)



def parse_direct_abstract_declarator_head(tokens, pos, ctx):
    save_pos = []
    
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, decl) = parse_abstract_declarator(tokens, pos, ctx)
        if decl:
            if str(tokens[pos]) == ")":
                pos += 1
                return (pos, decl)
    pos = save_pos.pop()

    return (pos, None)

def parse_direct_abstract_declarator_tail(tokens, pos, ctx):
    save_pos = []
        
    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        raise NotImplementedError()
    pos = save_pos.pop()
        
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, param_type_list) = parse_parameter_type_list(tokens, pos, ctx)
        
        if str(tokens[pos]) == ")":
            pos += 1
            return (pos, param_type_list)
    pos = save_pos.pop()

    return (pos, None)

def parse_direct_abstract_declarator(tokens, pos, ctx):
    save_pos = []

    # ( abstract-declarator )
    # direct-abstract-declarator_opt [ type-qualifier-list_opt assignment-expression_opt ]
    # direct-abstract-declarator_opt [ static type-qualifier-list_opt assignment-expression ]
    # direct-abstract-declarator_opt [ type-qualifier-list static assignment-expression ]
    # direct-abstract-declarator_opt [ * ]
    # direct-abstract-declarator_opt ( parameter-type-list_opt )

    save_pos.append(pos)
    (pos, decl) = parse_direct_abstract_declarator_head(tokens, pos, ctx)
    if decl:
        tails = []
        while True:
            (pos, tail) = parse_direct_abstract_declarator_tail(tokens, pos, ctx)

            if not tail:
                break

            tails.append(tail)

        return (pos, (decl, tails))
    pos = save_pos.pop()

    return (pos, None)


def parse_abstract_declarator(tokens, pos, ctx):
    save_pos = []

    # pointer
    # pointer_opt direct-abstract-declarator
    save_pos.append(pos)
    (pos, pointer) = parse_pointer(tokens, pos, ctx)

    (pos, direct_decl) = parse_direct_abstract_declarator(tokens, pos, ctx)
    if direct_decl:
        return (pos, (pointer, direct_decl))

    if pointer:
        return (pos, pointer)

    pos = save_pos.pop()

    return (pos, None)

def parse_parameter_declaration(tokens, pos, ctx):
    # declaration-specifiers declarator
    # declaration-specifiers abstract-declarator_opt

    (pos, specs) = parse_declaration_specifiers(tokens, pos, ctx)
    if specs:
        (pos, decl) = parse_declarator(tokens, pos, ctx)
        if decl:
            return (pos, (specs, decl))

        (pos, decl) = parse_abstract_declarator(tokens, pos, ctx)
        return (pos, (specs, decl))
        

    return (pos, None)


def parse_identifier_list(tokens, pos, ctx):
    ident_list = []
    while True:
        (pos, ident) = parse_identifier(tokens, pos, ctx)
        if not ident:
            break

        ident_list.append(ident)

    if len(ident_list) > 0:
        return (pos, ident_list)

    return (pos, None)


class TypeQualifier(Enum):
    Const = "const"
    Restrict = "restrict"
    Volatile = "volatile"
    Atomic = "atomic"


def parse_type_qualifier(tokens, pos, ctx):
    if str(tokens[pos]) in ["const", "restrict", "volatile", "_Atomic"]:
        value = enum_from_name(TypeQualifier, str(tokens[pos]))
        return (pos + 1, value)

    return (pos, None)


def parse_constant_expression(tokens, pos, ctx):
    return parse_conditional_expression(tokens, pos, ctx)

def parse_designator(tokens, pos, ctx):
    save_pos = []

    # [ constant-expression ]
    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        pos += 1
        (pos, expr) = parse_constant_expression(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == "]":
                pos += 1
                return (pos, expr)
    pos = save_pos.pop()

    # . identifier
    save_pos.append(pos)
    if str(tokens[pos]) == ".":
        pos += 1
        (pos, ident) = parse_identifier(tokens, pos, ctx)
        if ident:
            return (pos, ident)
    pos = save_pos.pop()

    return (pos, None)

def parse_designator_list(tokens, pos, ctx):
    # designator
    # designator-list designator
    designators = []
    while True:
        (pos, designator) = parse_designator(tokens, pos, ctx)
        if not designator:
            break

        designators.append(designator)

    if len(designators) > 0:
        return (pos, designators)

    return (pos, None)
    
def parse_designation(tokens, pos, ctx):
    save_pos = []

    # designator-list =
    save_pos.append(pos)
    (pos, designator_list) = parse_designator_list(tokens, pos, ctx)
    if designator_list:
        if str(tokens[pos]) == "=":
            pos += 1
            return (pos, designator_list)
    pos = save_pos.pop()

    return (pos, None)

def parse_designation_initializer(tokens, pos, ctx):
    (pos, designation) = parse_designation(tokens, pos, ctx)

    (pos, init) = parse_initializer(tokens, pos, ctx)
    if init:
        return (pos, (designation, init))

    return (pos, None)

def parse_initializer_list(tokens, pos, ctx):
    # designation_opt initializer
    # initializer-list , designation_opt initializer
    (pos, lst) = parse_list(tokens, pos, ctx, parse_designation_initializer)
    if len(lst) > 0:
        return (pos, InitializerList(lst))
        
    return (pos, None)


def parse_initializer(tokens, pos, ctx):
    save_pos = []

    # assignment-expression
    (pos, expr) = parse_assignment_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    # { initializer-list }
    # { initializer-list , }
    save_pos.append(pos)
    if str(tokens[pos]) == "{":
        pos += 1

        (pos, expr) = parse_initializer_list(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == ",":
                pos += 1

            if str(tokens[pos]) == "}":
                pos += 1
                return (pos, expr)
    pos = save_pos.pop()

    return (pos, None)


def parse_init_declarator(tokens, pos, ctx):
    save_pos = []

    # declarator
    # declarator = initializer
    (pos, decl) = parse_declarator(tokens, pos, ctx)
    if decl:
        save_pos.append(pos)
        if str(tokens[pos]) == "=":
            pos += 1
            (pos, init) = parse_initializer(tokens, pos, ctx)
            if init:
                return (pos, (decl, init))
        pos = save_pos.pop()

        return (pos, (decl, None))

    return (pos, None)

def parse_init_declarator_list(tokens, pos, ctx):
    # init-declarator
    # init-declarator-list , init-declarator
    return parse_list(tokens, pos, ctx, parse_init_declarator)

def parse_struct_or_union(tokens, pos, ctx):
    if str(tokens[pos]) in ["struct", "union"]:
        return (pos + 1, str(tokens[pos]))

    return (pos, None)

def parse_struct_or_union_specifier(tokens, pos, ctx):
    save_pos = []

    # struct-or-union identifier_opt { struct-declaration-list }
    # struct-or-union identifier
    save_pos.append(pos)
    (pos, struct_or_union) = parse_struct_or_union(tokens, pos, ctx)
    if struct_or_union:
        (pos, ident) = parse_identifier(tokens, pos, ctx)
        if ident:
            ctx.typenames.append(ident)

        save_pos.append(pos)
        if str(tokens[pos]) == "{":
            pos += 1
            (pos, decls) = parse_struct_declaration_list(tokens, pos, ctx)
            if decls:
                if str(tokens[pos]) == "}":
                    pos += 1
                    return (pos, StructSpecifier(ident, decls))
        pos = save_pos.pop()

        if ident:
            return (pos, StructSpecifier(ident, None))

    pos = save_pos.pop()

    return (pos, None)


def parse_struct_declaration_list(tokens, pos, ctx):
    # struct-declaration
    # struct-declaration-list struct-declaration
    decls = []
    while True:
        (pos, decl) = parse_struct_declaration(tokens, pos, ctx)
        if not decl:
            break

        decls.append(decl)

    if len(decls) > 0:
        return (pos, decls)

    return (pos, None)


def parse_struct_declaration(tokens, pos, ctx):
    save_pos = []

    # specifier-qualifier-list struct-declarator-list_opt ;
    save_pos.append(pos)
    (pos, spec_quals) = parse_specifier_qualifier_list(tokens, pos, ctx)
    if spec_quals:
        (pos, decls) = parse_struct_declarator_list(tokens, pos, ctx)
        
        if str(tokens[pos]) == ";":
            pos += 1
            return (pos, StructDeclaration(spec_quals, decls))
    pos = save_pos.pop()
    
    # static_assert-declaration
    (pos, decl) = parse_static_assert_declaration(tokens, pos, ctx)
    if decl:
        return (pos, decl)

    return (pos, None)


def parse_struct_declarator_list(tokens, pos, ctx):
    # struct-declarator
    # struct-declarator-list , struct-declarator
    return parse_list(tokens, pos, ctx, parse_struct_declarator)


def parse_struct_declarator(tokens, pos, ctx):
    save_pos = []

    # declarator
    # declarator_opt : constant-expression
    (pos, decl) = parse_declarator(tokens, pos, ctx)
    
    save_pos.append(pos)
    if str(tokens[pos]) == ':':
        pos += 1
        (pos, const) = parse_constant_expression(tokens, pos, ctx)
        if const:
            return (pos, StructDeclarator(decl, None))
    pos = save_pos.pop()

    if decl:
        return (pos, StructDeclarator(decl, None))

    return (pos, None)

def parse_static_assert_declaration(tokens, pos, ctx):
    if str(tokens[pos]) == "_Static_assert":
        raise NotImplementedError()

    return (pos, None)


def parse_declaration(tokens, pos, ctx):
    save_pos = []

    # declaration-specifiers init-declarator-list_opt ;
    save_pos.append(pos)
    (pos, specs) = parse_declaration_specifiers(tokens, pos, ctx)
    if specs:
        (pos, init) = parse_init_declarator_list(tokens, pos, ctx)
        
        if str(tokens[pos]) == ";":
            pos += 1
            return (pos, specs)
    pos = save_pos.pop()

    # static_assert-declaration
    (pos, decl) = parse_static_assert_declaration(tokens, pos, ctx)
    if decl:
        return (pos, decl)

    return (pos, None)


def parse_declaration_statement(tokens, pos, ctx):
    return parse_declaration(tokens, pos, ctx)


def parse_iteration_statement(tokens, pos, ctx):
    save_pos = []

    # while ( expression ) statement
    save_pos.append(pos)
    if str(tokens[pos]) == "while":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, cond) = parse_expression(tokens, pos, ctx)
            if cond:
                if str(tokens[pos]) == ")":
                    pos += 1
                    (pos, stmt) = parse_statement(tokens, pos, ctx)
                    if stmt:
                        return (pos, WhileStmt(cond, stmt))
    pos = save_pos.pop()

    # do statement while ( expression ) ;
    save_pos.append(pos)
    if str(tokens[pos]) == "do":
        pos += 1
        (pos, stmt) = parse_statement(tokens, pos, ctx)
        if stmt:
            if str(tokens[pos]) == "while":
                pos += 1
                if str(tokens[pos]) == "(":
                    pos += 1
                    (pos, cond) = parse_expression(tokens, pos, ctx)
                    if cond:
                        if str(tokens[pos]) == ")":
                            pos += 1
                            if str(tokens[pos]) == ";":
                                pos += 1
                                return (pos, DoWhileStmt(cond, stmt))
    pos = save_pos.pop()

    # for ( expression_opt ; expression_opt ; expression_opt ) statement
    save_pos.append(pos)
    if str(tokens[pos]) == "for":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, init_expr) = parse_expression(tokens, pos, ctx)
            if str(tokens[pos]) == ";":
                pos += 1
                (pos, cond_expr) = parse_expression(tokens, pos, ctx)
                if str(tokens[pos]) == ";":
                    pos += 1
                    (pos, cont_expr) = parse_expression(tokens, pos, ctx)
                    if str(tokens[pos]) == ")":
                        pos += 1
                        (pos, stmt) = parse_statement(tokens, pos, ctx)
                        if stmt:
                            return (pos, ForStmt(init_expr, cond_expr, cont_expr, stmt))
    pos = save_pos.pop()

    # for ( declaration expression_opt ; expression_opt ) statement
    save_pos.append(pos)
    if str(tokens[pos]) == "for":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, init_decl) = parse_declaration(tokens, pos, ctx)
            if init_decl:
                (pos, cond_expr) = parse_expression(tokens, pos, ctx)
                if str(tokens[pos]) == ";":
                    pos += 1
                    (pos, cont_expr) = parse_expression(tokens, pos, ctx)
                    if str(tokens[pos]) == ")":
                        pos += 1
                        (pos, stmt) = parse_statement(tokens, pos, ctx)
                        if stmt:
                            return (pos, ForStmt(init_expr, cond_expr, cont_expr, stmt))
    pos = save_pos.pop()

    return (pos, None)


def parse_jump_statement(tokens, pos, ctx):
    save_pos = []

    # goto identifier ;
    save_pos.append(pos)
    if str(tokens[pos]) == "goto":
        pos += 1
        (pos, ident) = parse_identifier(tokens, pos, ctx)
        if ident:
            if str(tokens[pos]) == ";":
                pos += 1
                raise NotImplementedError()
    pos = save_pos.pop()

    # continue ;
    save_pos.append(pos)
    if str(tokens[pos]) == "continue":
        pos += 1
        if str(tokens[pos]) == ";":
            return (pos + 1, ContinueStmt())
    pos = save_pos.pop()

    # break ;
    save_pos.append(pos)
    if str(tokens[pos]) == "break":
        pos += 1
        if str(tokens[pos]) == ";":
            return (pos + 1, BreakStmt())
    pos = save_pos.pop()

    # return expression_opt ;
    save_pos.append(pos)
    if str(tokens[pos]) == "return":
        pos += 1

        (pos, expr) = parse_expression(tokens, pos, ctx)

        if str(tokens[pos]) == ";":
            pos += 1
            return (pos, ReturnStmt(expr))
    pos = save_pos.pop()

    return (pos, None)

def parse_selection_statement(tokens, pos, ctx):
    save_pos = []

    # if ( expression ) statement
    # if ( expression ) statement else statement
    save_pos.append(pos)
    if str(tokens[pos]) == "if":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, expr) = parse_expression(tokens, pos, ctx)
            if expr:
                if str(tokens[pos]) == ")":
                    pos += 1
                    (pos, stmt) = parse_statement(tokens, pos, ctx)
                    if stmt:
                        save_pos.append(pos)
                        if str(tokens[pos]) == "else":
                            pos += 1
                            (pos, else_stmt) = parse_statement(tokens, pos, ctx)
                            if else_stmt:
                                return (pos, IfStmt(expr, stmt, else_stmt))
                        pos = save_pos.pop()

                        return (pos, IfStmt(expr, stmt))
    pos = save_pos.pop()

    # switch ( expression ) statement
    save_pos.append(pos)
    if str(tokens[pos]) == "switch":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, expr) = parse_expression(tokens, pos, ctx)
            if expr:
                if str(tokens[pos]) == ")":
                    pos += 1
                    (pos, stmt) = parse_statement(tokens, pos, ctx)
                    if stmt:
                        return (pos, IfStmt(expr, stmt))
    pos = save_pos.pop()

    return (pos, None)


def compute_next_source_pos(span):
    string = span.src[: span.end]

    lines = string.splitlines()

    line = len(lines)
    column = len(lines[-1])

    return (line, column + 1)


def parse_expression_statement(tokens, pos, ctx):
    save_pos = []

    # expression_opt ;
    save_pos.append(pos)
    (pos, expr) = parse_expression(tokens, pos, ctx)
    if str(tokens[pos]) == ";":
        pos += 1
        return (pos, ExprStmt(expr))

    # line, column = compute_next_source_pos(tokens[pos-1].span)

    # print("; is need at line: {0}, column: {1}".format(line, column))
    pos = save_pos.pop()

    return (pos, None)


def parse_labeled_statement(tokens, pos, ctx):
    save_pos = []

    # identifier : statement
    save_pos.append(pos)
    (pos, ident) = parse_identifier(tokens, pos, ctx)
    if ident:
        if str(tokens[pos]) == ":":
            pos += 1
            (pos, stmt) = parse_statement(tokens, pos, ctx)
            if stmt:
                raise NotImplementedError()
    pos = save_pos.pop()

    if str(tokens[pos]) == "case":
        raise NotImplementedError()

    if str(tokens[pos]) == "default":
        raise NotImplementedError()

    return (pos, None)



def parse_statement(tokens, pos, ctx):
    # labeled-statement
    (pos, stmt) = parse_labeled_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    # compound-statement
    (pos, stmt) = parse_compound_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    # expression-statement
    (pos, stmt) = parse_expression_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    # selection-statement
    (pos, stmt) = parse_selection_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    # iteration-statement
    (pos, stmt) = parse_iteration_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    # jump-statement
    (pos, stmt) = parse_jump_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    return (pos, None)

def parse_block_item(tokens, pos, ctx):
    # declaration
    (pos, decl) = parse_declaration(tokens, pos, ctx)
    if decl:
        return (pos, decl)
        
    # statement
    (pos, stmt) = parse_statement(tokens, pos, ctx)
    if stmt:
        return (pos, stmt)

    return (pos, None)


def parse_block_item_list(tokens, pos, ctx):
    # block-item
    # block-item-list block-item
    item_list = []
    while True:
        (pos, item) = parse_block_item(tokens, pos, ctx)
        if not item:
            break

        item_list.append(item)

    if len(item_list) > 0:
        return (pos, item_list)

    return (pos, None)

def parse_compound_statement(tokens, pos, ctx):
    save_pos = []

    # { block-item-list_opt }
    save_pos.append(pos)
    if str(tokens[pos]) == "{":
        pos += 1

        (pos, block_items) = parse_block_item_list(tokens, pos, ctx)
        
        if str(tokens[pos]) == "}":
            pos += 1
            return (pos, CompoundStmt(block_items))
    pos = save_pos.pop()

    return (pos, None)

class StorageClass(Enum):
    Typedef = "typedef"
    Extern = "extern"
    Static = "static"
    ThreadLocal = "_Thread_local"
    Auto = "auto"
    Register = "register"

def enum_from_name(clz, name):
    for member in list(clz):
        if member.value == name:
            return member
    raise ValueError('{} is not a valid name'.format(name))


def parse_storage_class_specifier(tokens, pos, ctx):
    if str(tokens[pos]) in ["typedef"]:
        value = enum_from_name(StorageClass, str(tokens[pos]))
        return (pos + 1, value)

    if str(tokens[pos]) in ["typedef", "extern", "static", "_Thread_local", "auto", "register"]:
        value = enum_from_name(StorageClass, str(tokens[pos]))
        return (pos + 1, value)

    return (pos, None)

def parse_specifier_qualifier(tokens, pos, ctx):
    (pos, specifier) = parse_type_specifier(tokens, pos, ctx)
    if specifier:
        return (pos, specifier)

    (pos, qualifier) = parse_type_qualifier(tokens, pos, ctx)
    if qualifier:
        return (pos, qualifier)

    return (pos, None)

def parse_specifier_qualifier_list(tokens, pos, ctx):
    # type-specifier specifier-qualifier-list_opt
    # type-qualifier specifier-qualifier-list_opt
    spec_quals = []
    while True:
        (pos, spec_qual) = parse_specifier_qualifier(tokens, pos, ctx)
        if not spec_qual:
            break

        spec_quals.append(spec_qual)

    if len(spec_quals) > 0:
        return (pos, spec_quals)

    return (pos, None)


def parse_type_name(tokens, pos, ctx):
    # specifier-qualifier-list abstract-declarator_opt
    (pos, spec_qual_list) = parse_specifier_qualifier_list(tokens, pos, ctx)
    if spec_qual_list:
        (pos, abs_decl) = parse_abstract_declarator(tokens, pos, ctx)
        return (pos, spec_qual_list)

    return (pos, None)

def parse_atomic_type_specifier(tokens, pos, ctx):
    save_pos = []

    # _Atomic ( type-name )
    save_pos.append(pos)
    if str(tokens[pos]) == "_Atomic":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, typename) = parse_type_name(tokens, pos, ctx)
            if typename:
                if str(tokens[pos]) == ")":
                    pos += 1

                    raise NotImplementedError()
    pos = save_pos.pop()

    return (pos, None)

def parse_typedef_name(tokens, pos, ctx):
    # identifier
    if str(tokens[pos]) in ctx.typenames:
        return (pos + 1, str(tokens[pos]))

    return (pos, None)

def parse_enumeration_constant(tokens, pos, ctx):
    # identifier
    return parse_identifier(tokens, pos, ctx)

def parse_enumerator(tokens, pos, ctx):
    save_pos = []

    # enumeration-constant
    # enumeration-constant = constant-expression
    (pos, enum_const) = parse_enumeration_constant(tokens, pos, ctx)
    if enum_const:
        save_pos.append(pos)
        if str(tokens[pos]) == "=":
            pos += 1
            (pos, const) = parse_constant_expression(tokens, pos, ctx)
            if const:
                return (pos, EnumConstantDecl(enum_const, const))
        pos = save_pos.pop()
        
        return (pos, EnumConstantDecl(enum_const, None))
    pos = save_pos.pop()

    return (pos, None)

def parse_enumerator_list(tokens, pos, ctx):
    # enumerator
    # enumerator-list , enumerator
    return parse_list(tokens, pos, ctx, parse_enumerator)

def parse_enum_specifier(tokens, pos, ctx):
    save_pos = []

    # enum identifieropt { enumerator-list }
    # enum identifieropt { enumerator-list , }
    # enum identifier
    save_pos.append(pos)
    if str(tokens[pos]) == "enum":
        pos += 1
        (pos, ident) = parse_identifier(tokens, pos, ctx)

        save_pos.append(pos)
        if str(tokens[pos]) == "{":
            pos += 1
            (pos, enumerators) = parse_enumerator_list(tokens, pos, ctx)
            
            if str(tokens[pos]) == ",":
                pos += 1

            if enumerators:
                if str(tokens[pos]) == "}":
                    pos += 1
                    return (pos, EnumDecl(ident, enumerators))
        pos = save_pos.pop()

        if ident:
            raise NotImplementedError()

    pos = save_pos.pop()

    return (pos, None)

class TypeSpecifierType(Enum):
    Void = "void"
    Char = "char"
    Short = "short"
    Int = "int"
    Long = "long"
    Float = "float"
    Double = "double"
    Signed = "signed"
    Unsigned = "unsigned"
    Bool = "_Bool"
    Complex = "_Complex"

    Struct = "struct"
    Enum = "enum"




def parse_type_specifier(tokens, pos, ctx):
    if str(tokens[pos]) in [
        "void", "char", "short", "int", "long", "float",
        "double", "signed", "unsigned", "_Bool", "_Complex"]:
        value = enum_from_name(TypeSpecifierType, str(tokens[pos]))
        pos += 1
        return (pos, value)

    # atomic-type-specifier
    (pos, spec) = parse_atomic_type_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    # struct-or-union-specifier
    (pos, spec) = parse_struct_or_union_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    # enum-specifier
    (pos, spec) = parse_enum_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    # typedef-name
    (pos, name) = parse_typedef_name(tokens, pos, ctx)
    if name:
        return (pos, name)

    return (pos, None)


def parse_function_specifier(tokens, pos, ctx):
    if str(tokens[pos]) in ["inline", "_Noreturn"]:
        return (pos + 1, str(tokens[pos]))

    return (pos, None)


def parse_alignment_specifier(tokens, pos, ctx):
    if str(tokens[pos]) == "_Alignas":
        raise NotImplementedError()

    return (pos, None)

def parse_declaration_specifier(tokens, pos, ctx):
    # storage-class-specifier declaration-specifiers_opt
    (pos, spec) = parse_storage_class_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    # type-specifier declaration-specifiers_opt
    (pos, spec) = parse_type_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    # type-qualifier declaration-specifiers_opt
    (pos, qual) = parse_type_qualifier(tokens, pos, ctx)
    if qual:
        return (pos, qual)

    # function-specifier declaration-specifiers_opt
    (pos, spec) = parse_function_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    # alignment-specifier declaration-specifiers_opt
    (pos, spec) = parse_alignment_specifier(tokens, pos, ctx)
    if spec:
        return (pos, spec)

    return (pos, None)

class TypeSpecifierWidth(Enum):
    Unspecified = auto()
    Short = auto()
    Long = auto()
    LongLong = auto()


class TypeSpecifierSign(Enum):
    Unspecified = auto()
    Signed = auto()
    Unsigned = auto()


class TypeSpecifierInfo:
    def __init__(self, sign, width, ty):
        self.sign = sign
        self.width = width
        self.ty = ty


def get_type_spec_type(specs):
    names = []
    for spec in specs:
        if isinstance(spec, TypeSpecifierType):
            names.append(spec.value)
        elif isinstance(spec, StructSpecifier):
            names.append("struct")
        elif isinstance(spec, EnumDecl):
            names.append("enum")

    import collections
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    patterns = [
        ("void",),
        ("char",),
        ("signed", "char"),
        ("unsigned", "char"),
        ("short",), ("signed", "short"), ("short", "int"), ("signed", "short", "int"),
        ("unsigned", "short"), ("unsigned", "short", "int"),
        ("int",), ("signed",), ("signed", "int"),
        ("unsigned",), ("unsigned", "int"),
        ("long",), ("signed", "long"), ("long", "int"), ("signed", "long", "int"),
        ("unsigned", "long"), ("unsigned", "long", "int"),
        # §6.7.2 Language 111
        # ISO/IEC 9899:201x Committee Draft - April 12, 2011 N1570
        ("long", "long"), ("signed", "long", "long"), ("long", "long", "int"), ("signed", "long", "long", "int"),
        ("unsigned", "long", "long"), ("unsigned", "long", "long", "int"),
        ("float",),
        ("double",),
        ("long", "double"),
        ("_Bool",),
        ("float", "_Complex"),
        ("double", "_Complex"),
        ("long", "double", "_Complex"),

        ("struct",), ("enum",)
    ]

    for pattern in patterns:
        if compare(pattern, names):
            spec_width = TypeSpecifierWidth.Unspecified
            if collections.Counter(pattern)["short"] >= 1:
                spec_width = TypeSpecifierWidth.Short
            elif collections.Counter(pattern)["long"] >= 2:
                spec_width = TypeSpecifierWidth.LongLong
            elif collections.Counter(pattern)["long"] >= 1:
                spec_width = TypeSpecifierWidth.Long

            spec_sign = TypeSpecifierSign.Unspecified
            if "signed" in pattern:
                spec_sign = TypeSpecifierSign.Signed
            elif "unsigned" in pattern:
                spec_sign = TypeSpecifierSign.Unsigned

            spec_ty = None
            for member in list(TypeSpecifierType):
                if member.value in pattern:
                    spec_ty = member
                    break

            if pattern[0] == "struct":
                spec_ty = TypeSpecifierType.Struct
            elif pattern[0] == "enum":
                spec_ty = TypeSpecifierType.Enum

            assert(spec_ty is not None)

            return TypeSpecifierInfo(spec_sign, spec_width, spec_ty)

    return None


class TypeSpecifier:
    def __init__(self, spec, quals):
        self.spec = spec
        self.quals = quals


def parse_declaration_specifiers(tokens, pos, ctx):
    # storage-class-specifier declaration-specifiers_opt
    # type-specifier declaration-specifiers_opt
    # type-qualifier declaration-specifiers_opt
    # function-specifier declaration-specifiers_opt
    # alignment-specifier declaration-specifiers_opt
    specs = []
    while True:
        (pos, spec) = parse_declaration_specifier(tokens, pos, ctx)
        if not spec:
            break
        
        specs.append(spec)

    if len(specs) > 0:
        type_quals = []
        type_specs = []
        for spec in specs:
            if isinstance(spec, TypeSpecifierType):
                type_specs.append(spec)
            elif isinstance(spec, StructSpecifier):
                type_specs.append(spec)
            elif isinstance(spec, EnumDecl):
                type_specs.append(spec)
            else:
                type_quals.append(spec)

        type_spec_type = get_type_spec_type(type_specs)
        if not type_spec_type:
            raise ValueError()

        return (pos, TypeSpecifier(type_spec_type, type_quals))
        
    return (pos, None)


def parse_type_qualifier_list(tokens, pos, ctx):
    qual_list = []
    while True:
        (pos, qual) = parse_type_qualifier(tokens, pos, ctx)
        if not qual:
            break

        qual_list.append(qual)

    if len(qual_list) > 0:
        return (pos, qual_list)

    return (pos, None)


def parse_pointer(tokens, pos, ctx):
    # * type-qualifier-list_opt
    # * type-qualifier-list_opt pointer

    pointer = []
    while True:
        if str(tokens[pos]) != "*":
            break

        pos += 1
        (pos, qual_list) = parse_type_qualifier_list(tokens, pos, ctx)
        pointer.append(qual_list)

    if len(pointer) > 0:
        return (pos, pointer)

    return (pos, None)

def parse_identifier(tokens, pos, ctx):
    if isinstance(tokens[pos], Identifier):
        return (pos + 1, tokens[pos])

    return (pos, None)

def parse_parameter_list(tokens, pos, ctx):
    # parameter-declaration
    # parameter-list , parameter-declaration
    return parse_list(tokens, pos, ctx, parse_parameter_declaration)

def parse_parameter_type_list(tokens, pos, ctx):
    save_pos = []
        
    # parameter-list
    # parameter-list , ...
    (pos, param_list) = parse_parameter_list(tokens, pos, ctx)
    if param_list:
        save_pos.append(pos)
        if str(tokens[pos]) == ",":
            pos += 1
            if str(tokens[pos]) == "...":
                pos += 1
                return (pos, param_list)
        pos = save_pos.pop()

        return (pos, param_list)

    return (pos, None)


def parse_direct_declarator_head(tokens, pos, ctx):
    save_pos = []
    
    (pos, ident) = parse_identifier(tokens, pos, ctx)
    if ident:
        return (pos, ident)
        
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, decl) = parse_declarator(tokens, pos, ctx)
        if decl:
            if str(tokens[pos]) == ")":
                pos += 1
                return (pos, decl)
    pos = save_pos.pop()

    return (pos, None)

class FunctionDeclaratorChunk:
    def __init__(self, params):
        self.params = params

class ArrayDeclaratorChunk:
    def __init__(self, quals, num_elems, is_static, is_star):
        self.quals = quals
        self.num_elems = num_elems
        self.is_static = is_static
        self.is_star = is_star

def parse_direct_declarator_tail(tokens, pos, ctx):
    save_pos = []
        
    # direct-declarator [ type-qualifier-list_opt assignment-expression_opt ]
    # direct-declarator [ static type-qualifier-list_opt assignment-expression ]
    # direct-declarator [ type-qualifier-list static assignment-expression ]
    # direct-declarator [ type-qualifier-list_opt * ]
    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        pos += 1

        save_pos.append(pos)
        if str(tokens[pos]) == "static":
            pos += 1

            (pos, quals) = parse_type_qualifier_list(tokens, pos, ctx)

            (pos, expr) = parse_assignment_expression(tokens, pos, ctx)
            if expr:
                if str(tokens[pos]) == "]":
                    pos += 1
                    return (pos, ArrayDeclaratorChunk(quals, expr, True, False))
        pos = save_pos.pop()

        (pos, quals) = parse_type_qualifier_list(tokens, pos, ctx)
        
        save_pos.append(pos)
        if quals:
            if str(tokens[pos]) == "static":
                pos += 1

                (pos, expr) = parse_assignment_expression(tokens, pos, ctx)
                if expr:
                    if str(tokens[pos]) == "]":
                        pos += 1
                        return (pos, ArrayDeclaratorChunk(quals, expr, True, False))
        pos = save_pos.pop()

        if str(tokens[pos]) == "*":
            pos += 1
            if str(tokens[pos]) == "]":
                pos += 1
                return (pos, ArrayDeclaratorChunk(quals, None, False, True))

        (pos, expr) = parse_assignment_expression(tokens, pos, ctx)
        
        if str(tokens[pos]) == "]":
            pos += 1
            return (pos, ArrayDeclaratorChunk(quals, expr, False, False))
    pos = save_pos.pop()
        
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, param_type_list) = parse_parameter_type_list(tokens, pos, ctx)
        if param_type_list:
            if str(tokens[pos]) == ")":
                pos += 1
                return (pos, FunctionDeclaratorChunk(param_type_list))
    pos = save_pos.pop()
        
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, ident_list) = parse_identifier_list(tokens, pos, ctx)
        
        if str(tokens[pos]) == ")":
            pos += 1
            return (pos, FunctionDeclaratorChunk(ident_list))
    pos = save_pos.pop()

    return (pos, None)

class Declarator:
    def __init__(self):
        self.pointer = []
        self.ident_or_decl = None
        self.chunks = []

    def add_chunk(self, chunk):
        self.chunks.append(chunk)

    @property
    def is_function_decl(self):
        if not isinstance(self.ident_or_decl, Identifier):
            return False

        if len(self.chunks) != 1 or not isinstance(self.chunks[0], FunctionDeclaratorChunk):
            return False

        return True

    @property
    def is_pointer_decl(self):
        if len(self.pointer) == 0:
            return False

        return True



def parse_direct_declarator(tokens, pos, ctx, declarator):
    save_pos = []

    # identifier
    # ( declarator )
    # direct-declarator [ type-qualifier-list_opt assignment-expression_opt ]
    # direct-declarator [ static type-qualifier-list_opt assignment-expression ]
    # direct-declarator [ type-qualifier-list static assignment-expression ]
    # direct-declarator [ type-qualifier-list_opt * ]
    # direct-declarator ( parameter-type-list )
    # direct-declarator ( identifier-list_opt )

    save_pos.append(pos)
    (pos, ident) = parse_direct_declarator_head(tokens, pos, ctx)
    declarator.ident_or_decl = ident
    if ident:
        tails = []
        while True:
            (pos, tail) = parse_direct_declarator_tail(tokens, pos, ctx)
            if not tail:
                break

            declarator.add_chunk(tail)
            tails.append(tail)

        return (pos, declarator)
    pos = save_pos.pop()

    return (pos, None)


def parse_declarator(tokens, pos, ctx):
    save_pos = []

    # pointer_opt direct-declarator
    save_pos.append(pos)
    (pos, pointer) = parse_pointer(tokens, pos, ctx)

    if str(tokens[pos]) == "__cdecl":
        pos += 1

    declarator = Declarator()
    (pos, direct_decl) = parse_direct_declarator(tokens, pos, ctx, declarator)
    if direct_decl:
        if pointer:
            direct_decl.pointer = pointer
        return (pos, declarator)
    pos = save_pos.pop()

    return (pos, None)

def parse_declaration_list(tokens, pos, ctx):
    # declaration
    # declaration-list declaration
    declaration_list = []
    while True:
        (pos, declaration) = parse_declaration(tokens, pos, ctx)
        if not declaration:
            break

        declaration_list.append(declaration)

    if len(declaration_list):
        return (pos, declaration_list)

    return (pos, None)

class FunctionDecl:
    def __init__(self, qual_spec, declarator, decl_list, stmt):
        self.qual_spec = qual_spec
        self.declarator = declarator
        self.decl_list = decl_list
        self.stmt = stmt

def parse_function_definition(tokens, pos, ctx):
    save_pos = []

    # declaration-specifiers declarator declaration-list_opt compound-statement
    save_pos.append(pos)
    (pos, spec) = parse_declaration_specifiers(tokens, pos, ctx)
    if spec:
        (pos, declarator) = parse_declarator(tokens, pos, ctx)
        if declarator:
            (pos, decl_list) = parse_declaration_list(tokens, pos, ctx)

            (pos, stmt) = parse_compound_statement(tokens, pos, ctx)
            if stmt:
                return (pos, FunctionDecl(spec, declarator, decl_list, stmt))
    pos = save_pos.pop()

    return (pos, None)


def parse_external_declaration(tokens, pos, ctx):
    if pos >= len(tokens):
        return (pos, None)

    # function-definition
    (pos, decl) = parse_function_definition(tokens, pos, ctx)
    if decl:
        return (pos, decl)

    # declaration
    (pos, decl) = parse_declaration(tokens, pos, ctx)
    if decl:
        return (pos, decl)

    return (pos, None)


def parse_translation_unit(tokens, pos, ctx):
    # external-declaration
    # translation-unit external-declaration
    decls = []
    while True:
        (pos, decl) = parse_external_declaration(tokens, pos, ctx)
        if not decl:
            break

        decls.append(decl)

    if len(decls) > 0:
        return (pos, decls)

    return (pos, None)


class Context:
    def __init__(self):
        self.typenames = []

    def is_typename(self, value):
        return value in self.typenames


def parse(tokens):
    ctx = Context()
    ctx.typenames.extend([
        "__int8", "__int16", "__int32", "__int64"
    ])
    pos = 0
    (pos, node) = parse_translation_unit(tokens, pos, ctx)
    assert(pos == len(tokens))
    return node
