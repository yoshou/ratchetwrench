#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import traceback
import struct

from glsl.lex import tokenize, Identifier, IntegerConstant, FloatingConstant
from ast.node import *


"""
The OpenGL® Shading Language, Version 4.60.5
"""


def parse_precision_qualifier(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    if str(tokens[pos]) in ["mediump", "lowp", "highp"]:
        return (pos + 1, (str(tokens[pos])))
    pos = save_pos.pop()

    return (pos, None)


def parse_precise_qualifier(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    if str(tokens[pos]) in ["precision"]:
        return (pos + 1, (str(tokens[pos])))
    pos = save_pos.pop()

    return (pos, None)


def parse_parameter_declarator(tokens, pos, ctx):
    save_pos = []

    # type_specifier IDENTIFIER
    # type_specifier IDENTIFIER array_specifier
    save_pos.append(pos)
    (pos, s) = parse_type_specifier(tokens, pos, ctx)
    if s is not None:
        if isinstance(tokens[pos], Identifier):
            ident = Ident(str(tokens[pos]))
            pos += 1

            (pos, array_specifier) = parse_array_specifier(tokens, pos, ctx)
            if array_specifier is not None:
                raise NotImplementedError()

            return (pos, (s, ident))
    pos = save_pos.pop()

    return (pos, None)


def parse_parameter_type_specifier(tokens, pos, ctx):
    return parse_type_specifier(tokens, pos, ctx)


def parse_parameter_declaration(tokens, pos, ctx):
    save_pos = []

    # type_qualifier
    save_pos.append(pos)
    (pos, qual) = parse_type_qualifier(tokens, pos, ctx)

    # parameter_declarator
    save_pos.append(pos)
    (pos, decl) = parse_parameter_declarator(tokens, pos, ctx)
    if decl is not None:
        return (pos, FunctionParam(*decl, qual))
    pos = save_pos.pop()

    # parameter_type_specifier
    save_pos.append(pos)
    (pos, s) = parse_parameter_type_specifier(tokens, pos, ctx)
    if s is not None:
        return (pos, FunctionParam(s, None, qual))
    pos = save_pos.pop()

    pos = save_pos.pop()

    return (pos, None)


def parse_function_header(tokens, pos, ctx):
    # p.202
    save_pos = []

    # fully_specified_type IDENTIFIER LEFT_PAREN
    save_pos.append(pos)
    (pos, t) = parse_fully_specified_type(tokens, pos, ctx)
    if t is not None:
        if isinstance(tokens[pos], Identifier):
            ident = Ident(str(tokens[pos]))
            pos += 1
            if str(tokens[pos]) == "(":
                return (pos + 1, (t, ident))
    pos = save_pos.pop()

    return (pos, None)


def parse_function_declarator(tokens, pos, ctx):
    # p.202
    save_pos = []

    # fully_specified_type IDENTIFIER LEFT_PAREN
    save_pos.append(pos)
    (pos, header) = parse_function_header(tokens, pos, ctx)
    if header is not None:
        (pos, param) = parse_parameter_declaration(tokens, pos, ctx)
        params = []
        if param is not None:
            params.append(param)
            while str(tokens[pos]) == ",":
                pos += 1
                (pos, param) = parse_parameter_declaration(tokens, pos, ctx)
                if param is None:
                    raise Exception
                params.append(param)

        return (pos, FunctionProto(header[0], header[1], params))

    pos = save_pos.pop()

    return (pos, None)


def parse_function_prototype(tokens, pos, ctx):
    # p.202
    save_pos = []

    # function_declarator RIGHT_PAREN
    save_pos.append(pos)
    (pos, decl) = parse_function_declarator(tokens, pos, ctx)
    if decl is not None:
        if str(tokens[pos]) == ")":
            return (pos + 1, (decl))

        raise Exception

    pos = save_pos.pop()

    return (pos, None)


def parse_identifier_list(tokens, pos, ctx):
    pass


def parse_storage_qualifier(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    if str(tokens[pos]) in ["const", "in", "out", "inout", "centroid", "patch", "sample", "uniform", "buffer", "shared", "coherent", "volatile", "restrict", "readonly", "writeonly"]:
        return (pos + 1, (str(tokens[pos])))
    pos = save_pos.pop()

    save_pos.append(pos)
    if str(tokens[pos]) == "subroutine":
        token1 = str(tokens[pos])
        pos += 1

        if str(tokens[pos]) == "(":
            # TODO: Not implemented.
            assert(False)

        return (pos, (token1))
    pos = save_pos.pop()

    return (pos, None)


def parse_single_type_qualifier(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, q) = parse_storage_qualifier(tokens, pos, ctx)
    if q is not None:
        return (pos, q)
    pos = save_pos.pop()

    save_pos.append(pos)
    (pos, q) = parse_precision_qualifier(tokens, pos, ctx)
    if q is not None:
        return (pos, q)
    pos = save_pos.pop()

    save_pos.append(pos)
    (pos, q) = parse_precise_qualifier(tokens, pos, ctx)
    if q is not None:
        return (pos, q)
    pos = save_pos.pop()

    return (pos, None)


def parse_type_qualifier(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, q) = parse_single_type_qualifier(tokens, pos, ctx)
    qualifiers = []
    while q:
        qualifiers.append(q)
        (pos, q) = parse_single_type_qualifier(tokens, pos, ctx)

    if len(qualifiers) > 0:
        return (pos, qualifiers)
    pos = save_pos.pop()

    return (pos, None)


buildin_types = [
    "VOID",
    "FLOAT",
    "DOUBLE",
    "INT",
    "UINT",
    "BOOL",
    "VEC2",
    "VEC3",
    "VEC4",
    "DVEC2",
    "DVEC3",
    "DVEC4",
    "BVEC2",
    "BVEC3",
    "BVEC4",
    "IVEC2",
    "IVEC3",
    "IVEC4",
    "UVEC2",
    "UVEC3",
    "UVEC4",
    "MAT2",
    "MAT3",
    "MAT4",
    "MAT2X2",
    "MAT2X3",
    "MAT2X4",
    "MAT3X2",
    "MAT3X3",
    "MAT3X4",
    "MAT4X2",
    "MAT4X3",
    "MAT4X4",
    "DMAT2",
    "DMAT3",
    "DMAT4",
    "DMAT2X2",
    "DMAT2X3",
    "DMAT2X4",
    "DMAT3X2",
    "DMAT3X3",
    "DMAT3X4",
    "DMAT4X2",
    "DMAT4X3",
    "DMAT4X4",
    "ATOMIC_UINT",
    "SAMPLER2D",
    "SAMPLER3D",
    "SAMPLERCUBE",
    "SAMPLER2DSHADOW",
    "SAMPLERCUBESHADOW",
    "SAMPLER2DARRAY",
    "SAMPLER2DARRAYSHADOW",
    "SAMPLERCUBEARRAY",
    "SAMPLERCUBEARRAYSHADOW",
    "ISAMPLER2D",
    "ISAMPLER3D",
    "ISAMPLERCUBE",
    "ISAMPLER2DARRAY",
    "ISAMPLERCUBEARRAY",
    "USAMPLER2D",
    "USAMPLER3D",
    "USAMPLERCUBE",
    "USAMPLER2DARRAY",
    "USAMPLERCUBEARRAY",
    "SAMPLER1D",
    "SAMPLER1DSHADOW",
    "SAMPLER1DARRAY",
    "SAMPLER1DARRAYSHADOW",
    "ISAMPLER1D",
    "ISAMPLER1DARRAY",
    "USAMPLER1D",
    "USAMPLER1DARRAY",
    "SAMPLER2DRECT",
    "SAMPLER2DRECTSHADOW",
    "ISAMPLER2DRECT",
    "USAMPLER2DRECT",
    "SAMPLERBUFFER",
    "ISAMPLERBUFFER",
    "USAMPLERBUFFER",
    "SAMPLER2DMS",
    "ISAMPLER2DMS",
    "USAMPLER2DMS",
    "SAMPLER2DMSARRAY",
    "ISAMPLER2DMSARRAY",
    "USAMPLER2DMSARRAY",
    "IMAGE2D",
    "IIMAGE2D",
    "UIMAGE2D",
    "IMAGE3D",
    "IIMAGE3D",
    "UIMAGE3D",
    "IMAGECUBE",
    "IIMAGECUBE",
    "UIMAGECUBE",
    "IMAGEBUFFER",
    "IIMAGEBUFFER",
    "UIMAGEBUFFER",
    "IMAGE1D",
    "IIMAGE1D",
    "UIMAGE1D",
    "IMAGE1DARRAY",
    "IIMAGE1DARRAY",
    "UIMAGE1DARRAY",
    "IMAGE2DRECT",
    "IIMAGE2DRECT",
    "UIMAGE2DRECT",
    "IMAGE2DARRAY",
    "IIMAGE2DARRAY",
    "UIMAGE2DARRAY",
    "IMAGECUBEARRAY",
    "IIMAGECUBEARRAY",
    "UIMAGECUBEARRAY",
    "IMAGE2DMS",
    "IIMAGE2DMS",
    "UIMAGE2DMS",
    "IMAGE2DMSARRAY",
    "IIMAGE2DMSARRAY",
    "UIMAGE2DMSARRAY",
]


def parse_type_specifier_nonarray(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    if str(tokens[pos]).upper() in buildin_types:
        return (pos + 1, str(tokens[pos]))
    pos = save_pos.pop()

    save_pos.append(pos)
    (pos, spec) = parse_struct_specifier(tokens, pos, ctx)
    if spec is not None:
        return (pos, spec)
    pos = save_pos.pop()

    save_pos.append(pos)
    if ctx.is_typename(str(tokens[pos])):
        return (pos + 1, str(tokens[pos]))
    pos = save_pos.pop()

    return (pos, None)


def parse_constant_expression(tokens, pos, ctx):
    return parse_conditional_expression(tokens, pos, ctx)


def parse_array_specifier_part(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        pos += 1
        if str(tokens[pos]) == "]":
            return (pos + 1, [])

        (pos, constant) = parse_constant_expression(tokens, pos, ctx)

        if constant is not None:
            if str(tokens[pos]) == "]":
                return (pos + 1, [constant])

    pos = save_pos.pop()

    return (pos, None)


def parse_array_specifier(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, part) = parse_array_specifier_part(tokens, pos, ctx)
    if part is not None:
        constants = []
        while part is not None:
            save_pos.append(pos)
            constants.extend(part)
            (pos, part) = parse_array_specifier_part(tokens, pos, ctx)

        return (pos, ArraySpecifier(constants))

    pos = save_pos.pop()

    return (pos, None)


def parse_type_specifier(tokens, pos, ctx):
    save_pos = []

    # type_specifier_nonarray
    # type_specifier_nonarray array_specifier
    save_pos.append(pos)
    (pos, spec) = parse_type_specifier_nonarray(tokens, pos, ctx)
    if spec is not None:
        (pos, array_spec) = parse_array_specifier(tokens, pos, ctx)
        if array_spec is not None:
            return (pos, Type(spec, array_spec))
        return (pos, Type(spec, None))
    pos = save_pos.pop()

    return (pos, None)


def parse_initializer_list(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)

    (pos, initializer) = parse_initializer(tokens, pos, ctx)
    initializers = []
    while initializer is not None:
        initializers.append(initializer)

        save_pos.append(pos)
        if str(tokens[pos]) == ",":
            pos += 1
            (pos, initializer) = parse_initializer(tokens, pos, ctx)
            if initializer is not None:
                continue

        initializer = None
        pos = save_pos.pop()

    if len(initializers) > 0:
        return (pos, InitializerList(initializers))

    pos = save_pos.pop()
    return (pos, None)


def parse_initializer(tokens, pos, ctx):
    save_pos = []

    # assignment_expression
    save_pos.append(pos)
    (pos, expr) = parse_assignment_expression(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    # LEFT_BRACE initializer_list RIGHT_BRACE
    # LEFT_BRACE initializer_list COMMA RIGHT_BRACE
    save_pos.append(pos)
    if str(tokens[pos]) == "{":
        pos += 1

        (pos, expr) = parse_initializer_list(tokens, pos, ctx)
        if expr is not None:
            if str(tokens[pos]) == "}":
                return (pos + 1, expr)

    pos = save_pos.pop()

    return (pos, None)


def parse_fully_specified_type(tokens, pos, ctx):
    save_pos = []

    # type_qualifier type_specifier
    save_pos.append(pos)
    (pos, q) = parse_type_qualifier(tokens, pos, ctx)
    if q is not None:
        (pos, s) = parse_type_specifier(tokens, pos, ctx)
        if s is not None:
            return (pos, FullType(q, s.specifier, s.array_specifier))
    pos = save_pos.pop()

    # type_specifier
    save_pos.append(pos)
    (pos, t) = parse_type_specifier(tokens, pos, ctx)
    if t is not None:
        return (pos, t)
    pos = save_pos.pop()

    return (pos, None)


def parse_single_declaration(tokens, pos, ctx):
    # p.202
    save_pos = []

    # fully_specified_type
    save_pos.append(pos)
    (pos, t) = parse_fully_specified_type(tokens, pos, ctx)
    if t is not None:
        # IDENTIFIER
        save_pos.append(pos)
        if isinstance(tokens[pos], Identifier):
            ident = Ident(str(tokens[pos]))
            pos += 1

            # array_specifier
            save_pos.append(pos)
            (pos, array_specifier) = parse_array_specifier(tokens, pos, ctx)
            if array_specifier is None:
                pos = save_pos.pop()

            # EQUAL initializer
            save_pos.append(pos)
            if str(tokens[pos]) == "=":
                pos += 1
                (pos, init) = parse_initializer(tokens, pos, ctx)
                if init is not None:
                    return (pos, Variable(t, [[ident, array_specifier, init]]))
            pos = save_pos.pop()

            return (pos, Variable(t, [[ident, array_specifier, None]]))
        pos = save_pos.pop()

        return (pos, t)
    pos = save_pos.pop()

    return (pos, None)


def parse_init_declarator_list_tail(tokens, pos, ctx):
    # p.202
    save_pos = []

    # single_declaration
    save_pos.append(pos)
    if str(tokens[pos]) == ",":
        pos += 1
        if isinstance(tokens[pos], Identifier):
            ident = Ident(str(tokens[pos]))
            pos += 1
            save_pos.append(pos)
            if str(tokens[pos]) == "=":
                pos += 1
                if isinstance(tokens[pos], Identifier):
                    ident = str(tokens[pos])
                    pos += 1
                    (pos, init) = parse_initializer(tokens, pos, ctx)
                    if init is not None:
                        return (pos, [ident, None, init])
            pos = save_pos.pop()

            return (pos, [ident, None, None])
    pos = save_pos.pop()

    return (pos, None)


def parse_init_declarator_list(tokens, pos, ctx):
    # p.202
    save_pos = []

    # single_declaration
    save_pos.append(pos)
    (pos, decl) = parse_single_declaration(tokens, pos, ctx)
    if decl is not None:
        save_pos.append(pos)
        (pos, tail) = parse_init_declarator_list_tail(tokens, pos, ctx)
        while tail is not None:
            decl.idents.append(tail)
            save_pos.append(pos)
            (pos, tail) = parse_init_declarator_list_tail(tokens, pos, ctx)
        pos, save_pos.pop()

        return (pos, decl)
    pos = save_pos.pop()

    return (pos, None)


def parse_struct_specifier(tokens, pos, ctx):
    # p.207
    save_pos = []

    # STRUCT IDENTIFIER LEFT_BRACE struct_declaration_list RIGHT_BRACE
    # STRUCT LEFT_BRACE struct_declaration_list RIGHT_BRACE
    save_pos.append(pos)
    if str(tokens[pos]) == "struct":
        pos += 1
        if isinstance(tokens[pos], Identifier):
            ident = str(tokens[pos])
            pos += 1

            ctx.typenames.append(ident)
        else:
            ident = ""

        if str(tokens[pos]) == "{":
            pos += 1
            (pos, decls) = parse_struct_declaration_list(tokens, pos, ctx)
            if decls is not None:
                if str(tokens[pos]) == "}":
                    return (pos + 1, StructSpecifier(ident, decls, False))
    pos = save_pos.pop()

    return (pos, None)


def parse_struct_declaration_list(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, decl) = parse_struct_declaration(tokens, pos, ctx)
    decls = []
    while decl is not None:
        decls.append(decl)
        (pos, decl) = parse_struct_declaration(tokens, pos, ctx)
    if len(decls) > 0:
        return (pos, decls)
    pos = save_pos.pop()

    return (pos, None)


def parse_struct_declaration(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, s) = parse_type_specifier(tokens, pos, ctx)
    if s is not None:
        (pos, decls) = parse_struct_declarator_list(tokens, pos, ctx)
        if decls is not None:
            if str(tokens[pos]) == ";":
                pos += 1
                return (pos, StructDeclaration(s, decls))
    pos = save_pos.pop()

    return (pos, None)


def parse_struct_declarator_list(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, decl) = parse_struct_declarator(tokens, pos, ctx)
    decls = []
    while decl is not None:
        decls.append(decl)

        if str(tokens[pos]) == ",":
            pos += 1
            (pos, decl) = parse_struct_declarator(tokens, pos, ctx)
        else:
            (pos, decl) = (pos, None)
    if len(decls) > 0:
        return (pos, decls)
    pos = save_pos.pop()

    return (pos, None)


def parse_struct_declarator(tokens, pos, ctx):
    save_pos = []

    # IDENTIFIER
    # IDENTIFIER array_specifier
    save_pos.append(pos)
    if isinstance(tokens[pos], Identifier):
        ident = str(tokens[pos])
        pos += 1

        (pos, array_specifier) = parse_array_specifier(tokens, pos, ctx)
        if array_specifier is not None:
            raise NotImplementedError()

        return (pos, StructDeclarator(ident, None))
    pos = save_pos.pop()

    return (pos, None)


def parse_declaration(tokens, pos, ctx):
    # p.201
    save_pos = []

    # function_prototype SEMICOLON
    save_pos.append(pos)
    (pos, proto) = parse_function_prototype(tokens, pos, ctx)
    if proto is not None:
        if str(tokens[pos]) == ";":
            pos += 1
            return (pos, proto)
    pos = save_pos.pop()

    # init_declarator_list SEMICOLON
    save_pos.append(pos)
    (pos, decl) = parse_init_declarator_list(tokens, pos, ctx)
    if decl is not None:
        if str(tokens[pos]) == ";":
            pos += 1
            return (pos, decl)
    pos = save_pos.pop()

    return (pos, None)


def parse_declaration_statement(tokens, pos, ctx):
    return parse_declaration(tokens, pos, ctx)


assignment_operator = [
    "=", "*=", "/=", "%=", "+=", "-=", "<=", ">=", "&=", "^=", "|="
]


def parse_primary_expression(tokens, pos, ctx):
    save_pos = []

    if isinstance(tokens[pos], Identifier):
        return (pos + 1, IdentExpr(str(tokens[pos])))

    if isinstance(tokens[pos], FloatingConstant):
        t = Type('float', None)
        if tokens[pos].suffix in ["f", "F"]:
            t = Type('float', None)
        if tokens[pos].suffix in ["lf", "LF"]:
            t = Type('double', None)
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

    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr is not None:
            if str(tokens[pos]) == ")":
                return (pos + 1, expr)

        raise Exception
    pos = save_pos.pop()

    return (pos, None)


def parse_function_identifier_tail(tokens, pos, ctx, lhs):
    save_pos = []

    save_pos.append(pos)
    (pos, expr) = parse_postfix_expression_tail(tokens, pos, ctx, lhs)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    return (pos, lhs)


def parse_function_call_header_tail(tokens, pos, ctx, lhs):
    save_pos = []

    save_pos.append(pos)
    (pos, ident) = parse_function_identifier_tail(tokens, pos, ctx, lhs)
    if ident is not None:
        if str(tokens[pos]) == "(":
            return (pos + 1, ident)
    pos = save_pos.pop()

    return (pos, None)


def parse_function_call_header_no_parameters_tail(tokens, pos, ctx, lhs):
    save_pos = []

    save_pos.append(pos)
    (pos, header) = parse_function_call_header_tail(tokens, pos, ctx, lhs)
    if header is not None:
        if str(tokens[pos]) == "void":
            return (pos + 1, FunctionCall(header, []))

        return (pos, FunctionCall(header, []))
    pos = save_pos.pop()

    return (pos, None)


def parse_function_call_header_with_parameters_tail(tokens, pos, ctx, lhs):
    save_pos = []

    save_pos.append(pos)
    (pos, header) = parse_function_call_header_tail(tokens, pos, ctx, lhs)
    if header is not None:
        (pos, param) = parse_assignment_expression(tokens, pos, ctx)
        params = []
        while param is not None:
            params.append(param)

            if str(tokens[pos]) == ",":
                pos += 1
                (pos, param) = parse_assignment_expression(tokens, pos, ctx)
            else:
                (pos, param) = (pos, None)

        if len(params) > 0:
            return (pos, FunctionCall(header, params))
    pos = save_pos.pop()

    return (pos, None)


def parse_function_call_generic_tail(tokens, pos, ctx, lhs):
    save_pos = []

    save_pos.append(pos)
    (pos, header) = parse_function_call_header_no_parameters_tail(
        tokens, pos, ctx, lhs)
    if header is not None:
        if str(tokens[pos]) == ")":
            return (pos + 1, header)
    pos = save_pos.pop()

    save_pos.append(pos)
    (pos, header) = parse_function_call_header_with_parameters_tail(
        tokens, pos, ctx, lhs)
    if header is not None:
        if str(tokens[pos]) == ")":
            return (pos + 1, header)
    pos = save_pos.pop()

    return (pos, None)


def parse_function_call_tail(tokens, pos, ctx, lhs):
    return parse_function_call_generic_tail(tokens, pos, ctx, lhs)


def parse_postfix_expression_tail(tokens, pos, ctx, lhs):
    save_pos = []

    # DOT FIELD_SELECTION
    save_pos.append(pos)
    if str(tokens[pos]) == ".":
        pos += 1
        if isinstance(tokens[pos], Identifier):
            return (pos + 1, AccessorOp(lhs, Ident(str(tokens[pos]))))
    pos = save_pos.pop()

    # LEFT_BRACKET integer_expression RIGHT_BRACKET
    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        pos += 1
        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == "]":
                pos += 1
                return (pos, ArrayIndexerOp(lhs, expr))
    pos = save_pos.pop()

    # INC_OP
    # DEC_OP
    save_pos.append(pos)
    if str(tokens[pos]) in ["++", "--"]:
        op = str(tokens[pos])
        pos += 1
        return (pos, PostOp(op, lhs))
    pos = save_pos.pop()

    return (pos, None)


def parse_function_call_head(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, s) = parse_type_specifier(tokens, pos, ctx)
    if s is not None:
        return (pos, s)
    pos = save_pos.pop()

    save_pos.append(pos)
    (pos, expr) = parse_postfix_expression_head(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    return (pos, None)


def parse_function_call(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, func_call) = parse_function_call_head(tokens, pos, ctx)
    if func_call is not None:
        save_pos.append(pos)
        (pos, tail) = parse_function_call_tail(tokens, pos, ctx, func_call)
        if tail is not None:
            while tail is not None:
                expr = tail
                save_pos.append(pos)
                (pos, tail) = parse_function_call_tail(
                    tokens, pos, ctx, func_call)
            pos = save_pos.pop()
            return (pos, expr)
    pos = save_pos.pop()

    return (pos, None)


def parse_postfix_expression_head(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, expr) = parse_primary_expression(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    return (pos, None)


def parse_postfix_expression(tokens, pos, ctx):
    save_pos = []

    # primary_expression
    # postfix_expression LEFT_BRACKET integer_expression RIGHT_BRACKET
    # function_call -> function_call_head function_call_tail*
    # postfix_expression DOT FIELD_SELECTION
    # postfix_expression INC_OP
    # postfix_expression DEC_OP
    save_pos.append(pos)
    (pos, expr) = parse_function_call(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    save_pos.append(pos)
    (pos, expr) = parse_postfix_expression_head(tokens, pos, ctx)

    if expr is not None:
        save_pos.append(pos)
        (pos, tail) = parse_postfix_expression_tail(tokens, pos, ctx, expr)
        while tail is not None:
            expr = tail
            save_pos.append(pos)
            (pos, tail) = parse_postfix_expression_tail(tokens, pos, ctx, expr)
        pos = save_pos.pop()
        return (pos, expr)

    pos = save_pos.pop()

    return (pos, None)


def gen_parse_operator_generic(ops, parse_operand):
    def func(tokens, pos, ctx):
        save_pos = []

        save_pos.append(pos)
        (pos, lhs) = parse_operand(tokens, pos, ctx)

        if lhs is not None:
            while str(tokens[pos]) in ops:
                op = str(tokens[pos])
                pos += 1

                (pos, rhs) = parse_operand(tokens, pos, ctx)

                if rhs is None:
                    raise Exception

                lhs = BinaryOp(op, lhs, rhs)

            return (pos, lhs)

        pos = save_pos.pop()

        return (pos, None)

    return func


unary_operator = ['+', '-', '!', '~']


def parse_unary_expression(tokens, pos, ctx):
    save_pos = []

    # INC_OP unary_expression
    # DEC_OP unary_expression
    save_pos.append(pos)
    if str(tokens[pos]) in ["--", "++"]:
        op = str(tokens[pos])
        pos += 1
        (pos, expr) = parse_unary_expression(tokens, pos, ctx)
        if expr is not None:
            return (pos, UnaryOp(op, expr))
    pos = save_pos.pop()

    # postfix_expression
    save_pos.append(pos)
    (pos, expr) = parse_postfix_expression(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    # unary_operator unary_expression
    save_pos.append(pos)
    if str(tokens[pos]) in unary_operator:
        op = str(tokens[pos])
        pos += 1
        (pos, expr) = parse_unary_expression(tokens, pos, ctx)
        if expr is not None:
            return (pos, UnaryOp(op, expr))
    pos = save_pos.pop()

    return (pos, None)


parse_multiplicative_expression = gen_parse_operator_generic(
    ["*", "/", "%"], parse_unary_expression)


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


parse_logical_xor_expression = gen_parse_operator_generic(
    ["^^"], parse_logical_and_expression)


parse_logical_or_expression = gen_parse_operator_generic(
    ["||"], parse_logical_xor_expression)


def parse_conditional_expression(tokens, pos, ctx):
    (pos, expr) = parse_logical_or_expression(tokens, pos, ctx)

    save_pos = []

    # logical_or_expression QUESTION expression COLON assignment_expression
    save_pos.append(pos)
    if str(tokens[pos]) == "?":
        pos += 1
        (pos, true_expr) = parse_expression(tokens, pos, ctx)
        if true_expr:
            if str(tokens[pos]) == ":":
                pos += 1
                (pos, false_expr) = parse_assignment_expression(tokens, pos, ctx)
                if false_expr:
                    return (pos, ConditionalExpr(expr, true_expr, false_expr))

    pos = save_pos.pop()

    return (pos, expr)


def parse_assignment_expression(tokens, pos, ctx):
    save_pos = []

    # unary_expression assignment_operator assignment_expression
    save_pos.append(pos)
    (pos, left) = parse_unary_expression(tokens, pos, ctx)
    if left is not None:
        if str(tokens[pos]) in assignment_operator:
            op = str(tokens[pos])
            pos += 1
            (pos, right) = parse_assignment_expression(tokens, pos, ctx)
            if right is not None:
                return (pos, BinaryOp(op, left, right))
    pos = save_pos.pop()

    # conditional_expression
    save_pos.append(pos)
    (pos, expr) = parse_conditional_expression(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    return (pos, None)


def parse_expression(tokens, pos, ctx):
    return parse_assignment_expression(tokens, pos, ctx)


def parse_condition(tokens, pos, ctx):
    save_pos = []

    # expression
    save_pos.append(pos)
    (pos, expr) = parse_expression(tokens, pos, ctx)
    if expr is not None:
        return (pos, expr)
    pos = save_pos.pop()

    # fully_specified_type IDENTIFIER EQUAL initializer
    # TODO: Not implemented

    return (pos, None)


def parse_for_init_statement(tokens, pos, ctx):
    save_pos = []

    # declaration_statement
    save_pos.append(pos)
    (pos, stmt) = parse_declaration_statement(tokens, pos, ctx)
    if stmt is not None:
        return (pos, stmt)
    pos = save_pos.pop()

    # expression_statement
    save_pos.append(pos)
    (pos, stmt) = parse_expression_statement(tokens, pos, ctx)
    if stmt is not None:
        return (pos, stmt)
    pos = save_pos.pop()

    return (pos, None)


def parse_conditionopt(tokens, pos, ctx):
    save_pos = []

    # condition
    save_pos.append(pos)
    (pos, cond) = parse_condition(tokens, pos, ctx)
    if cond is not None:
        return (pos, cond)
    pos = save_pos.pop()

    # empty
    if True:
        return (pos, ((),))

    return (pos, None)


def parse_for_rest_statement(tokens, pos, ctx):
    save_pos = []

    # conditionopt SEMICOLON
    # conditionopt SEMICOLON expression
    save_pos.append(pos)
    (pos, cond) = parse_conditionopt(tokens, pos, ctx)
    if cond is not None:
        if str(tokens[pos]) == ";":
            pos += 1
            (pos, expr) = parse_expression(tokens, pos, ctx)
            if expr is not None:
                return (pos, (cond, expr))

            return (pos, (cond, ((),)))

        raise Exception

    pos = save_pos.pop()

    return (pos, None)


def parse_statement_no_new_scope(tokens, pos, ctx):
    save_pos = []
    save_pos.append(pos)

    # compound_statement
    save_pos.append(pos)
    (pos, stmt) = parse_compound_statement(tokens, pos, ctx)
    if stmt is not None:
        return (pos, stmt)
    pos = save_pos.pop()

    # simple_statement
    save_pos.append(pos)
    (pos, stmt) = parse_simple_statement(tokens, pos, ctx)
    if stmt is not None:
        return (pos, stmt)
    pos = save_pos.pop()

    return (pos, None)


def parse_iteration_statement(tokens, pos, ctx):
    save_pos = []

    # WHILE LEFT_PAREN condition RIGHT_PAREN statement_no_new_scope
    save_pos.append(pos)
    if str(tokens[pos]) == "while":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, cond) = parse_condition(tokens, pos, ctx)
            if cond is not None:
                if str(tokens[pos]) == ")":
                    pos += 1
                    (pos, stmt) = parse_statement_no_new_scope(
                        tokens, pos, ctx)
                    if stmt is not None:
                        return (pos, WhileStmt(cond, stmt))

    pos = save_pos.pop()

    # DO statement WHILE LEFT_PAREN expression RIGHT_PAREN SEMICOLON
    save_pos.append(pos)
    if str(tokens[pos]) == "do":
        pos += 1
        (pos, stmt) = parse_statement(tokens, pos, ctx)
        if stmt is not None:
            if str(tokens[pos]) == "while":
                pos += 1
                if str(tokens[pos]) == "(":
                    pos += 1
                    (pos, cond) = parse_expression(tokens, pos, ctx)
                    if cond is not None:
                        if str(tokens[pos]) == ")":
                            pos += 1
                            if str(tokens[pos]) == ";":
                                pos += 1
                                return (pos, DoWhileStmt(cond, stmt))

        raise Exception()

    pos = save_pos.pop()

    # FOR LEFT_PAREN for_init_statement for_rest_statement RIGHT_PAREN statement_no_new_scope
    save_pos.append(pos)
    if str(tokens[pos]) == "for":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, init_stmt) = parse_for_init_statement(tokens, pos, ctx)
            if init_stmt is not None:
                (pos, rest_stmt) = parse_for_rest_statement(
                    tokens, pos, ctx)
                if rest_stmt is not None:
                    if str(tokens[pos]) == ")":
                        pos += 1
                        (pos, stmt) = parse_statement_no_new_scope(
                            tokens, pos, ctx)
                        if stmt is not None:
                            return (pos, ForStmt(init_stmt, *rest_stmt, stmt))

    pos = save_pos.pop()
    return (pos, None)


def parse_jump_statement(tokens, pos, ctx):
    save_pos = []

    # CONTINUE SEMICOLON
    save_pos.append(pos)
    if str(tokens[pos]) == "continue":
        pos += 1
        if str(tokens[pos]) == ";":
            return (pos + 1, ContinueStmt())
    pos = save_pos.pop()

    # BREAK SEMICOLON
    save_pos.append(pos)
    if str(tokens[pos]) == "break":
        pos += 1
        if str(tokens[pos]) == ";":
            return (pos + 1, BreakStmt())
    pos = save_pos.pop()

    # RETURN SEMICOLON
    # RETURN expression SEMICOLON
    save_pos.append(pos)
    if str(tokens[pos]) == "return":
        pos += 1
        if str(tokens[pos]) == ";":
            return (pos + 1, ReturnStmt(None))

        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr is not None:
            if str(tokens[pos]) == ";":
                return (pos + 1, ReturnStmt(expr))
    pos = save_pos.pop()

    # DISCARD SEMICOLON // Fragment shader only
    # Not supporting.
    if str(tokens[pos]) == "discard":
        raise NotImplementedError()

    return (pos, None)


def parse_selection_rest_statement(tokens, pos, ctx):
    save_pos = []

    # statement ELSE statement
    # statement
    save_pos.append(pos)
    (pos, stmt1) = parse_statement(tokens, pos, ctx)
    if stmt1 is not None:
        save_pos.append(pos)
        if str(tokens[pos]) == "else":
            pos += 1
            (pos, stmt2) = parse_statement(tokens, pos, ctx)
            if stmt2 is not None:
                return (pos, (stmt1, stmt2))
        pos = save_pos.pop()
        return (pos, (stmt1, CompoundStmt([])))
    pos = save_pos.pop()

    return (pos, None)


def parse_selection_statement(tokens, pos, ctx):
    save_pos = []

    # IF LEFT_PAREN expression RIGHT_PAREN selection_rest_statement
    save_pos.append(pos)
    if str(tokens[pos]) == "if":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, expr) = parse_expression(tokens, pos, ctx)
            if expr is not None:
                if str(tokens[pos]) == ")":
                    pos += 1
                    (pos, stmt) = parse_selection_rest_statement(
                        tokens, pos, ctx)
                    if stmt is not None:
                        return (pos, IfStmt(expr, *stmt))

                    raise Exception

    pos = save_pos.pop()
    return (pos, None)


def parse_switch_statement_list(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, stmts) = parse_statement_list(tokens, pos, ctx)

    return (pos, (stmts))


def parse_switch_statement(tokens, pos, ctx):
    save_pos = []

    # SWITCH LEFT_PAREN expression RIGHT_PAREN LEFT_BRACE switch_statement_list RIGHT_BRACE
    save_pos.append(pos)
    if str(tokens[pos]) == "switch":
        pos += 1
        if str(tokens[pos]) == "(":
            pos += 1
            (pos, expr) = parse_expression(tokens, pos, ctx)
            if expr is not None:
                if str(tokens[pos]) == ")":
                    pos += 1
                    if str(tokens[pos]) == "{":
                        pos += 1
                        (pos, stmt) = parse_switch_statement_list(
                            tokens, pos, ctx)
                        if stmt is not None:
                            if str(tokens[pos]) == "}":
                                pos += 1
                                return (pos, SwitchStmt(expr, stmt))

        raise Exception

    pos = save_pos.pop()
    return (pos, None)


def parse_case_label(tokens, pos, ctx):
    save_pos = []

    # CASE expression COLON
    save_pos.append(pos)
    if str(tokens[pos]) == "case":
        pos += 1
        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr is not None:
            if str(tokens[pos]) == ":":
                pos += 1
                return (pos, CaseLabel(expr))

        raise Exception

    pos = save_pos.pop()

    # DEFAULT COLON
    save_pos.append(pos)
    if str(tokens[pos]) == "default":
        pos += 1
        if str(tokens[pos]) == ":":
            pos += 1
            return (pos, CaseLabel(None))

        raise Exception

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

    # SEMICOLON
    if str(tokens[pos]) == ";":
        return (pos + 1, ((),))

    # expression SEMICOLON
    save_pos.append(pos)
    (pos, expr) = parse_expression(tokens, pos, ctx)
    if expr is not None:
        if str(tokens[pos]) == ";":
            return (pos + 1, ExprStmt(expr))

        line, column = compute_next_source_pos(tokens[pos-1].span)

        print("; is need at line: {0}, column: {1}".format(line, column))
        raise Exception
    pos = save_pos.pop()

    return (pos, None)


def parse_simple_statement(tokens, pos, ctx):
    save_pos = []

    # declaration_statement
    save_pos.append(pos)
    (pos, stmts) = parse_declaration_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # expression_statement
    save_pos.append(pos)
    (pos, stmts) = parse_expression_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # selection_statement
    save_pos.append(pos)
    (pos, stmts) = parse_selection_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # switch_statement
    save_pos.append(pos)
    (pos, stmts) = parse_switch_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # case_label
    save_pos.append(pos)
    (pos, stmts) = parse_case_label(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # iteration_statement
    save_pos.append(pos)
    (pos, stmts) = parse_iteration_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # jump_statement
    save_pos.append(pos)
    (pos, stmts) = parse_jump_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    return (pos, None)


def parse_statement(tokens, pos, ctx):
    save_pos = []

    # simple_statement
    save_pos.append(pos)
    (pos, stmts) = parse_simple_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    # compound_statement
    save_pos.append(pos)
    (pos, stmts) = parse_compound_statement(tokens, pos, ctx)
    if stmts is not None:
        return (pos, stmts)
    pos = save_pos.pop()

    return (pos, None)


def parse_statement_list(tokens, pos, ctx):
    save_pos = []

    save_pos.append(pos)
    (pos, stmt) = parse_statement(tokens, pos, ctx)

    stmts = []

    while stmt is not None:
        stmts.append(stmt)
        save_pos.append(pos)
        (pos, stmt) = parse_statement(tokens, pos, ctx)
    if len(stmts) > 0:
        return (pos, (stmts))

    pos = save_pos.pop()

    return (pos, None)


def parse_compound_statement(tokens, pos, ctx):
    save_pos = []

    # LEFT_BRACE RIGHT_BRACE
    # LEFT_BRACE statement_list RIGHT_BRACE
    save_pos.append(pos)
    if str(tokens[pos]) == "{":
        pos += 1
        if str(tokens[pos]) == "}":
            return (pos + 1, ((),))

        (pos, stmts) = parse_statement_list(tokens, pos, ctx)
        if stmts is not None:
            if str(tokens[pos]) == "}":
                return (pos + 1, CompoundStmt(stmts))

    pos = save_pos.pop()

    return (pos, None)


def parse_compound_statement_no_new_scope(tokens, pos, ctx):
    save_pos = []

    # LEFT_BRACE RIGHT_BRACE
    # LEFT_BRACE statement_list RIGHT_BRACE
    save_pos.append(pos)
    if str(tokens[pos]) == "{":
        pos += 1
        if str(tokens[pos]) == "}":
            return (pos + 1, [])

        (pos, stmts) = parse_statement_list(tokens, pos, ctx)
        if stmts is not None:
            if str(tokens[pos]) == "}":
                return (pos + 1, stmts)

    pos = save_pos.pop()

    return (pos, None)


def parse_function_definition(tokens, pos, ctx):
    save_pos = []

    # function_prototype compound_statement_no_new_scope
    save_pos.append(pos)
    (pos, proto) = parse_function_prototype(tokens, pos, ctx)
    if proto is not None:
        (pos, stmt) = parse_compound_statement_no_new_scope(tokens, pos, ctx)
        if stmt is not None:
            return (pos, Function(proto, stmt))
    pos = save_pos.pop()

    return (pos, None)


def parse_external_declaration(tokens, pos, ctx):
    if pos >= len(tokens):
        return (pos, None)

    # p.209
    save_pos = []

    # function_definition
    save_pos.append(pos)
    (pos, decl) = parse_function_definition(tokens, pos, ctx)
    if decl is not None:
        return (pos, decl)
    pos = save_pos.pop()

    # declaration
    save_pos.append(pos)
    (pos, decl) = parse_declaration(tokens, pos, ctx)
    if decl is not None:
        return (pos, decl)
    pos = save_pos.pop()

    # SEMICOLON
    if str(tokens[pos]) == ";":
        return (pos + 1, ";")

    return (pos, None)


def parse_translation_unit(tokens, pos, ctx):
    # p.209
    save_pos = []

    save_pos.append(pos)
    decls = []
    (pos, decl) = parse_external_declaration(tokens, pos, ctx)
    while decl is not None:
        decls.append(decl)
        (pos, decl) = parse_external_declaration(tokens, pos, ctx)
    if len(decls) > 0:
        return (pos, decls)
    pos = save_pos.pop()

    return (pos, None)


class Context:
    def __init__(self):
        self.typenames = []

    def is_typename(self, value):
        return value in self.typenames


def parse(tokens):
    ctx = Context()
    pos = 0
    (pos, node) = parse_translation_unit(tokens, pos, ctx)
    assert(pos == len(tokens))
    return node
