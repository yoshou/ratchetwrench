#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast.node import Ident
from ast.types import *
from c.symtab import FunctionSymbol, VariableSymbol
from c.symtab import Symbol
from ast.node import *
from c.symtab import SymbolTable
from c.parse import *

import ast


anonymous_id = 0


def get_type(sym_table, decl_spec, ctx):
    scope = ctx.scope_map[ctx.top_scope]

    assert(isinstance(decl_spec, DeclSpec))

    if decl_spec.ident:
        ty = sym_table.find_type(Symbol(decl_spec.ident.val, scope))
        if ty:
            return ty

    spec = decl_spec.spec

    if spec.ty == TypeSpecifierType.Enum:
        enum_decl = decl_spec.decl

        enum_ty = None
        if enum_decl.ident:
            enum_ty = sym_table.find_type(
                Symbol(enum_decl.ident.val, scope))

            if enum_ty and enum_ty.values:
                return enum_ty

        if not enum_ty:
            if enum_decl.ident:
                name = enum_decl.ident.val
            else:
                name = ""

            enum_ty = EnumType()
            sym_table.register(name, enum_ty)

        values = {}
        val = 0

        if enum_decl.const_decls:
            for const_decl in enum_decl.const_decls:
                value_name = const_decl.ident.val
                if not const_decl.value:
                    value_val = val
                    val += 1
                else:
                    assert(isinstance(const_decl.value, IntegerConstantExpr))
                    value_val = const_decl.value.val
                    val += 1

                assert(isinstance(value_val, int))

                values[value_name] = value_val

            for const_decl in enum_decl.const_decls:
                value_name = const_decl.ident.val

                var = VariableSymbol(value_name, enum_ty)
                var = sym_table.register_object(value_name, var)

            enum_ty.values = values

        return enum_ty

    if spec.ty == TypeSpecifierType.Struct:
        struct_decl = decl_spec.decl
        field_decls = struct_decl.decls
        is_union = struct_decl.is_union

        record_ty = None
        if struct_decl.ident:
            record_ty = sym_table.find_type(
                Symbol(struct_decl.ident.val, scope))

            if record_ty and record_ty.fields:
                return record_ty

        if not record_ty:
            if struct_decl.ident:
                name = struct_decl.ident.val
            else:
                global anonymous_id
                anonymous_id += 1
                name = f"anon{anonymous_id}"

            record_ty = CompositeType(name, None, is_union)
            record_ty.is_packed = struct_decl.is_packed
            sym_table.register(name, record_ty)

        def get_struct_type(type_spec, type_quals, declors):
            fields = []
            for declor in declors:
                decl = declor.decl

                name, ty = get_decl_name_and_type(
                    type_spec, decl, sym_table, ctx)
                arrspec = None

                fields.append((ty, name, declor.bit))
            return fields

        if field_decls:
            fields = []
            for decl in field_decls:
                type_quals = [qual for qual in decl.type if isinstance(
                    qual, TypeQualifier)]

                type_spec = DeclSpec()
                get_type_spec_type(decl.type, type_spec)
                type_spec.quals = type_quals

                assert(type_spec.spec)

                ty = get_type(sym_table, type_spec, ctx)

                if decl.declarators:
                    fields.extend(get_struct_type(ty, type_quals,
                                                  decl.declarators))
                else:
                    fields.append((ty, None, None))

            record_ty.fields = fields
            return record_ty

        return record_ty

    if spec.ty == TypeSpecifierType.Typename:
        ty = sym_table.find_type(Symbol(decl_spec.ident.val, scope))
        return ty

    ty_name = spec.ty.value
    if spec.ty in [TypeSpecifierType.Int, TypeSpecifierType.Double]:
        if spec.width == TypeSpecifierWidth.Long:
            ty_name = "long " + ty_name

    if spec.ty in [TypeSpecifierType.Char, TypeSpecifierType.Short, TypeSpecifierType.Int, TypeSpecifierType.Long]:
        if spec.sign == TypeSpecifierSign.Unsigned:
            ty_name = "unsigned " + ty_name

    return sym_table.find_type(Symbol(ty_name, scope))


def get_type_qual(ty, decl_spec, ctx):
    quals = Qualifier.Undefined
    if decl_spec.type_quals & TypeQualifier.Volatile:
        quals = Qualifier.Volatile | quals
    if decl_spec.type_quals & TypeQualifier.Const:
        quals = Qualifier.Const | quals
    if decl_spec.type_quals & TypeQualifier.Restrict:
        quals = Qualifier.Restrict | quals

    if quals != Qualifier.Undefined:
        return QualType(ty, quals)

    return ty


def get_decl_name_and_type(ty, decl, sym_table, ctx):
    name = None

    decls = []
    while True:
        if not decl:
            break

        if isinstance(decl, Ident):
            name = decl.val
            break

        decls.append(decl)
        decl = decl.ident_or_decl

    for decl in decls:
        for _ in decl.pointer:
            ty = PointerType(ty)

        if decl.is_array_decl:
            assert(not decl.is_function_decl)
            if not decl.chunks[0].num_elems:
                ty = ArrayType(ty, 0)
            else:
                ty = ArrayType(ty, evaluate_constant_expr(
                    decl.chunks[0].num_elems, ctx))

        if decl.is_function_decl:
            assert(not decl.is_array_decl)
            decl_params = decl.function_params

            if not decl_params:
                decl_params = []

            params = []
            for param_ty, param_decl in decl_params:
                param_quals = []

                param_ty = get_type(sym_table, param_ty, ctx)
                param_name, param_ty = get_decl_name_and_type(
                    param_ty, param_decl, sym_table, ctx)

                if isinstance(param_ty, ArrayType):
                    param_ty = PointerType(param_ty.elem_ty)

                params.append([param_ty, param_quals, param_name])

                # param = params[-1]
                # if not param_ty:
                #     def f():
                #         param[0] = get_type(sym_table, param_ty, ctx)
                #     ctx.assignments.append(f)

            ty = FunctionType(ty, params, decl.function_is_variadic)

    return name, ty


def register_function(qual_spec, declarator, sym_table, ctx):
    ty = get_type(sym_table, qual_spec, ctx)
    name, ty = get_decl_name_and_type(ty, declarator, sym_table, ctx)

    if is_func_type(ty):
        params = []
        for param_ty, param_name, param_quals in ty.params:
            assert(isinstance(param_ty, ast.types.Type))
            if isinstance(param_ty, ast.types.ArrayType):
                param_ty = PointerType(param_ty.elem_ty)
            assert(not isinstance(param_ty, ast.types.ArrayType))

            params.append((param_ty, param_name, param_quals))

        func = FunctionSymbol(name, PointerType(ty))

        func = sym_table.register_object(name, func)

        return TypedFunctionProto(ty.return_ty, func, params, qual_spec)

    assert(is_pointer_type(ty))
    sym_table.register(name, ty)

    return None


def is_enum_type(ty):
    return isinstance(ty, EnumType)


def evaluate_constant_expr(expr, ctx):
    sym_table = ctx.sym_table
    scope = ctx.scope_map[ctx.top_scope]

    if isinstance(expr, IntegerConstantExpr):
        return expr.val

    if isinstance(expr, BinaryOp):
        op = expr.op

        lhs = evaluate_constant_expr(expr.lhs, ctx)
        rhs = evaluate_constant_expr(expr.rhs, ctx)

        if op == "+":
            return lhs + rhs

    if isinstance(expr, TypedIdentExpr):
        if is_enum_type(expr.type):
            return expr.type.values[expr.val.name]

    raise ValueError()


def enter_node(node, depth, ctx):
    sym_table = ctx.sym_table

    if isinstance(node, (CompoundStmt, VarDecl)):
        ctx.push_scope(node)

    type_scope = ctx.scope_map[ctx.top_scope]

    if isinstance(node, VarDecl):
        get_type(sym_table, node.qual_spec, ctx)

        if node.decls:
            for decl, init in node.decls:
                if decl.is_function_decl and not decl.is_pointer_decl:
                    decl_params = decl.function_params

                    if not decl_params:
                        decl_params = []

                    for param_ty, param_decl in decl_params:
                        ty = get_type(sym_table, param_ty, ctx)

                        param_name, var_ty = get_decl_name_and_type(
                            ty, param_decl, sym_table, ctx)

                        if isinstance(var_ty, ArrayType):
                            var_ty = PointerType(var_ty.elem_ty)

                        assert(var_ty)

                        if param_name:
                            var = VariableSymbol(param_name, var_ty)
                            var = sym_table.register_object(param_name, var)

                    func_decl = register_function(
                        node.qual_spec, decl, sym_table, ctx)

                    if func_decl:
                        node = func_decl

    def get_decl_name(decl):
        if not decl:
            return None

        ident_or_decl = decl.ident_or_decl
        while isinstance(ident_or_decl, Declarator):
            ident_or_decl = ident_or_decl.ident_or_decl

        assert(isinstance(ident_or_decl, Ident))

        return ident_or_decl.val

    if isinstance(node, FunctionDecl):
        decl_params = node.declarator.function_params
        decl_params = decl_params if decl_params else []

        decl = register_function(
            node.qual_spec, node.declarator, sym_table, ctx)

        assert(isinstance(node.stmt, CompoundStmt))

        if decl:
            param_names = []
            for param_ty, param_decl in decl_params:
                param_name = get_decl_name(param_decl)
                param_names.append(param_name)

            ctx.push_scope(node)

            params = []
            for param_name, (param_ty, param_decl) in zip(param_names, decl_params):
                if param_name is not None:
                    ty = get_type(sym_table, param_ty, ctx)

                    param_name, var_ty = get_decl_name_and_type(
                        ty, param_decl, sym_table, ctx)

                    if isinstance(var_ty, ArrayType):
                        var_ty = PointerType(var_ty.elem_ty)

                    assert(var_ty)
                    var = VariableSymbol(param_name, var_ty)
                    var = sym_table.register_object(param_name, var)
                    params.append(var)

            assert(decl)
            node = TypedFunction(decl, params, [node.stmt])

    return node


def is_integer_type(t):
    if not isinstance(t, PrimitiveType):
        return False

    return t.name in [
        'char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long']


def is_incr_func_type(t):
    if is_pointer_type(t):
        return True

    if not isinstance(t, PrimitiveType):
        return False

    return t.name in ['char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long']


implicit_convertable = [
    # (to, from)

    ('int', '_Bool'),
    ('int', 'char'),
    ('int', 'unsigned char'),
    ('int', 'short'),
    ('int', 'unsigned short'),

    ('unsigned char', '_Bool'),

    ('short', '_Bool'),
    ('short', 'char'),
    ('short', 'unsigned char'),

    ('_Bool', 'int'),
    ('char', 'int'),
    ('unsigned char', 'int'),
    ('short', 'int'),
    ('unsigned short', 'int'),
    ('unsigned int', 'int'),
    ('long', 'int'),
    ('unsigned long', 'int'),

    ('unsigned short', '_Bool'),
    ('unsigned short', 'char'),
    ('unsigned short', 'unsigned char'),
    ('unsigned short', 'short'),
    ('unsigned short', 'unsigned int'),
    ('unsigned short', 'long'),
    ('unsigned short', 'unsigned long'),

    ('unsigned int', '_Bool'),
    ('unsigned int', 'unsigned char'),
    ('unsigned int', 'unsigned short'),
    ('unsigned int', 'int'),

    ('long', 'unsigned int'),

    ('unsigned long', '_Bool'),
    ('unsigned long', 'short'),
    ('unsigned long', 'unsigned short'),
    ('unsigned long', 'unsigned int'),
    ('unsigned long', 'long'),

    ('float', 'int'),
    ('float', 'unsigned int'),

    ('double', '_Bool'),
    ('double', 'int'),
    ('double', 'unsigned int'),
    ('double', 'float'),

    ('long double', 'double'),

    ('unsigned long', 'unsigned int'),

    # TODO: Not implemented. Need additional items.
]

integer_types = [
    '_Bool', 'char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long'
]


def is_binary_arith_op(op):
    return op in ['+', '-', '*', '/', '%']


def promote_default_type(ty):
    if isinstance(ty, EnumType):
        return PrimitiveType("int")

    if not isinstance(ty, PrimitiveType):
        return ty

    if ty.name in ["_Bool", "char", "unsigned char", "short", "unsigned short"]:
        return PrimitiveType("int")

    return ty


def promote_by_rank(lhs_ty, rhs_ty):
    able_conv_lhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == lhs_ty.name] + [lhs_ty.name]

    if rhs_ty.name in able_conv_lhs:
        return rhs_ty, rhs_ty

    able_conv_rhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == rhs_ty.name] + [rhs_ty.name]

    if lhs_ty.name in able_conv_rhs:
        return lhs_ty, lhs_ty

    raise Exception("Unsupporting cast.")


def compute_binary_arith_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    if is_array_type(lhs_type):
        lhs_type = PointerType(lhs_type.elem_ty)
        lhs = cast_if_need(lhs, lhs_type)

    if is_array_type(rhs_type):
        rhs_type = PointerType(rhs_type.elem_ty)
        rhs = cast_if_need(rhs, rhs_type)

    if is_pointer_type(lhs_type) and is_integer_type(rhs_type):
        return lhs_type, lhs, cast_if_need(rhs, PrimitiveType("long"))

    if is_pointer_type(lhs_type) and is_pointer_type(lhs_type):
        return PrimitiveType("long"), lhs, rhs

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    if lhs_type == rhs_type:
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)

    lhs_type, rhs_type = promote_by_rank(lhs_type, rhs_type)

    return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)


def is_bitwise_op(op):
    return op in ['&', '^', '|']


def compute_binary_bitwise_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    if lhs_type == rhs_type:
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)

    lhs_type, rhs_type = promote_by_rank(lhs_type, rhs_type)

    return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)


def is_compare_op(op):
    return op in ['==', '!=', '<', '>', '<=', '>=']


def cast_if_need(node, ty):
    if get_type_of(node) != ty:
        return TypedCastExpr(node, ty)

    return node


def is_integer_zero(expr):
    if not isinstance(get_type_of(expr), PrimitiveType):
        return False

    return expr.val == 0


def compute_binary_compare_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)
    result_ty = PrimitiveType('int')

    if is_array_type(rhs_type):
        rhs_type = PointerType(rhs_type.elem_ty)

    if is_pointer_type(lhs_type) and is_pointer_type(rhs_type):
        return result_ty, lhs, rhs

    if is_pointer_type(lhs_type) and is_integer_zero(rhs):
        return result_ty, lhs, rhs

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    if lhs_type == rhs_type:
        return result_ty, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)

    lhs_type, rhs_type = promote_by_rank(lhs_type, rhs_type)

    return result_ty, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)


def is_shift_op(op):
    return op in ['<<', '>>']


def compute_binary_shift_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    if lhs_type == rhs_type:
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)

    lhs_type, rhs_type = promote_by_rank(lhs_type, rhs_type)

    return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)


def is_logical_op(op):
    return op in ['&&', '^^', '||']


def compute_binary_logical_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    if is_pointer_type(rhs_type):
        rhs = TypedBinaryOp("!=", rhs, IntegerConstantExpr(
            0, PrimitiveType("int")), PrimitiveType("int"))
        rhs_type = get_type_of(rhs)

    if is_pointer_type(lhs_type):
        lhs = TypedBinaryOp("!=", lhs, IntegerConstantExpr(
            0, PrimitiveType("int")), PrimitiveType("int"))
        lhs_type = get_type_of(lhs)

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    # Logical operators
    if lhs_type.name == 'int' and rhs_type.name == 'int':
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, lhs_type)

    if lhs_type.name == 'int' and rhs_type.name == 'unsigned int':
        return rhs_type, cast_if_need(lhs, rhs_type), cast_if_need(rhs, rhs_type)

    if lhs_type.name == 'unsigned int' and rhs_type.name == 'int':
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, lhs_type)

    if lhs_type.name == 'unsigned long' and rhs_type.name == 'int':
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, lhs_type)

    raise Exception("Unsupporting operation.")


def is_assignment_op(op):
    return op in ['=', '+=', '-=', '*=', '/=', '>>=', '<<=', '&=', '|=', '^=']


def is_void_pointer_type(ty):
    if not is_pointer_type(ty):
        return False

    return is_void_type(ty.elem_ty)


def is_void_type(ty):
    return isinstance(ty, VoidType)


def is_pointer_type(ty):
    if isinstance(ty, QualType):
        ty = ty.ty

    return isinstance(ty, PointerType)


def is_array_type(ty):
    if isinstance(ty, QualType):
        ty = ty.ty

    return isinstance(ty, ArrayType)


def is_composite_type(ty):
    if isinstance(ty, QualType):
        ty = ty.ty

    return isinstance(ty, CompositeType)


def is_func_type(ty):
    return isinstance(ty, FunctionType)


def get_type_of(value):
    if isinstance(value.type, QualType):
        return value.type.ty

    return value.type


def compute_binary_assignment_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    if is_array_type(rhs_type):
        rhs_type = PointerType(rhs_type.elem_ty)

    if is_func_type(rhs_type):
        rhs_type = PointerType(rhs_type)

    if is_pointer_type(lhs_type) and is_integer_type(rhs_type):
        return lhs_type, lhs, rhs

    if is_pointer_type(lhs_type) and is_void_pointer_type(rhs_type):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    if is_void_pointer_type(lhs_type) and is_pointer_type(rhs_type):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    if is_integer_type(lhs_type) and is_pointer_type(rhs_type):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    if lhs_type == rhs_type:
        return lhs_type, lhs, rhs

    if is_integer_type(lhs_type) and is_integer_type(rhs_type):
        return lhs_type, lhs, rhs

    able_conv_rhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]
    if lhs_type.name in able_conv_rhs:
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    raise Exception("Unsupporting operation.")


def compute_binary_op_type(op, lhs, rhs, sym_table):
    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    if is_binary_arith_op(op):
        return compute_binary_arith_op_type(op, lhs, rhs, sym_table)

    if is_bitwise_op(op):
        return compute_binary_bitwise_op_type(op, lhs, rhs, sym_table)

    if is_compare_op(op):
        return compute_binary_compare_op_type(op, lhs, rhs, sym_table)

    if is_logical_op(op):
        return compute_binary_logical_op_type(op, lhs, rhs, sym_table)

    if is_shift_op(op):
        return compute_binary_shift_op_type(op, lhs, rhs, sym_table)

    if is_assignment_op(op):
        return compute_binary_assignment_op_type(op, lhs, rhs, sym_table)

    raise Exception("Unsupporting operation.")


def is_implicit_convertable(from_type: str, to_type: str):
    return (to_type, from_type) in implicit_convertable


def is_typed_expr(node):
    return isinstance(node, (
        TypedBinaryOp,
        TypedCommaOp,
        TypedSizeOfExpr,
        TypedIdentExpr,
        TypedConditionalExpr,
        IntegerConstantExpr,
        StringLiteralExpr,
        FloatingConstantExpr,
        TypedCastExpr,
        TypedUnaryOp,
        TypedPostOp,
        TypedAccessorOp,
        TypedArrayIndexerOp,
        TypedFunctionCall))


def mangle_func_name(name):
    return name


def is_float_type(ty):
    return ty.name in ["float", "double"]


def get_type_initializer(init, ty, ctx, idx=0):
    if isinstance(ty, CompositeType) and ty.is_union:
        ty, _, _ = ty.fields[0]

    if is_composite_type(ty):
        if isinstance(init, InitializerList):
            exprs = []
            expr_idx = idx
            for field_ty, _, _ in ty.fields:
                if expr_idx >= len(init.exprs):
                    break

                designator, expr = init.exprs[expr_idx]
                assert(not designator)

                if isinstance(field_ty, (ArrayType, CompositeType)):
                    if not isinstance(expr, InitializerList):
                        if get_type_of(expr) == field_ty:
                            expr, _ = get_type_initializer(
                                expr, field_ty, ctx, 0)
                            expr_idx += 1
                        else:
                            expr, expr_idx = get_type_initializer(
                                init, field_ty, ctx, expr_idx)
                    else:
                        expr, _ = get_type_initializer(
                            expr, field_ty, ctx, 0)
                        expr_idx += 1
                    exprs.append([None, expr])
                else:
                    assert(not isinstance(expr, InitializerList))
                    expr_idx += 1
                    exprs.append([None, expr])

            return TypedInitializerList(exprs, ty), expr_idx
        else:
            return init, idx
    elif isinstance(ty, PrimitiveType):
        return init, idx
    elif is_array_type(ty):
        if isinstance(init, InitializerList):
            exprs = []
            expr_idx = idx
            field_ty = ty.elem_ty
            size = 0
            for i in range(len(init.exprs)):
                if expr_idx >= len(init.exprs):
                    break

                designators, expr = init.exprs[expr_idx]

                if isinstance(field_ty, (ArrayType, CompositeType)):
                    if not isinstance(expr, InitializerList):
                        expr, expr_idx = get_type_initializer(
                            init, field_ty, ctx, expr_idx)
                    else:
                        expr, _ = get_type_initializer(
                            expr, field_ty, ctx, 0)
                        expr_idx += 1
                else:
                    assert(not isinstance(expr, InitializerList))
                    expr_idx += 1

                if designators:
                    for designator in designators:
                        if isinstance(designator, IntegerConstantExpr):
                            designator_idx = designator.val
                        else:
                            assert(isinstance(designator, TypedIdentExpr))
                            designator_idx = evaluate_constant_expr(
                                designator, ctx)
                        size = max(size, designator_idx + 1)
                else:
                    size += 1
                exprs.append((designators, expr))

            assert(size > 0)

            ty = ArrayType(ty.elem_ty, size)

            return TypedInitializerList(exprs, ty), expr_idx
        else:
            return init, idx
    elif is_pointer_type(ty):
        if isinstance(init, InitializerList):
            exprs = []
            expr_idx = idx
            field_ty = ty.elem_ty
            for i in range(len(init.exprs)):
                if expr_idx >= len(init.exprs):
                    break

                _, expr = init.exprs[expr_idx]

                if isinstance(field_ty, (ArrayType, CompositeType)):
                    if not isinstance(expr, InitializerList):
                        expr, expr_idx = get_type_initializer(
                            init, field_ty, ctx, expr_idx)
                    else:
                        expr, _ = get_type_initializer(
                            expr, field_ty, ctx, 0)
                        expr_idx += 1
                    exprs.append([None, expr])
                else:
                    assert(not isinstance(expr, InitializerList))
                    expr_idx += 1
                    exprs.append([None, expr])

            ty = ArrayType(ty.elem_ty, len(exprs))

            return TypedInitializerList(exprs, ty), expr_idx
        else:
            return init, idx
    else:
        raise NotImplementedError()


def type_check_binary_op(node, sym_table, type_scope, obj_scope, ctx):
    assert(is_typed_expr(node.lhs))
    assert(is_typed_expr(node.rhs))

    lhs = node.lhs
    rhs = node.rhs

    lhs_type = get_type_of(lhs)
    rhs_type = get_type_of(rhs)

    op = node.op

    if node.op in ['+=', '-=', '*=', '/=', '>>=', '<<=', '&=', '|=', '^=']:
        rhs = type_check_binary_op(
            BinaryOp(node.op[:-1], lhs, rhs), sym_table, type_scope, obj_scope, ctx)
        op = '='

    return_type, lhs, rhs = compute_binary_op_type(
        op, lhs, rhs, sym_table)

    if return_type is None:
        raise RuntimeError(
            f"Error: Undefined operation between types \"{lhs_type.name}\" and \"{rhs_type.name}\" with op \"{node.op}\"")

    ty = return_type

    return TypedBinaryOp(op, lhs, rhs, ty)


def get_canonical_type(ty):
    if isinstance(ty, QualType):
        return ty.ty

    return ty


def exit_node(node, depth, ctx):
    sym_table = ctx.sym_table

    type_scope = ctx.scope_map[ctx.top_scope]
    obj_scope = ctx.scope_map[ctx.top_scope]

    if isinstance(node, (CompoundStmt, TypedFunction, VarDecl)):
        ctx.pop_scope(node)

    if isinstance(node, IntegerConstantExpr):
        ty = get_type_of(node)
        assert(is_integer_type(ty))
        node = IntegerConstantExpr(node.val, ty)

    if isinstance(node, FloatingConstantExpr):
        ty = get_type_of(node)
        assert(is_float_type(ty))
        node = FloatingConstantExpr(node.val, ty)

    if isinstance(node, VarDecl):
        variables = []

        is_typedef = node.qual_spec.storage_class_spec == StorageClass.Typedef
        is_extern = node.qual_spec.storage_class_spec == StorageClass.Extern
        is_static = node.qual_spec.storage_class_spec == StorageClass.Static

        ty = get_type(sym_table, node.qual_spec, ctx)

        if node.decls:
            for decl, init in node.decls:
                if decl.is_function_decl and not decl.is_pointer_decl:
                    continue

                name, var_ty = get_decl_name_and_type(ty, decl, sym_table, ctx)
                var_ty = get_type_qual(var_ty, node.qual_spec, ctx)

                if is_typedef:
                    sym_table.register(name, var_ty)
                else:
                    if init:
                        init, _ = get_type_initializer(
                            init, get_canonical_type(var_ty), ctx)
                        if is_array_type(get_type_of(init)):
                            var_ty = get_type_of(init)

                        init = cast_if_need(init, var_ty)
                        assert(get_canonical_type(var_ty) == get_type_of(init))

                        var_ty = init.type
                        if is_array_type(var_ty):
                            assert(get_canonical_type(var_ty).size > 0)

                    assert(var_ty)
                    var = VariableSymbol(name, var_ty)
                    var = sym_table.register_object(name, var)
                    variables.append([TypedIdentExpr(var, var_ty), init])

        storage_class = []
        if is_extern:
            storage_class.append("extern")
        if is_static:
            storage_class.append("static")
        if node.qual_spec.thread_storage_class_spec == StorageClass.ThreadLocal:
            storage_class.append("thread_local")

        if not is_typedef:
            node = TypedVariable(ty, variables, storage_class)

    if isinstance(node, IdentExpr):
        assert(isinstance(node.val, str))

        if node.val in ["__builtin_va_arg"]:
            pass
        else:
            var = sym_table.find_object(Symbol(node.val, obj_scope))
            if var:
                assert(var.ty)
                typed_node = TypedIdentExpr(var, var.ty)

            if not typed_node:
                raise RuntimeError(f"Error: Undefined identity \"{node.val}\"")

            node = typed_node

    if isinstance(node, CommaOp):
        exprs = node.exprs

        for expr in exprs:
            assert(is_typed_expr(expr))

        ty = get_type_of(exprs[-1])

        assert(ty)

        node = TypedCommaOp(exprs, ty)

    if isinstance(node, CastExpr):
        expr = node.expr

        assert(is_typed_expr(expr))

        decl_spec = DeclSpec()
        get_type_spec_type(node.type.qual_spec, decl_spec)

        assert(decl_spec.spec)

        ty = get_type(sym_table, decl_spec, ctx)

        name, ty = get_decl_name_and_type(ty, node.type.decl, sym_table, ctx)

        node = TypedCastExpr(expr, ty)

    if isinstance(node, SizeOfExpr):
        return_type = PrimitiveType("unsigned long")

        if node.type:
            type_spec = DeclSpec()
            get_type_spec_type(node.type.qual_spec, type_spec)

            assert(type_spec.spec)

            ty = get_type(sym_table, type_spec, ctx)
            name, ty = get_decl_name_and_type(
                ty, node.type.decl, sym_table, ctx)
        else:
            ty = None

        node = TypedSizeOfExpr(node.expr, ty, return_type)

    if isinstance(node, BinaryOp):
        node = type_check_binary_op(
            node, sym_table, type_scope, obj_scope, ctx)

    if isinstance(node, ConditionalExpr):
        assert(is_typed_expr(node.true_expr))
        assert(is_typed_expr(node.false_expr))
        assert(is_typed_expr(node.cond_expr))

        true_expr = node.true_expr
        false_expr = node.false_expr
        cond_expr = node.cond_expr

        true_expr_type = promote_default_type(get_type_of(true_expr))
        false_expr_type = promote_default_type(get_type_of(false_expr))

        result_type = None

        if true_expr_type == false_expr_type:
            result_type = true_expr_type
        elif is_implicit_convertable(true_expr_type.name, false_expr_type.name):
            result_type = false_expr_type
        elif is_implicit_convertable(false_expr_type.name, true_expr_type.name):
            result_type = true_expr_type

        assert(result_type)

        true_expr = cast_if_need(true_expr, result_type)
        false_expr = cast_if_need(false_expr, result_type)

        ty = result_type

        node = TypedConditionalExpr(cond_expr, true_expr, false_expr, ty)

    if isinstance(node, UnaryOp):
        assert(is_typed_expr(node.expr))

        if node.op in ["++", "--"]:
            if not is_incr_func_type(get_type_of(node.expr)):
                expr_typename = get_type_of(node.expr).name
                raise RuntimeError(
                    f"Error: Not supporting types with increment \"{expr_typename}\"")

            return_ty = get_type_of(node.expr)

        if node.op in ["*"]:
            if not is_pointer_type(get_type_of(node.expr)):
                raise RuntimeError(
                    f"The type is not dereferencable")

            return_ty = get_type_of(node.expr).elem_ty

        if node.op in ["&"]:
            return_ty = PointerType(get_type_of(node.expr))

        if node.op in ["!"]:
            return_ty = PrimitiveType("int")

        if node.op in ["~"]:
            return_ty = get_type_of(node.expr)

        if node.op in ["+", "-"]:
            return_ty = get_type_of(node.expr)

        node = TypedUnaryOp(node.op, node.expr, return_ty)

    if isinstance(node, PostOp):
        assert(is_typed_expr(node.expr))

        if not is_incr_func_type(get_type_of(node.expr)):
            expr_typename = node.expr.type.name
            raise RuntimeError(
                f"Error: Not supporting types with increment \"{expr_typename}\"")

        node = TypedPostOp(node.op, node.expr, get_type_of(node.expr))

    if isinstance(node, ArrayIndexerOp):
        assert(is_typed_expr(node.arr))

        arr_type = get_type_of(node.arr)
        if isinstance(node.arr.type, QualType):
            arr_type = QualType(arr_type, node.arr.type.quals)

        assert(is_pointer_type(arr_type) or is_array_type(arr_type))

        idx = cast_if_need(node.idx, PrimitiveType("long"))

        node = TypedArrayIndexerOp(node.arr, idx, arr_type.elem_ty)

    if isinstance(node, AccessorOp):
        assert(is_typed_expr(node.obj))

        if is_pointer_type(get_type_of(node.obj)):
            obj_typename = get_type_of(node.obj).elem_ty.name
        else:
            obj_typename = get_type_of(node.obj).name

        field_type = sym_table.find_type_and_field(
            Symbol(obj_typename, type_scope), node.field.val)

        field_type, field_name, field_qual = field_type

        if field_type is None:
            raise RuntimeError(
                f"Error: Invalid field access \"{node.field.val}\" in \"{obj_typename}\"")

        node = TypedAccessorOp(node.obj, node.field, field_type)

    if isinstance(node, FunctionCall):
        if isinstance(node.ident, IdentExpr):
            if node.ident.val == "__builtin_va_arg":

                type_spec = DeclSpec()
                get_type_spec_type(node.params[1].qual_spec, type_spec)
                return_ty = get_type(sym_table, type_spec, ctx)
                _, return_ty = get_decl_name_and_type(
                    return_ty, node.params[1].decl, sym_table, ctx)

                if return_ty == PrimitiveType("int"):
                    func_name = node.ident.val + "_int"
                elif return_ty == PrimitiveType("char"):
                    func_name = node.ident.val + "_char"
                elif return_ty == PrimitiveType("unsigned int"):
                    func_name = node.ident.val + "_uint"
                elif return_ty == PrimitiveType("unsigned long"):
                    func_name = node.ident.val + "_ulong"
                elif is_pointer_type(return_ty):
                    func_name = node.ident.val + "_ptr"
                else:
                    raise NotImplementedError()

                func_def = sym_table.find_object(Symbol(func_name, obj_scope))

                assert(func_def)

                node = TypedFunctionCall(func_def, [node.params[0]], return_ty)
        else:
            if isinstance(node.ident, TypedIdentExpr):
                func_name = node.ident.val.name
                func_name = mangle_func_name(func_name)

                func_def = sym_table.find_object(Symbol(func_name, obj_scope))

                if not func_def:
                    var_def = sym_table.find_object(func_name)

                    if not var_def:
                        raise RuntimeError(
                            f"Error: The function \"{func_name}\" not defined.")

                    func_type = var_def.ty.elem_ty
                    func_def = node.ident
                else:
                    func_type = func_def.ty.elem_ty
            else:
                ptr_ty = get_type_of(node.ident)
                assert(is_pointer_type(ptr_ty))
                assert(is_func_type(ptr_ty.elem_ty))

                func_def = node.ident
                func_type = ptr_ty.elem_ty

            formal_params = func_type.params
            return_ty = func_type.return_ty

            # check parameter
            pos = 0
            for (param_type, param_quals, param_name), arg in zip(formal_params, node.params):
                if is_integer_type(param_type) == is_integer_type(get_type_of(arg)):
                    pos += 1
                    continue

                if (is_array_type(param_type) or is_pointer_type(param_type)) and (is_array_type(get_type_of(arg)) or is_pointer_type(get_type_of(arg))):
                    pos += 1
                    continue

                if is_pointer_type(param_type) and is_integer_zero(arg):
                    pos += 1
                    continue

                raise RuntimeError(
                    f"Error: Invalid type \"{get_type_of(arg).name}\" found at position \"{pos}\" must be \"{param_type.name}\".")

            node = TypedFunctionCall(func_def, node.params, return_ty)

    if isinstance(node, AsmStmt):
        operands_list = []
        for operands in node.operands:
            new_operands = []
            for constraint, name in operands:
                new_operands.append((constraint, name))

            operands_list.append(new_operands)

        node = AsmStmt(node.template, operands_list)

    return node


class BlockScope:
    def __init__(self, block, parent):
        self.block = block

    def __hash__(self):
        return hash(tuple([self.block]))

    def __eq__(self, other):
        if not isinstance(other, BlockScope):
            return False

        return self.block == other.block


class FileScope:
    def __init__(self):
        pass

    def __hash__(self):
        return hash("file")

    def __eq__(self, other):
        return isinstance(other, FileScope)

    @property
    def parent(self):
        return None


class Context:
    def __init__(self):
        self.sym_table = SymbolTable()
        self.scope_map = {}
        file_scope = FileScope()
        self.scope = [file_scope]
        self.scope_map[file_scope] = self.sym_table.global_scope
        self.type_assignments = []

    def push_scope(self, node):
        self.scope.append(BlockScope(node, self.top_scope))
        self.sym_table.push_scope()
        self.scope_map[self.top_scope] = self.sym_table.top_scope

    def pop_scope(self, node):
        # assert(self.top_scope.block == node)
        self.scope.pop()
        self.sym_table.pop_scope()

    @property
    def top_scope(self):
        return self.scope[-1]


def enter_build_type_table(node, depth, ctx):
    sym_table = ctx.sym_table

    if isinstance(node, (CompoundStmt,)):
        ctx.push_scope(node)

    if isinstance(node, VarDecl):
        is_typedef = node.qual_spec.storage_class_spec == StorageClass.Typedef
        ty = get_type(sym_table, node.qual_spec, ctx)

        if is_typedef:
            for decl, _ in node.decls:
                if decl.is_function_decl:
                    continue

                name, var_ty = get_decl_name_and_type(ty, decl, sym_table, ctx)

                sym_table.register(name, var_ty)

    return node


def exit_build_type_table(node, depth, ctx):
    sym_table = ctx.sym_table

    if isinstance(node, (CompoundStmt,)):
        ctx.pop_scope(node)

    return node


def register_buildin_types(sym_table: SymbolTable):
    sym_table.register('void', VoidType())

    sym_table.register('char', PrimitiveType('char'))
    sym_table.register('unsigned char', PrimitiveType('unsigned char'))
    sym_table.register('short', PrimitiveType('short'))
    sym_table.register('unsigned short', PrimitiveType('unsigned short'))
    sym_table.register('int', PrimitiveType('int'))
    sym_table.register('unsigned int', PrimitiveType('unsigned int'))
    sym_table.register('long', PrimitiveType('long'))
    sym_table.register('unsigned long', PrimitiveType('unsigned long'))

    sym_table.register('float', PrimitiveType('float'))
    sym_table.register('double', PrimitiveType('double'))

    sym_table.register('_Bool', PrimitiveType('_Bool'))
    sym_table.register('_Complex', PrimitiveType('_Complex'))
    sym_table.register('long double', PrimitiveType('long double'))

    sym_table.register('__int64', PrimitiveType('__int64'))
    sym_table.register('unsigned __int64', PrimitiveType('unsigned __int64'))
    sym_table.register('unsigned', PrimitiveType('unsigned int'))

    va_list_tag = CompositeType("__va_list_tag", [
        [PrimitiveType('unsigned int'), None, None],
        [PrimitiveType('unsigned int'), None, None],
        [PointerType(PrimitiveType('unsigned char')), None, None],
        [PointerType(PrimitiveType('unsigned char')), None, None],
    ])
    va_list_tag.is_packed = False

    sym_table.register('__builtin_va_list', ArrayType(va_list_tag, 1))


def register_buildin_funcs(sym_table: SymbolTable):
    func = FunctionSymbol("__builtin_return_address", PointerType(FunctionType(
        PointerType(VoidType()), [[PrimitiveType("unsigned int"), [], "level"]])))

    sym_table.register_object("__builtin_return_address", func)

    va_list_ty = sym_table.find_type(
        Symbol("__builtin_va_list", sym_table.top_scope))

    func = FunctionSymbol("__builtin_va_arg_int", PointerType(FunctionType(
        PrimitiveType("int"), [[va_list_ty, [], None]])))

    sym_table.register_object("__builtin_va_arg_int", func)

    func = FunctionSymbol("__builtin_va_arg_uint", PointerType(FunctionType(
        PrimitiveType("unsigned int"), [[va_list_ty, [], None]])))

    sym_table.register_object("__builtin_va_arg_uint", func)

    func = FunctionSymbol("__builtin_va_arg_ulong", PointerType(FunctionType(
        PrimitiveType("unsigned long"), [[va_list_ty, [], None]])))

    sym_table.register_object("__builtin_va_arg_ulong", func)

    func = FunctionSymbol("__builtin_va_arg_char", PointerType(FunctionType(
        PrimitiveType("char"), [[va_list_ty, [], None]])))

    sym_table.register_object("__builtin_va_arg_char", func)

    func = FunctionSymbol("__builtin_va_arg_ptr", PointerType(FunctionType(
        PointerType(VoidType()), [[va_list_ty, [], None]])))

    sym_table.register_object("__builtin_va_arg_ptr", func)

    func = FunctionSymbol("__builtin_va_start", PointerType(
        FunctionType(VoidType(), [[va_list_ty, [], None]], True)))

    sym_table.register_object("__builtin_va_start", func)


def semantic_analysis(ast):
    ctx = Context()
    register_buildin_types(ctx.sym_table)
    register_buildin_funcs(ctx.sym_table)

    traverse_depth_update(ast, enter_build_type_table,
                          exit_build_type_table, data=ctx)
    assert(ctx.top_scope == FileScope())
    analyzed = traverse_depth_update(ast, enter_node, exit_node, data=ctx)
    return (analyzed, ctx.sym_table)
