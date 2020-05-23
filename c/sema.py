#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast.node import *
from c.symtab import SymbolTable
from c.parse import *


anonymous_id = 0
from c.symtab import Symbol


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
                    raise NotImplementedError()

                values[value_name] = value_val

            for const_decl in enum_decl.const_decls:
                value_name = const_decl.ident.val

                var = VariableSymbol(value_name, enum_ty, [])
                sym_table.register_object(value_name, var)

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
                name = f"struct.anon{anonymous_id}"

            record_ty = CompositeType(name, None, is_union)
            record_ty.is_packed = struct_decl.is_packed
            sym_table.register(name, record_ty)

        def get_struct_type(type_spec, type_quals, decls):
            fields = []
            for decl in decls:
                name, ty = get_decl_name_and_type(
                    type_spec, decl, sym_table, ctx)
                arrspec = None

                fields.append((ty, name, arrspec))
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

            if isinstance(decl.chunks[0].num_elems, IntegerConstantExpr):
                ty = ArrayType(ty, decl.chunks[0].num_elems.val)
            else:
                ty = PointerType(ty)

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

                params.append([param_ty, param_quals, param_name])

                param = params[-1]
                if not param_ty:
                    def f():
                        param[0] = get_type(sym_table, param_ty, ctx)
                    ctx.assignments.append(f)

            ty = FunctionType(ty, params, decl.function_is_variadic)

    return name, ty


from c.symtab import FunctionSymbol, VariableSymbol


def register_function(qual_spec, declarator, sym_table, ctx):
    ty = get_type(sym_table, qual_spec, ctx)
    name, ty = get_decl_name_and_type(ty, declarator, sym_table, ctx)

    if isinstance(ty, FunctionType):
        for param_ty, _, _ in ty.params:
            assert(isinstance(param_ty, Type))

        func = FunctionSymbol(name, PointerType(ty))

        sym = sym_table.register_object(name, func)

        return TypedFunctionProto(ty.return_ty, func, ty.params, qual_spec)

    assert(isinstance(ty, PointerType))
    sym_table.register(name, ty)

    return None


def evaluate_constant_expr(expr):
    if isinstance(expr, IntegerConstantExpr):
        return expr.val

    raise ValueError()


def enter_node(node, depth, ctx):
    sym_table = ctx.sym_table

    if isinstance(node, CompoundStmt):
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

                    params = []
                    for param_ty, param_decl in decl_params:
                        ty = get_type(sym_table, param_ty, ctx)

                        param_name, var_ty = get_decl_name_and_type(
                            ty, param_decl, sym_table, ctx)

                        assert(var_ty)
                        var = VariableSymbol(param_name, var_ty, None)
                        sym_table.register_object(param_name, var)
                        params.append(var)

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

                    assert(var_ty)
                    var = VariableSymbol(param_name, var_ty, None)
                    sym_table.register_object(param_name, var)
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
    if isinstance(t, PointerType):
        return True

    if not isinstance(t, PrimitiveType):
        return False

    return t.name in ['char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long']


scalar_convertable = [
    # (to, from)
    ('int', 'uint'),
    ('int', '_Bool'),
    ('int', 'float'),
    ('int', 'double'),

    ('unsigned int', '_Bool'),
    ('unsigned int', 'char'),
    ('unsigned int', 'unsigned char'),
    ('unsigned int', 'short'),
    ('unsigned int', 'unsigned short'),
    ('unsigned int', 'int'),
    ('unsigned int', 'float'),
    ('unsigned int', 'double'),

    ('_Bool', 'int'),
    ('_Bool', 'unsigned int'),
    ('_Bool', 'float'),
    ('_Bool', 'double'),

    ('float', 'int'),
    ('float', 'unsigned int'),
    ('float', '_Bool'),
    ('float', 'double'),

    ('double', 'int'),
    ('double', 'unsigned int'),
    ('double', '_Bool'),
    ('double', 'float'),

    ('long double', 'double'),
]

implicit_convertable = [
    # (to, from)

    ('int', '_Bool'),
    ('int', 'char'),
    ('int', 'unsigned char'),
    ('int', 'short'),
    ('int', 'unsigned short'),
    ('int', 'unsigned int'),

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
    ('unsigned int', 'unsigned long'),
    ('unsigned int', 'int'),

    ('int', 'unsigned long'),

    ('unsigned long', '_Bool'),
    ('unsigned long', 'short'),
    ('unsigned long', 'unsigned short'),
    ('unsigned long', 'unsigned int'),

    ('long', 'unsigned int'),

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
    lhs_type = lhs.type
    rhs_type = rhs.type

    if isinstance(lhs_type, ArrayType):
        lhs_type = PointerType(lhs_type.elem_ty)
        lhs = cast_if_need(lhs, lhs_type)

    if isinstance(rhs_type, ArrayType):
        rhs_type = PointerType(rhs_type.elem_ty)
        rhs = cast_if_need(rhs, rhs_type)

    if isinstance(lhs_type, PointerType) and is_integer_type(rhs_type):
        return lhs_type, lhs, rhs

    if isinstance(lhs_type, PointerType) and isinstance(lhs_type, PointerType):
        return PrimitiveType("long"), lhs, rhs

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    if lhs_type == rhs_type:
        return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)

    lhs_type, rhs_type = promote_by_rank(lhs_type, rhs_type)

    return lhs_type, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)


def is_bitwise_op(op):
    return op in ['&', '^', '|']


def compute_binary_bitwise_op_type(op, lhs_type, rhs_type, sym_table):
    # Bitwise operators
    able_conv_lhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == lhs_type.name] + [lhs_type.name]

    if rhs_type.name in able_conv_lhs:
        return rhs_type

    able_conv_rhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]

    if lhs_type.name in able_conv_rhs:
        return lhs_type

    raise Exception("Unsupporting operation.")


def is_compare_op(op):
    return op in ['==', '!=', '<', '>', '<=', '>=']


def cast_if_need(node, ty):
    if node.type != ty:
        return CastExpr(node, ty)

    return node


def is_integer_zero(expr):
    if not isinstance(expr.type, PrimitiveType):
        return False

    return expr.val == 0


def compute_binary_compare_op_type(op, lhs, rhs, sym_table):
    lhs_type = lhs.type
    rhs_type = rhs.type
    result_ty = PrimitiveType('int')

    if isinstance(lhs_type, PointerType) and isinstance(rhs_type, PointerType):
        return result_ty, lhs, rhs

    if isinstance(lhs_type, PointerType) and is_integer_zero(rhs):
        return result_ty, lhs, rhs

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    if lhs_type == rhs_type:
        return result_ty, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)

    lhs_type, rhs_type = promote_by_rank(lhs_type, rhs_type)

    return result_ty, cast_if_need(lhs, lhs_type), cast_if_need(rhs, rhs_type)


def is_shift_op(op):
    return op in ['<<', '>>']


def compute_binary_shift_op_type(op, lhs_type, rhs_type, sym_table):
    if lhs_type.name in integer_types and rhs_type.name in integer_types:
        return lhs_type

    raise Exception("Unsupporting operation.")


def is_logical_op(op):
    return op in ['&&', '^^', '||']


def compute_binary_logical_op_type(op, lhs, rhs, sym_table):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if isinstance(rhs_type, PointerType):
        rhs = TypedBinaryOp("!=", rhs, IntegerConstantExpr(
            0, PrimitiveType("int")), PrimitiveType("int"))
        rhs_type = rhs.type

    if isinstance(lhs_type, PointerType):
        lhs = TypedBinaryOp("!=", lhs, IntegerConstantExpr(
            0, PrimitiveType("int")), PrimitiveType("int"))
        lhs_type = lhs.type

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
    if not isinstance(ty, PointerType):
        return False

    return isinstance(ty.elem_ty, VoidType)


def compute_binary_assignment_op_type(op, lhs, rhs, sym_table):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if isinstance(rhs_type, ArrayType):
        rhs_type = PointerType(rhs_type.elem_ty)

    if isinstance(rhs_type, FunctionType):
        rhs_type = PointerType(rhs_type)

    if isinstance(lhs_type, PointerType) and is_integer_type(rhs_type):
        return lhs_type, lhs, rhs

    if isinstance(lhs_type, PointerType) and is_void_pointer_type(rhs_type):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    if is_void_pointer_type(lhs_type) and isinstance(rhs_type, PointerType):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    if is_integer_type(lhs_type) and isinstance(rhs_type, PointerType):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    if lhs_type == rhs_type:
        return lhs_type, lhs, rhs

    if is_integer_type(lhs_type) and is_integer_type(rhs_type):
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    able_conv_rhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]
    if lhs_type.name in able_conv_rhs:
        return lhs_type, lhs, cast_if_need(rhs, lhs_type)

    raise Exception("Unsupporting operation.")


def compute_binary_op_type(op, lhs, rhs, sym_table):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if is_binary_arith_op(op):
        return compute_binary_arith_op_type(op, lhs, rhs, sym_table)

    if is_bitwise_op(op):
        return compute_binary_bitwise_op_type(op, lhs_type, rhs_type, sym_table), lhs, rhs

    if is_compare_op(op):
        return compute_binary_compare_op_type(op, lhs, rhs, sym_table)

    if is_logical_op(op):
        return compute_binary_logical_op_type(op, lhs, rhs, sym_table)

    if is_shift_op(op):
        return compute_binary_shift_op_type(op, lhs_type, rhs_type, sym_table), lhs, rhs

    if is_assignment_op(op):
        return compute_binary_assignment_op_type(op, lhs, rhs, sym_table)

    raise Exception("Unsupporting operation.")


def is_scalar_convertable(from_type: str, to_type: str):
    return (to_type, from_type) in scalar_convertable


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
        CastExpr,
        TypedUnaryOp,
        TypedPostOp,
        TypedAccessorOp,
        TypedArrayIndexerOp,
        TypedFunctionCall))


def mangle_func_name(name):
    return name


def is_float_type(ty):
    return ty.name in ["float", "double"]


from ast.types import *
from ast.node import Ident


def get_type_initializer(init, ty, idx=0):
    if isinstance(ty, CompositeType) and ty.is_union:
        ty, _, _ = ty.fields[0]

    if isinstance(ty, CompositeType):
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
                        if expr.type == field_ty:
                            expr, _ = get_type_initializer(
                                expr, field_ty, 0)
                            expr_idx += 1
                        else:
                            expr, expr_idx = get_type_initializer(
                                init, field_ty, expr_idx)
                    else:
                        expr, _ = get_type_initializer(
                            expr, field_ty, 0)
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
    elif isinstance(ty, ArrayType):
        if isinstance(init, InitializerList):
            exprs = []
            expr_idx = idx
            field_ty = ty.elem_ty
            for i in range(ty.size):
                if expr_idx >= len(init.exprs):
                    break

                designator, expr = init.exprs[expr_idx]

                if isinstance(field_ty, (ArrayType, CompositeType)):
                    if not isinstance(expr, InitializerList):
                        expr, expr_idx = get_type_initializer(
                            init, field_ty, expr_idx)
                    else:
                        expr, _ = get_type_initializer(
                            expr, field_ty, 0)
                        expr_idx += 1
                    exprs.append([designator, expr])
                else:
                    assert(not isinstance(expr, InitializerList))
                    expr_idx += 1
                    exprs.append([designator, expr])

            return TypedInitializerList(exprs, ty), expr_idx
        else:
            return init, idx
    elif isinstance(ty, PointerType):
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
                            init, field_ty, expr_idx)
                    else:
                        expr, _ = get_type_initializer(
                            expr, field_ty, 0)
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


def exit_node(node, depth, ctx):
    sym_table = ctx.sym_table

    type_scope = ctx.scope_map[ctx.top_scope]
    obj_scope = ctx.scope_map[ctx.top_scope]

    if isinstance(node, (CompoundStmt, TypedFunction)):
        ctx.pop_scope(node)

    if isinstance(node, IntegerConstantExpr):
        ty = node.type
        assert(is_integer_type(ty))
        node = IntegerConstantExpr(node.val, ty)

    if isinstance(node, FloatingConstantExpr):
        ty = node.type
        assert(is_float_type(ty))
        node = FloatingConstantExpr(node.val, ty)

    if isinstance(node, VarDecl):
        variables = []

        is_typedef = node.qual_spec.storage_class_spec == StorageClass.Typedef
        is_extern = node.qual_spec.storage_class_spec == StorageClass.Extern
        is_static = node.qual_spec.storage_class_spec == StorageClass.Static

        ty = get_type(sym_table, node.qual_spec, ctx)
        quals = []

        if node.decls:
            for decl, init in node.decls:
                if decl.is_function_decl and not decl.is_pointer_decl:
                    continue

                name, var_ty = get_decl_name_and_type(ty, decl, sym_table, ctx)

                if is_typedef:
                    sym_table.register(name, var_ty)
                else:
                    if init:
                        init, _ = get_type_initializer(init, var_ty)

                        if isinstance(var_ty, PointerType) and isinstance(init.type, ArrayType):
                            var_ty = init.type

                        init = cast_if_need(init, var_ty)
                        assert(var_ty == init.type)

                    assert(var_ty)
                    var = VariableSymbol(name, var_ty, quals)
                    sym_table.register_object(name, var)
                    variables.append([TypedIdentExpr(var, var_ty), init])

        storage_class = []
        if is_extern:
            storage_class.append("extern")
        if is_static:
            storage_class.append("static")

        if not is_typedef:
            node = TypedVariable(ty, variables, storage_class)

    if isinstance(node, IdentExpr):
        assert(isinstance(node.val, str))

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

        ty = exprs[-1].type

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

        node = CastExpr(expr, ty)

    if isinstance(node, SizeOfExpr):
        return_type = PrimitiveType("unsigned int")

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
        assert(is_typed_expr(node.lhs))
        assert(is_typed_expr(node.rhs))

        lhs = node.lhs
        rhs = node.rhs

        lhs_type = lhs.type
        rhs_type = rhs.type

        return_type, lhs, rhs = compute_binary_op_type(
            node.op, lhs, rhs, sym_table)

        if return_type is None:
            raise RuntimeError(
                f"Error: Undefined operation between types \"{lhs_type.name}\" and \"{rhs_type.name}\" with op \"{node.op}\"")

        ty = return_type

        node = TypedBinaryOp(node.op, lhs, rhs, ty)

    if isinstance(node, ConditionalExpr):
        assert(is_typed_expr(node.true_expr))
        assert(is_typed_expr(node.false_expr))
        assert(is_typed_expr(node.cond_expr))

        true_expr = node.true_expr
        false_expr = node.false_expr
        cond_expr = node.cond_expr

        true_expr_type = promote_default_type(true_expr.type)
        false_expr_type = promote_default_type(false_expr.type)

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
            if not is_incr_func_type(node.expr.type):
                expr_typename = node.expr.type.name
                raise RuntimeError(
                    f"Error: Not supporting types with increment \"{expr_typename}\"")

            return_ty = node.expr.type

        if node.op in ["*"]:
            if not isinstance(node.expr.type, PointerType):
                raise RuntimeError(
                    f"The type is not dereferencable")

            return_ty = node.expr.type.elem_ty

        if node.op in ["&"]:
            return_ty = PointerType(node.expr.type)

        if node.op in ["!"]:
            return_ty = PrimitiveType("int")

        if node.op in ["~"]:
            return_ty = node.expr.type

        if node.op in ["+", "-"]:
            return_ty = node.expr.type

        node = TypedUnaryOp(node.op, node.expr, return_ty)

    if isinstance(node, PostOp):
        assert(is_typed_expr(node.expr))

        if not is_incr_func_type(node.expr.type):
            expr_typename = node.expr.type.name
            raise RuntimeError(
                f"Error: Not supporting types with increment \"{expr_typename}\"")

        node = TypedPostOp(node.op, node.expr, node.expr.type)

    if isinstance(node, ArrayIndexerOp):
        assert(is_typed_expr(node.arr))

        arr_type = node.arr.type

        assert(isinstance(arr_type, (ArrayType, PointerType)))

        node = TypedArrayIndexerOp(node.arr, node.idx, arr_type.elem_ty)

    if isinstance(node, AccessorOp):
        assert(is_typed_expr(node.obj))

        if isinstance(node.obj.type, PointerType):
            obj_typename = node.obj.type.elem_ty.name
        else:
            obj_typename = node.obj.type.name

        field_type = sym_table.find_type_and_field(
            Symbol(obj_typename, type_scope), node.field.val)

        field_type, field_name, field_qual = field_type

        if field_type is None:
            raise RuntimeError(
                f"Error: Invalid field access \"{node.field.val}\" in \"{obj_typename}\"")

        node = TypedAccessorOp(node.obj, node.field, field_type)

    if isinstance(node, FunctionCall):
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
            ptr_ty = node.ident.type
            assert(isinstance(ptr_ty, PointerType))
            assert(isinstance(ptr_ty.elem_ty, FunctionType))

            func_def = node.ident
            func_type = ptr_ty.elem_ty

        formal_params = func_type.params
        return_ty = func_type.return_ty

        # check parameter
        pos = 0
        for (param_type, param_quals, param_name), arg in zip(formal_params, node.params):
            if is_integer_type(param_type) == is_integer_type(arg.type):
                pos += 1
                continue

            if isinstance(param_type, (PointerType, ArrayType)) and isinstance(arg.type, (PointerType, ArrayType)):
                pos += 1
                continue

            if isinstance(param_type, PointerType) and is_integer_zero(arg):
                pos += 1
                continue

            raise RuntimeError(
                f"Error: Invalid type \"{arg.type.name}\" found at position \"{pos}\" must be \"{param_type.name}\".")

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


def register_buildin_funcs(sym_table):
    func = FunctionSymbol("__builtin_return_address", PointerType(FunctionType(
        PointerType(VoidType()), [[PrimitiveType("unsigned int"), [], "level"]])))

    sym_table.register_object("__builtin_return_address", func)


def semantic_analysis(ast):
    ctx = Context()
    register_buildin_types(ctx.sym_table)
    register_buildin_funcs(ctx.sym_table)

    traverse_depth_update(ast, enter_build_type_table,
                          exit_build_type_table, data=ctx)
    assert(ctx.top_scope == FileScope())
    analyzed = traverse_depth_update(ast, enter_node, exit_node, data=ctx)
    return (analyzed, ctx.sym_table)
