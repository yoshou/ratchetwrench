#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast.node import *
from c.symtab import SymbolTable, Symbol
from c.parse import *


def register_function_proto(node, sym_table, is_constructor=False):
    proto = node
    ty = sym_table.find_type(proto.type.specifier)
    params = []
    for param in proto.params:
        param_name = param.ident.val if param.ident is not None else None
        param_ty = sym_table.find_type(param.type.specifier)
        if isinstance(param.type, FullType):
            param_quals = param.type.qualifiers
        else:
            param_quals = []
        params.append((param_ty, param_quals, param_name))

    name = proto.ident.val
    if is_constructor:
        name = FuncSignature(name, [param[0] for param in params])

    sym = sym_table.register_func(name, ty, params)

    return TypedFunctionProto(ty, sym, params)


anonymous_id = 0


def get_type(sym_table, qual_spec):
    assert(isinstance(qual_spec, TypeSpecifier))

    from ast.types import EnumType

    if qual_spec.ident:
        ty = sym_table.find_type(qual_spec.ident.val)
        if ty:
            return ty

    spec = qual_spec.spec
    ty_name = spec.ty.value

    if ty_name == "enum":
        values = {}
        val = 0
        for const_decl in spec.enum_spec.const_decls:
            value_name = const_decl.ident.val
            if not const_decl.value:
                value_val = val
                val += 1
            else:
                raise NotImplementedError()

            values[value_name] = value_val

        ty = EnumType(spec.enum_spec.ident.val, values)
        for const_decl in spec.enum_spec.const_decls:
            value_name = const_decl.ident.val

            sym_table.register_var(value_name, ty, [])
        return ty

    if ty_name == "struct":
        decls = spec.struct_spec.decls
        is_union = spec.struct_spec.is_union

        def get_struct_type(type_spec, type_quals, decls):
            fields = []
            for decl in decls:
                name, ty = get_decl_name_and_type(
                    type_spec, decl.ident, sym_table)
                arrspec = None

                fields.append((ty, name, arrspec))
            return fields

        if decls:
            fields = []
            for decl in decls:
                type_quals = [qual for qual in decl.type if isinstance(
                    qual, TypeQualifier)]

                if len(decl.type) == 1 and isinstance(decl.type[0], Ident):
                    type_spec = TypeSpecifier(None, [], decl.type[0], [])
                else:
                    spec_type = get_type_spec_type(decl.type)
                    type_spec = TypeSpecifier(spec_type, type_quals, None, [])

                ty = get_type(sym_table, type_spec)

                if decl.declarators:
                    fields.extend(get_struct_type(ty, type_quals,
                                                  decl.declarators))
                else:
                    global anonymous_id
                    anonymous_id += 1
                    fields.append((ty, f"struct.anon{anonymous_id}", None))
            if qual_spec.ident:
                name = qual_spec.ident.val
            else:
                anonymous_id += 1
                name = f"struct.anon{anonymous_id}"
            ty = sym_table.register_composite_type(name, fields, is_union).ty
        else:
            if qual_spec.ident:
                ty = sym_table.find_type(qual_spec.ident.val)
            return None

        assert(ty)

        return ty

    if spec.ty in [TypeSpecifierType.Int, TypeSpecifierType.Double]:
        if spec.width == TypeSpecifierWidth.Long:
            ty_name = "long " + ty_name

    if spec.ty in [TypeSpecifierType.Char, TypeSpecifierType.Short, TypeSpecifierType.Int, TypeSpecifierType.Long]:
        if spec.sign == TypeSpecifierSign.Unsigned:
            ty_name = "unsigned " + ty_name

    return sym_table.find_type(ty_name)


def get_decl_name_and_type(ty, decl, sym_table):
    from ast.types import PointerType, FunctionType, ArrayType, VoidType, PrimitiveType

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
            if isinstance(ty, VoidType):
                ty = PrimitiveType("int")
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
                param_quals = param_ty.quals

                param_ty = get_type(sym_table, param_ty)
                param_name, param_ty = get_decl_name_and_type(
                    param_ty, param_decl, sym_table)

                params.append((param_ty, param_quals, param_name))

            ty = FunctionType(ty, params, decl.function_is_variadic)

    return name, ty


def register_function(qual_spec, declarator, sym_table, is_constructor=False):
    from ast.types import PointerType, FunctionType

    ty = get_type(sym_table, qual_spec)
    name, ty = get_decl_name_and_type(ty, declarator, sym_table)

    if isinstance(ty, FunctionType):
        if is_constructor:
            name = FuncSignature(name, [param[0] for param in ty.params])

        sym = sym_table.register_func(
            name, ty.return_ty, ty.params, ty.is_variadic)

        return TypedFunctionProto(ty.return_ty, sym, ty.params, qual_spec.func_spec)

    assert(isinstance(ty, PointerType))
    sym_table.register_alias_type(name, ty, [])

    return None


def evaluate_constant_expr(expr):
    if isinstance(expr, IntegerConstantExpr):
        return expr.val

    raise ValueError()


def register_variable(node, sym_table):
    from ast.types import ArrayType

    ty = sym_table.find_type(
        node.type.specifier, node.type.array_specifier)

    if isinstance(node.type, FullType):
        quals = node.type.qualifiers
    else:
        quals = []

    variables = []
    for ident, arr_spec, initializer in node.idents:
        ty2 = ty
        if arr_spec:
            ty2 = ty
            for sz in arr_spec.sizes:
                ty2 = ArrayType(ty2, evaluate_constant_expr(sz))

        assert(ty2)
        var = sym_table.register_var(ident.val, ty2, quals)

        variables.append([TypedIdentExpr(var, ty2), initializer])

    return TypedVariable(ty, variables, [])


def enter_node(node, depth, sym_table: SymbolTable):
    if isinstance(node, Function):
        proto = register_function_proto(node.proto, sym_table)

        param_names = []
        for param in node.proto.params:
            param_name = param.ident.val if param.ident is not None else None
            param_names.append(param_name)

        sym_table.push_scope()

        params = []
        for param_names, (param_ty, param_quals, param_name) in zip(param_names, proto.params):
            if param_names is not None:
                assert(param_ty)
                params.append(sym_table.register_var(
                    param_names, param_ty, param_quals))

        assert(proto)

        node = TypedFunction(proto, params, node.stmts)

    if isinstance(node, Declaration):
        if node.decls:
            for decl, init in node.decls:
                if decl.is_function_decl:
                    decl_params = decl.function_params

                    if not decl_params:
                        decl_params = []

                    params = []
                    for param_ty, param_decl in decl_params:
                        ty = get_type(sym_table, param_ty)

                        param_name, var_ty = get_decl_name_and_type(
                            ty, param_decl, sym_table)

                        assert(var_ty)
                        params.append(sym_table.register_var(
                            param_name, var_ty, None))

                    func_decl = register_function(
                        node.qual_spec, decl, sym_table)

                    if func_decl:
                        node = TypedFunction(func_decl, params, None)
                        sym_table.push_scope()

    if isinstance(node, FunctionDecl):
        decl_params = node.declarator.function_params
        decl = register_function(node.qual_spec, node.declarator, sym_table)

        if decl:
            param_names = []
            for param_ty, param_decl in decl_params:
                param_name = param_decl.ident_or_decl.val if param_decl and param_decl.ident_or_decl else None
                param_names.append(param_name)

            sym_table.push_scope()

            params = []
            for param_name, (param_ty, param_decl) in zip(param_names, decl_params):
                if param_name is not None:
                    ty = get_type(sym_table, param_ty)

                    param_name, var_ty = get_decl_name_and_type(
                        ty, param_decl, sym_table)

                    assert(var_ty)
                    params.append(sym_table.register_var(
                        param_name, var_ty, None))

            assert(decl)
            node = TypedFunction(decl, params, node.stmt.stmts)

    if isinstance(node, FunctionProto):
        node = register_function_proto(node, sym_table)

    if isinstance(node, Variable):
        register_variable(node, sym_table)

    if isinstance(node, StructSpecifier):
        fields = []
        for decl in node.decls:
            for declor in decl.declarators:
                ty = sym_table.find_type(decl.type.specifier)
                fields.append((ty, declor.ident, declor.arrspec))

        assert(node.ident)
        sym_table.register_composite_type(node.ident, fields)

    if isinstance(node, CompoundStmt):
        sym_table.push_scope()

    if isinstance(node, IfStmt):
        sym_table.push_scope()

    return node


def is_integer_type(t):
    from ast.types import PrimitiveType

    if not isinstance(t, PrimitiveType):
        return False

    return t.name in [
        'char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long']


def is_incr_func_type(t):
    return t in ['char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long']


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
    ('unsigned int', 'int'),

    ('unsigned long', '_Bool'),
    ('unsigned long', 'short'),
    ('unsigned long', 'unsigned short'),
    ('unsigned long', 'unsigned int'),

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
    return op in ['+', '-', '*', '/']


def promote_default_type(ty):
    from ast.types import PrimitiveType

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
    if node.type != ty.name:
        return CastExpr(node, ty)

    return node


def compute_binary_compare_op_type(op, lhs, rhs, sym_table):
    lhs_type = lhs.type
    rhs_type = rhs.type

    lhs_type = promote_default_type(lhs_type)
    rhs_type = promote_default_type(rhs_type)

    result_ty = sym_table.find_type('int')
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


def compute_binary_logical_op_type(op, lhs_type, rhs_type, sym_table):
    # Logical operators
    if lhs_type.name == 'int' and rhs_type.name == 'int':
        return lhs_type

    raise Exception("Unsupporting operation.")


def is_assignment_op(op):
    return op in ['=', '+=', '-=', '*=', '/=', '>>=', '<<=']


def compute_binary_assignment_op_type(op, lhs_type, rhs_type, sym_table):
    if lhs_type == rhs_type:
        return lhs_type

    able_conv_rhs = [to_type for to_type,
                     from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]
    if lhs_type.name in able_conv_rhs:
        return lhs_type

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
        return compute_binary_logical_op_type(op, lhs_type, rhs_type, sym_table), lhs, rhs

    if is_shift_op(op):
        return compute_binary_shift_op_type(op, lhs_type, rhs_type, sym_table), lhs, rhs

    if is_assignment_op(op):
        return compute_binary_assignment_op_type(op, lhs_type, rhs_type, sym_table), lhs, rhs

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


def exit_node(node, depth, sym_table):
    from ast.types import ArrayType

    if isinstance(node, TypedFunction):
        sym_table.pop_scope()

    if isinstance(node, CompoundStmt):
        sym_table.pop_scope()

    if isinstance(node, IfStmt):
        sym_table.pop_scope()

    if isinstance(node, IntegerConstantExpr):
        ty = sym_table.find_type(node.type.specifier)
        assert(is_integer_type(ty))
        node = IntegerConstantExpr(node.val, ty)

    if isinstance(node, FloatingConstantExpr):
        ty = sym_table.find_type(node.type.specifier)
        assert(is_float_type(ty))
        node = FloatingConstantExpr(node.val, ty)

    if isinstance(node, Variable):
        variables = []
        ty = sym_table.find_type(node.type.specifier)
        for ident, arr_spec, initializer in node.idents:
            ty2 = ty
            if arr_spec:
                ty2 = ty
                for sz in arr_spec.sizes:
                    ty2 = ArrayType(ty2, evaluate_constant_expr(sz))
            var = sym_table.find_var(ident.val)

            assert(ty2)
            variables.append([TypedIdentExpr(var, ty2), initializer])

        node = TypedVariable(ty, variables, [])

    from ast.types import PointerType, CompositeType, PrimitiveType
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

                    _, expr = init.exprs[expr_idx]

                    if isinstance(field_ty, (ArrayType, CompositeType, PointerType)):
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

                    _, expr = init.exprs[expr_idx]

                    if isinstance(field_ty, (ArrayType, CompositeType, PointerType)):
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

                    if isinstance(field_ty, (ArrayType, CompositeType, PointerType)):
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

    if isinstance(node, Declaration):
        variables = []

        is_typedef = StorageClass.Typedef in node.qual_spec.quals
        is_extern = StorageClass.Extern in node.qual_spec.quals
        is_static = StorageClass.Static in node.qual_spec.quals

        ty = get_type(sym_table, node.qual_spec)
        quals = node.qual_spec.quals

        if node.decls:
            for decl, init in node.decls:
                if decl.is_function_decl:
                    continue

                name, var_ty = get_decl_name_and_type(ty, decl, sym_table)

                if is_typedef:
                    sym_table.register_alias_type(name, var_ty, quals)
                else:
                    if init:
                        init, _ = get_type_initializer(init, var_ty)

                        if isinstance(var_ty, PointerType) and isinstance(init.type, ArrayType):
                            var_ty = init.type

                    assert(var_ty)
                    var = sym_table.register_var(name, var_ty, quals)
                    variables.append([TypedIdentExpr(var, var_ty), init])

        storage_class = []
        if is_extern:
            storage_class.append("extern")
        if is_static:
            storage_class.append("static")

        if not is_typedef:
            node = TypedVariable(ty, variables, storage_class)

    if isinstance(node, IdentExpr):
        var = sym_table.find_var(node.val)
        if var:
            assert(var.ty)
            typed_node = TypedIdentExpr(var, var.ty)

        func_name = mangle_func_name(node.val)
        func = sym_table.find_func(func_name)
        if func:
            typed_node = TypedIdentExpr(func, func.ty)

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

        type_spec = get_type_spec_type(node.type.qual_spec)

        if not type_spec:
            return_type = sym_table.find_type(node.type.qual_spec[0].val)
        else:
            return_type = get_type(
                sym_table, TypeSpecifier(type_spec, [], None, None))

        if node.type.pointer:
            return_type = PointerType(return_type)

        node = CastExpr(expr, return_type)

    from ast.types import Type

    if isinstance(node, SizeOfExpr):
        return_type = sym_table.find_type("unsigned int")

        if node.type:
            if len(node.type.qual_spec) == 1 and isinstance(node.type.qual_spec[0], Ident):
                type_spec = TypeSpecifier(None, [], node.type.qual_spec[0], [])
            else:
                spec_type = get_type_spec_type(node.type.qual_spec)
                type_spec = TypeSpecifier(spec_type, [], None, [])

            ty = get_type(sym_table, type_spec)
            for _ in node.type.pointer:
                ty = PointerType(ty)
        else:
            ty = node.type
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

        assert(isinstance(return_type, Type))
        ty = return_type

        node = TypedBinaryOp(node.op, lhs, rhs, ty)

    if isinstance(node, ConditionalExpr):
        assert(is_typed_expr(node.true_expr))
        assert(is_typed_expr(node.false_expr))
        assert(is_typed_expr(node.cond_expr))

        true_expr = node.true_expr
        false_expr = node.false_expr
        cond_expr = node.cond_expr

        result_type = None

        if true_expr.type == false_expr.type:
            result_type = true_expr.type
        elif is_implicit_convertable(true_expr.type.name, false_expr.type.name):
            result_type = false_expr.type
        elif is_implicit_convertable(false_expr.type.name, true_expr.type.name):
            result_type = true_expr.type

        assert(result_type)

        ty = result_type

        node = TypedConditionalExpr(cond_expr, true_expr, false_expr, ty)

    if isinstance(node, UnaryOp):
        assert(is_typed_expr(node.expr))

        if node.op in ["++", "--"]:
            expr_typename = node.expr.type.name

            if not is_incr_func_type(expr_typename):
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
            return_ty = node.expr.type

        if node.op in ["+", "-"]:
            return_ty = node.expr.type

        node = TypedUnaryOp(node.op, node.expr, return_ty)

    if isinstance(node, PostOp):
        assert(is_typed_expr(node.expr))

        expr_typename = node.expr.type.name

        if not is_incr_func_type(expr_typename):
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
            obj_typename, node.field.val)

        if field_type is None:
            raise RuntimeError(
                f"Error: Invalid field access \"{node.field.val}\" in \"{obj_typename}\"")

        node = TypedAccessorOp(node.obj, node.field, field_type[0])

    if isinstance(node, FunctionCall):
        if isinstance(node.ident, TypedIdentExpr):
            func_name = node.ident.val.name
        elif isinstance(node.ident, Type):
            func_name = node.ident.specifier
        else:
            raise NotImplementedError()

        func_name = mangle_func_name(func_name)

        func_def = sym_table.find_func(func_name)

        if func_def is None:
            raise RuntimeError(
                f"Error: The function \"{func_name}\" not defined.")

        func_type = func_def.ty.return_ty
        formal_params = func_def.ty.params

        # check parameter
        pos = 0
        for (param_type, param_quals, param_name), arg in zip(formal_params, node.params):
            if is_integer_type(param_type) == is_integer_type(arg.type):
                pos += 1
                continue

            if isinstance(param_type, (PointerType, ArrayType)) and isinstance(arg.type, (PointerType, ArrayType)):
                pos += 1

            raise RuntimeError(
                f"Error: Invalid type \"{arg.type.name}\" found at position \"{pos}\" must be \"{param_type.name}\".")

        node = TypedFunctionCall(func_def, node.params, func_type)

    if isinstance(node, AsmStmt):
        operands_list = []
        for operands in node.operands:
            new_operands = []
            for constraint, name in operands:
                if name:
                    new_operands.append(
                        (constraint, sym_table.find_var(name.value)))
                else:
                    new_operands.append(
                        (constraint, None))

            operands_list.append(new_operands)

        node = AsmStmt(node.template, operands_list)

    return node


class FuncSignature:
    def __init__(self, name, param_types):
        self.name = name
        self.param_types = param_types

    def __hash__(self):
        return hash((self.name, *self.param_types))

    def __eq__(self, other):
        if not isinstance(other, FuncSignature):
            return False

        return self.name == other.name and self.param_types == other.param_types

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.name + "_" + str(len(self.param_types))


def semantic_analysis(ast):
    sym_table = SymbolTable()
    analyzed = traverse_depth_update(ast, enter_node, exit_node, sym_table)
    return (analyzed, sym_table)
