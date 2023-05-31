#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.ast.node import *
from rachetwrench.glsl.symtab import SymbolTable, Symbol


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

    return TypedFunctionProto(ty, sym, params, [])


def evaluate_constant_expr(expr):
    if isinstance(expr, IntegerConstantExpr):
        return expr.val

    raise ValueError()


def register_variable(node, sym_table):
    from rachetwrench.ast.types import ArrayType

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
        var = sym_table.register_var(ident.val, ty2, quals)

        assert(ty2)
        variables.append([TypedIdentExpr(var, ty2), initializer])

    return TypedVariable(ty, variables, [])


def enter_node(node, depth, sym_table: SymbolTable):
    from rachetwrench.ast.types import ArrayType

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
                params.append(sym_table.register_var(
                    param_names, param_ty, param_quals))

        node = TypedFunction(proto, params, node.stmts)

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

        sym_table.register_composite_type(node.ident, fields)

    if isinstance(node, CompoundStmt):
        sym_table.push_scope()

    if isinstance(node, IfStmt):
        sym_table.push_scope()

    return node


def is_incr_func_type(t):
    return t in ['int', 'uint']


scalar_convertable = [
    # (to, from)
    ('int', 'uint'),
    ('int', 'bool'),
    ('int', 'float'),
    ('int', 'double'),

    ('uint', 'int'),
    ('uint', 'bool'),
    ('uint', 'float'),
    ('uint', 'double'),

    ('bool', 'int'),
    ('bool', 'uint'),
    ('bool', 'float'),
    ('bool', 'double'),

    ('float', 'int'),
    ('float', 'uint'),
    ('float', 'bool'),
    ('float', 'double'),

    ('double', 'int'),
    ('double', 'uint'),
    ('double', 'bool'),
    ('double', 'float'),
]

implicit_convertable = [
    # (to, from)
    ('uint', 'int'),

    ('float', 'int'),
    ('float', 'uint'),

    ('double', 'int'),
    ('double', 'uint'),
    ('double', 'float'),

    # TODO: Not implemented. Need additional items.
]

integer_types = [
    'int', 'uint'
]


integer_vec_types = [
    # TODO: Not implemented.
]


def compute_binary_op_type(op, lhs_type, rhs_type, sym_table):
    from rachetwrench.ast.types import PrimitiveType, VectorType

    lhs_is_vec = isinstance(lhs_type, VectorType)
    rhs_is_vec = isinstance(rhs_type, VectorType)

    if lhs_is_vec or rhs_is_vec:
        if lhs_is_vec and rhs_is_vec:
            assert(lhs_type.size == rhs_type.size)
            size = lhs_type.size
        elif lhs_is_vec:
            size = lhs_type.size
        else:
            size = rhs_type.size

    if lhs_is_vec:
        lhs_type = lhs_type.elem_ty
    if rhs_is_vec:
        rhs_type = rhs_type.elem_ty

    # Arithematic operators
    if op in ['+', '-', '*', '/']:
        able_conv_lhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == lhs_type.name] + [lhs_type.name]

        if rhs_type.name in able_conv_lhs:
            if lhs_is_vec or rhs_is_vec:
                return VectorType(rhs_type, size)
            return rhs_type

        able_conv_rhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]

        if lhs_type.name in able_conv_rhs:
            if lhs_is_vec or rhs_is_vec:
                return VectorType(lhs_type, size)
            return lhs_type

    # Bitwise operators
    if op in ['&', '^', '|']:
        able_conv_lhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == lhs_type.name] + [lhs_type.name]

        if rhs_type.name in able_conv_lhs:
            return rhs_type

        able_conv_rhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]

        if lhs_type.name in able_conv_rhs:
            return lhs_type

    if op in ['==', '!=', '<', '>', '<=', '>=']:
        able_conv_lhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == lhs_type.name] + [lhs_type.name]

        if rhs_type.name in able_conv_lhs:
            return sym_table.find_type('bool')

        able_conv_rhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]

        if lhs_type.name in able_conv_rhs:
            return sym_table.find_type('bool')

    if op in ['<<', '>>']:
        if lhs_type.name in integer_types and rhs_type.name in integer_types:
            return lhs_type

        if lhs_type.name in integer_vec_types and rhs_type.name in integer_vec_types:
            return lhs_type

    # Logical operators
    if op in ['&&', '^^', '||']:
        if lhs_type.name == 'bool' and rhs_type.name == 'bool':
            return lhs_type

    if op in ['=', '+=', '-=', '*=', '/=']:
        able_conv_rhs = [to_type for to_type,
                         from_type in implicit_convertable if from_type == rhs_type.name] + [rhs_type.name]
        if lhs_type.name in able_conv_rhs:
            return lhs_type

    return None


def is_scalar_convertable(from_type: str, to_type: str):
    return (to_type, from_type) in scalar_convertable


def is_implicit_convertable(from_type: str, to_type: str):
    return (to_type, from_type) in implicit_convertable


def is_typed_expr(node):
    return isinstance(node, (
        TypedBinaryOp,
        TypedIdentExpr,
        IntegerConstantExpr,
        FloatingConstantExpr,
        TypedUnaryOp,
        TypedPostOp,
        TypedAccessorOp,
        TypedArrayIndexerOp,
        TypedFunctionCall))


def is_constructor(name):
    return name in [
        "float", "vec2", "vec3", "vec4"
    ]


def mangle_func_name(name):
    if name in ["sin", "cos", "abs", "sqrt", "mod", "min", "max"]:
        return "glsl_" + name

    return name


def exit_node(node, depth, sym_table):
    from rachetwrench.ast.types import ArrayType

    if isinstance(node, TypedFunction):
        sym_table.pop_scope()

    if isinstance(node, CompoundStmt):
        sym_table.pop_scope()

    if isinstance(node, IfStmt):
        sym_table.pop_scope()

    if isinstance(node, IntegerConstantExpr):
        ty = sym_table.find_type(node.type.specifier)
        assert(ty.name in ["int", "uint"])
        node = IntegerConstantExpr(node.val, ty)

    if isinstance(node, FloatingConstantExpr):
        ty = sym_table.find_type(node.type.specifier)
        assert(ty.name in ["float", "double"])
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

    if isinstance(node, IdentExpr):
        var = sym_table.find_var(node.val)
        if var is not None:
            assert(var.ty)
            typed_node = TypedIdentExpr(var, var.ty)

        func_name = mangle_func_name(node.val)
        func = sym_table.find_func(func_name)
        if func is not None:
            typed_node = TypedIdentExpr(func, func.ty)

        if typed_node is None:
            raise RuntimeError(f"Error: Undefined identity \"{node.val}\"")

        node = typed_node

    if isinstance(node, BinaryOp):
        from rachetwrench.ast.types import PrimitiveType, VectorType

        assert(is_typed_expr(node.lhs))
        assert(is_typed_expr(node.rhs))

        lhs = node.lhs
        rhs = node.rhs

        lhs_type = lhs.type
        rhs_type = rhs.type

        return_type = compute_binary_op_type(
            node.op, lhs_type, rhs_type, sym_table)

        if return_type is None:
            raise RuntimeError(
                f"Error: Undefined operation between types \"{lhs_typename}\" and \"{rhs_typename}\" with op \"{node.op}\"")

        ty = sym_table.find_type(return_type.name)

        # if lhs_typename != return_type:
        #     lhs = CastExpr(node.lhs, ty)

        # if rhs_typename != return_type:
        #     rhs = CastExpr(node.rhs, ty)

        node = TypedBinaryOp(node.op, lhs, rhs, ty)

    if isinstance(node, ConditionalExpr):
        assert(is_typed_expr(node.true_expr))
        assert(is_typed_expr(node.false_expr))
        assert(is_typed_expr(node.cond_expr))

        true_expr = node.true_expr
        false_expr = node.false_expr
        cond_expr = node.cond_expr

        assert(true_expr.type == false_expr.type)
        assert(cond_expr.type.name == "bool")

        ty = sym_table.find_type(true_expr.type.name)

        node = TypedConditionalExpr(
            node.cond_expr, node.true_expr, node.false_expr, ty)

    if isinstance(node, UnaryOp):
        assert(is_typed_expr(node.expr))

        expr_typename = node.expr.type.name

        if node.op in ["++", "--"]:
            if not is_incr_func_type(expr_typename):
                raise RuntimeError(
                    f"Error: Not supporting types with increment \"{expr_typename}\"")

        node = TypedUnaryOp(node.op, node.expr, node.expr.type)

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

        assert(isinstance(arr_type, ArrayType))

        node = TypedArrayIndexerOp(node.arr, node.idx, arr_type.elem_ty)

    if isinstance(node, AccessorOp):
        assert(is_typed_expr(node.obj))

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
        arg_tys = [arg.type for arg in node.params]

        if is_constructor(func_name):
            func_name = FuncSignature(func_name, arg_tys)

        func_def = sym_table.find_func(func_name)

        if func_def is None:
            raise RuntimeError(
                f"Error: The function \"{func_name}\" not defined.")

        func_type = func_def.ty.return_ty
        func_params = func_def.ty.params

        # check parameter
        pos = 0
        for (param_type, param_quals, param_name), arg in zip(func_params, node.params):
            if not (param_type.name == arg.type.name
                    or is_implicit_convertable(arg.type.name, param_type.name)):
                raise RuntimeError(
                    f"Error: Invalid type \"{arg.type.name}\" found at position \"{pos}\" must be \"{param_type.name}\".")

            pos += 1

        node = TypedFunctionCall(func_def, node.params, func_type)

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


def setup_buildin_decls(sym_table):
    # vec constructor
    for i in range(1, 4):
        size = i + 1

        params = [FunctionParam(Type('float', None), None, None)] * size
        proto = FunctionProto(
            Type(f'vec{size}', None), Ident(f'vec{size}'), params)

        yield register_function_proto(proto, sym_table, True)

        params = [FunctionParam(Type('float', None), None, None)]
        proto = FunctionProto(
            Type(f'vec{size}', None), Ident(f'vec{size}'), params)

        yield register_function_proto(proto, sym_table, True)

    for i in range(2, 4):
        size = i + 1

        params = [FunctionParam(Type(f'vec{size-1}', None), None, None),
                  FunctionParam(Type('float', None), None, None)]

        proto = FunctionProto(
            Type(f'vec{size}', None), Ident(f'vec{size}'), params)

        yield register_function_proto(proto, sym_table, True)

    # dot
    proto = FunctionProto(Type('float', None), Ident('dot'), [
                          FunctionParam(Type('vec3', None), None, None),
                          FunctionParam(Type('vec3', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # normalize
    proto = FunctionProto(Type('vec3', None), Ident('normalize'), [
                          FunctionParam(Type('vec3', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # reflect
    proto = FunctionProto(Type('vec3', None), Ident('reflect'), [
                          FunctionParam(Type('vec3', None), None, None),
                          FunctionParam(Type('vec3', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # clamp
    proto = FunctionProto(Type('float', None), Ident('clamp'), [
                          FunctionParam(Type('float', None), None, None),
                          FunctionParam(Type('float', None), None, None),
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # fract
    proto = FunctionProto(Type('float', None), Ident('glsl_fract'), [
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # mod
    proto = FunctionProto(Type('float', None), Ident('glsl_mod'), [
                          FunctionParam(Type('float', None), None, None),
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # min
    proto = FunctionProto(Type('float', None), Ident('glsl_min'), [
                          FunctionParam(Type('float', None), None, None),
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # max
    proto = FunctionProto(Type('float', None), Ident('glsl_max'), [
                          FunctionParam(Type('float', None), None, None),
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # sqrt
    proto = FunctionProto(Type('float', None), Ident('glsl_sqrt'), [
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # abs
    proto = FunctionProto(Type('float', None), Ident('glsl_abs'), [
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # sin
    proto = FunctionProto(Type('float', None), Ident('glsl_sin'), [
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # cos
    proto = FunctionProto(Type('float', None), Ident('glsl_cos'), [
                          FunctionParam(Type('float', None), None, None)])

    yield register_function_proto(proto, sym_table)

    # memoryBarrier
    proto = FunctionProto(Type('void', None), Ident('memoryBarrier'), [])

    yield register_function_proto(proto, sym_table)

    variable = Variable(FullType(["uniform"], 'vec3', None), [
                        [IdentExpr('gl_FragCoord', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["uniform"], 'vec4', None), [
                        [IdentExpr('gl_FragColor', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["uniform"], 'uvec3', None), [
                        [IdentExpr('gl_NumWorkGroups', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["uniform"], 'uvec3', None), [
                        [IdentExpr('gl_WorkGroupID', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["uniform"], 'uvec3', None), [
                        [IdentExpr('gl_LocalInvocationID', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["uniform"], 'uvec3', None), [
                        [IdentExpr('gl_GlobalInvocationID', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["uniform"], 'uint', None), [
                        [IdentExpr('gl_LocalInvocationIndex', ), None, None]])

    yield register_variable(variable, sym_table)

    variable = Variable(FullType(["const"], 'uvec3', None), [
                        [IdentExpr('gl_WorkGroupSize', ), None, None]])

    yield register_variable(variable, sym_table)


def semantic_analysis(ast):
    sym_table = SymbolTable()
    buildin_decls = list(setup_buildin_decls(sym_table))
    analyzed = buildin_decls + \
        traverse_depth_update(ast, enter_node, exit_node, sym_table)
    return (analyzed, sym_table)
