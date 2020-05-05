#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ast.node import *
from ir.types import *
from ir.values import *
from ir.data_layout import *
import ast


def evaluate_constant_expr(ctx, expr):
    if isinstance(expr, TypedFunctionCall):
        if expr.ident.name.name == "vec3":
            params = []
            for param in expr.params:
                param_expr = evaluate_constant_expr(ctx, param)
                assert(isinstance(param_expr, Constant))
                params.append(param_expr)

            if len(params) == 3:
                param_w = ConstantFP(0.0, f32)
                return ConstantVector(params + [param_w], VectorType("", f32, 4))

            if len(params) == 1:
                return ConstantVector(params * 4, VectorType("", f32, 4))

    if isinstance(expr, FloatingConstantExpr):
        return ConstantFP(expr.val, ctx.get_ir_type(expr.type))

    if isinstance(expr, IntegerConstantExpr):
        return ConstantInt(expr.val, ctx.get_ir_type(expr.type))

    if isinstance(expr, TypedUnaryOp):
        op = expr.op

        val = evaluate_constant_expr(ctx, expr.expr)
        if op == "-":
            def calc(a): return -a

        if isinstance(val, ConstantInt):
            return ConstantInt(calc(val.value), val.ty)
        elif isinstance(val, ConstantFP):
            return ConstantFP(calc(val.value), val.ty)
        else:
            raise ValueError("Invalid type")

    if isinstance(expr, TypedBinaryOp):
        op = expr.op

        lhs_val = evaluate_constant_expr(ctx, expr.lhs)
        rhs_val = evaluate_constant_expr(ctx, expr.rhs)
        if op == "+":
            def calc(a, b): return a + b
        elif op == "-":
            def calc(a, b): return a - b
        elif op == "*":
            def calc(a, b): return a * b
        elif op == "/":
            def calc(a, b): return a / b

        if isinstance(lhs_val, ConstantInt):
            return ConstantInt(calc(lhs_val.value, rhs_val.value), lhs_val.ty)
        elif isinstance(lhs_val, ConstantFP):
            return ConstantFP(calc(lhs_val.value, rhs_val.value), lhs_val.ty)
        else:
            raise ValueError("Invalid type")

    if isinstance(expr, TypedIdentExpr):
        if not isinstance(expr.type, ast.types.EnumType):
            raise ValueError("Invalid type")

        value = expr.type.values[expr.val.name]

        return ConstantInt(value, i32)

    if isinstance(expr, TypedInitializerList):
        field_idx = 0
        if isinstance(expr.type, ast.types.CompositeType):
            field_vals = []
            for field_ty, field_name, _ in expr.type.fields:
                _, field_val = expr.exprs[field_idx]
                field_idx += 1
                field_vals.append(evaluate_constant_expr(ctx, field_val))
            return ConstantStruct(field_vals, ctx.get_ir_type(expr.type))
        elif isinstance(expr.type, (ast.types.ArrayType, ast.types.PointerType)):
            values = []
            for _, elem_expr in expr.exprs:
                values.append(evaluate_constant_expr(ctx, elem_expr))

            ty = ctx.get_ir_type(expr.type)
            ty = ArrayType(ty.elem_ty, len(expr.exprs))

            return ConstantArray(values, ty)

    if isinstance(expr, CastExpr):
        return evaluate_constant_expr(ctx, expr.expr)

    raise NotImplementedError()


class FuncContext:
    def __init__(self):
        self.continue_target = []
        self.break_target = []
        self.return_target = []
        self.branch_target = []
        self.preds = []
        self.named_values = {}
        self.return_value = None


class Context:
    def __init__(self, module):
        self.module = module
        self.func_ctx = []
        self.funcs = {}

    @property
    def in_inline(self):
        return len(self.func_ctx) >= 2

    @property
    def return_value(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].return_value

    @return_value.setter
    def return_value(self, value):
        if not self.func_ctx:
            raise ValueError()

        self.func_ctx[-1].return_value = value

    @property
    def named_values(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].named_values

    @property
    def continue_target(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].continue_target

    @property
    def break_target(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].break_target

    @property
    def return_target(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].return_target

    @property
    def branch_target(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].branch_target

    @property
    def preds(self):
        if not self.func_ctx:
            return None

        return self.func_ctx[-1].preds

    def begin_func(self):
        self.func_ctx.append(FuncContext())

    def end_func(self):
        self.func_ctx.pop()

    def push_continue_target(self, block):
        self.continue_target.append(block)

    def pop_continue_target(self):
        return self.continue_target.pop()

    def get_continue_target(self):
        return self.continue_target[-1]

    def push_break_target(self, block):
        self.break_target.append(block)

    def pop_break_target(self):
        return self.break_target.pop()

    def get_break_target(self):
        return self.break_target[-1]

    def push_return_target(self, block):
        self.return_target.append(block)

    def pop_return_target(self):
        return self.return_target.pop()

    def get_return_target(self):
        return self.return_target[-1]

    def push_branch_target(self, block):
        self.branch_target.append(block)

    def pop_branch_target(self):
        return self.branch_target.pop()

    def get_branch_target(self):
        return self.branch_target[-1]

    def get_ir_type(self, ast_ty):
        if isinstance(ast_ty, ast.types.CompositeType):
            if ast_ty.is_union:
                default_ty, _, _ = ast_ty.fields[0]
                return self.get_ir_type(default_ty)

        if isinstance(ast_ty, ast.types.PrimitiveType):
            if ast_ty.name in ["_Bool"]:
                return i1
            if ast_ty.name in ["unsigned char", "char"]:
                return i8
            if ast_ty.name in ["unsigned short", "short"]:
                return i16
            if ast_ty.name in ["unsigned int", "int"]:
                return i32
            if ast_ty.name in ["unsigned long", "long", "unsigned long long", "long long"]:
                return i64
            if ast_ty.name in ["float"]:
                return f32
            if ast_ty.name in ["double", "long double"]:
                return f64
        if isinstance(ast_ty, ast.types.CompositeType):
            if ast_ty.name and self.module.contains_struct_type(ast_ty.name):
                return self.module.structs[ast_ty.name]

            fields = [self.get_ir_type(ty) for ty, name, arr in ast_ty.fields]
            ty = StructType(ast_ty.name, fields)
            assert(ast_ty.name)
            self.module.add_struct_type(ast_ty.name, ty)
            return ty

        if isinstance(ast_ty, ast.types.VoidType):
            return void

        if isinstance(ast_ty, ast.types.FunctionType):
            func_info = compute_func_info(self, ast_ty)

            params_ty = []

            return_ty = func_info.return_info.ty
            if func_info.return_info.kind == ABIArgKind.Direct:
                pass
            elif func_info.return_info.kind in [ABIArgKind.Indirect, ABIArgKind.Ignore]:
                params_ty.append(return_ty)
                return_ty = void
            else:
                raise NotImplementedError()

            params_ty.extend([self.get_ir_type(ast_param_ty)
                              for (ast_param_ty, _, _) in ast_ty.params])

            return FunctionType(None, return_ty, params_ty, ast_ty.is_variadic)

        if isinstance(ast_ty, ast.types.VectorType):
            elem_ty = self.get_ir_type(ast_ty.elem_ty)
            return VectorType(ast_ty.name, elem_ty, ast_ty.size)

        if isinstance(ast_ty, ast.types.ArrayType):
            elem_ty = self.get_ir_type(ast_ty.elem_ty)
            return ArrayType(elem_ty, ast_ty.size)

        if isinstance(ast_ty, ast.types.PointerType):
            if not ast_ty.elem_ty or isinstance(ast_ty.elem_ty, ast.types.VoidType):
                return PointerType(i8, 0)

            elem_ty = self.get_ir_type(ast_ty.elem_ty)
            return PointerType(elem_ty, 0)

        if isinstance(ast_ty, ast.types.EnumType):
            return i32

        raise Exception

    def get_memcpy_func(self, rd, rs, size):
        arg_tys = [rd.ty, rs.ty, i32, i32, i1]
        return self.module.get_or_declare_intrinsic_func("llvm.memcpy", arg_tys)

    def get_va_start_func(self, ptr):
        arg_tys = [PointerType(i8, 0)]
        return self.module.get_or_declare_intrinsic_func("llvm.va_start", arg_tys)

    def get_va_end_func(self, ptr):
        arg_tys = [PointerType(i8, 0)]
        return self.module.get_or_declare_intrinsic_func("llvm.va_end", arg_tys)


def get_lvalue(node, block, ctx):
    if isinstance(node, TypedIdentExpr):
        return block, ctx.named_values[node.val]

    if isinstance(node, TypedUnaryOp):
        if node.op == "*":
            block, expr = get_lvalue(node.expr, block, ctx)
            return block, LoadInst(block, expr)

    if isinstance(node, TypedCommaOp):
        return block, get_lvalue(node.exprs[-1], block, ctx)

    def get_field_ptr(ty, field):
        assert(isinstance(ty, ast.types.CompositeType))

        if ty.contains_field(field):
            field_idx = ty.get_field_idx(field)
            if ty.is_union:
                return [(None, ty.get_field_type_by_name(field))]
            return [(field_idx, None)]

        for idx, (field_ty, field_name, _) in enumerate(ty.fields):
            if field_name.startswith("struct.anon"):
                result = get_field_ptr(field_ty, field)
                if result:
                    if ty.is_union:
                        return [(None, field_ty)] + result
                    return [(idx, None)] + result

        return []

    if isinstance(node, TypedAccessorOp):
        block, ptr = get_lvalue(node.obj, block, ctx)
        obj_ty = ctx.get_ir_type(node.obj.type)

        if isinstance(obj_ty, PointerType):
            ptr = LoadInst(block, ptr)

            idx_or_casts = get_field_ptr(node.obj.type.elem_ty, node.field.val)

            idxs = []
            for idx, cast_ty in idx_or_casts:
                if idx is not None:
                    idxs.append(ConstantInt(idx, i32))

                if cast_ty:
                    ptr = GetElementPtrInst(
                        block, ptr, ptr.ty, ConstantInt(0, i32), *idxs)
                    ptr = BitCastInst(block, ptr, PointerType(
                        ctx.get_ir_type(cast_ty), 0))
                    idxs = []

            inst = GetElementPtrInst(
                block, ptr, ptr.ty, ConstantInt(0, i32), *idxs)
        elif isinstance(obj_ty, CompositeType):
            idx_or_casts = get_field_ptr(node.obj.type, node.field.val)

            idxs = []
            for idx, cast_ty in idx_or_casts:
                if idx is not None:
                    idxs.append(ConstantInt(idx, i32))

                if cast_ty:
                    ptr = GetElementPtrInst(
                        block, ptr, ptr.ty, ConstantInt(0, i32), *idxs)
                    ptr = BitCastInst(block, ptr, PointerType(
                        ctx.get_ir_type(cast_ty), 0))
                    idxs = []

            inst = GetElementPtrInst(
                block, ptr, ptr.ty, ConstantInt(0, i32), *idxs)
        else:
            raise ValueError()

        ctx.named_values[node] = inst

        return block, inst

    if isinstance(node, TypedArrayIndexerOp):
        if node in ctx.named_values:
            return block, ctx.named_values[node]

        ty = ctx.get_ir_type(node.arr.type)

        block, ptr = get_lvalue(node.arr, block, ctx)
        if isinstance(ptr.ty.elem_ty, ArrayType):

            block, idx = build_ir_expr(node.idx, block, ctx)
            inst = GetElementPtrInst(
                block, ptr, ptr.ty, ConstantInt(0, i32), idx)
        elif isinstance(ptr.ty.elem_ty, PointerType):
            ptr = LoadInst(block, ptr)

            block, idx = build_ir_expr(node.idx, block, ctx)
            inst = GetElementPtrInst(
                block, ptr, ptr.ty, idx)
        else:
            raise ValueError()

        ctx.named_values[node] = inst

        return block, inst

    if isinstance(node, StringLiteralExpr):
        if node in ctx.named_values:
            return block, ctx.named_values[node]

        values = []
        for value in node.val:
            values.append(ConstantInt(ord(value), i8))
        values.append(ConstantInt(0, i8))

        value = ConstantArray(values, ctx.get_ir_type(node.type))

        value = GlobalVariable(
            value.ty, True, GlobalLinkage.Global, f".str{len(ctx.module.global_variables)}", initializer=value)

        ctx.module.add_global_variable(value)

        ctx.named_values[node] = value

        return block, value

    if isinstance(node, (TypedBinaryOp, TypedFunctionCall, CastExpr, IntegerConstantExpr)):
        alloca_insert_pt = get_alloca_insert_pt(ctx)

        block, rhs = build_ir_expr(node, block, ctx)
        mem = alloca_insert_pt = AllocaInst(
            alloca_insert_pt, ConstantInt(1, i32), rhs.ty, 0)
        StoreInst(block, rhs, mem)
        return block, mem

    print(node)
    raise Exception("Unreachable")


def build_ir_assign_op_init_list(lhs_node, rhs_lst, block, ctx):
    if isinstance(lhs_node.type, ast.types.CompositeType):
        lhs_node_ty = lhs_node.type
        if lhs_node_ty.is_union:
            lhs_node_ty, _, _ = lhs_node_ty.fields[0]

        field_idx = 0
        for field_ty, field_name, _ in lhs_node_ty.fields:
            field_val = ast.node.TypedAccessorOp(
                lhs_node, ast.node.Ident(field_name), field_ty)

            block, _ = build_ir_assign_op(
                field_val, rhs_lst[field_idx], block, ctx)

            field_idx += 1

        return block, None

    if isinstance(lhs_node.type, ast.types.ArrayType):
        elem_idx = 0
        elem_ty = lhs_node.type.elem_ty

        for i in range(lhs_node.type.size):
            elem_val = ast.node.TypedArrayIndexerOp(
                lhs_node, ast.node.IntegerConstantExpr(i, ast.types.PrimitiveType("int")), elem_ty)

            block, _ = build_ir_assign_op(
                elem_val, rhs_lst[elem_idx], block, ctx)

            elem_idx += 1

        return block, None

    raise Exception("Unreachable")


def build_ir_assign_op(lhs_node, rhs, block, ctx):
    if isinstance(rhs, list):
        return build_ir_assign_op_init_list(lhs_node, rhs, block, ctx)

    block, lhs = get_lvalue(lhs_node, block, ctx)

    if lhs.ty.elem_ty != rhs.ty:
        rhs = cast(block, rhs, lhs.ty.elem_ty,
                   is_signed_type(lhs_node.type), ctx)
    return block, StoreInst(block, rhs, lhs)


def build_ir_expr_assign_op(node, block, ctx):
    if node.op in ["+=", "-=", "*=", "/=", "<<=", ">>="]:
        block, lhs = build_ir_expr(node.lhs, block, ctx)
        block, rhs = build_ir_expr(node.rhs, block, ctx)

        if rhs.ty != lhs.ty:
            rhs = cast(block, rhs, lhs.ty, is_signed_type(node.lhs.type), ctx)

        if is_integer_ty(lhs.ty):
            prefix = ""
        else:
            prefix = "f"

        if node.op == "+=":
            op = "add"
        elif node.op == "-=":
            op = "sub"
        elif node.op == "*=":
            op = "mul"
        elif node.op == "/=":
            if is_signed_type(node.lhs.type):
                op = "sdiv"
            else:
                op = "udiv"
        elif node.op == "<<=":
            op = "shl"
        elif node.op == ">>=":
            if is_signed_type(node.lhs.type):
                op = "ashr"
            else:
                op = "lshr"

        op = prefix + op
        rhs = BinaryInst(block, op, lhs, rhs)
    else:
        assert(node.op == "=")
        block, rhs = build_ir_expr(node.rhs, block, ctx)

    return build_ir_assign_op(node.lhs, rhs, block, ctx)


def is_signed_type(ty):
    if isinstance(ty, ast.types.PrimitiveType):
        if ty.name in ['_Bool', 'unsigned char', 'unsigned short', 'unsigned int', 'unsigned long']:
            return False
        elif ty.name in ['char', 'short', 'int', 'long']:
            return True
        elif ty.name in ['float', 'double', 'long double']:
            return True
        raise NotImplementedError()
    return False


def cast(block, value, ty, signed, ctx):
    from_type = value.ty
    to_type = ty

    if from_type == to_type:
        return value

    src_size = ctx.module.data_layout.get_type_alloc_size(
        from_type)
    dst_size = ctx.module.data_layout.get_type_alloc_size(to_type)

    if from_type in [f16, f32, f64, f128]:
        if to_type in [f16, f32, f64, f128]:
            if dst_size < src_size:
                return FPTruncInst(block, value, to_type)
            elif dst_size > src_size:
                return FPExtInst(block, value, to_type)
            else:
                raise NotImplementedError()
        elif to_type in [i1, i8, i16, i32, i64]:
            if signed:
                return FPToSIInst(block, value, to_type)
            else:
                return FPToUIInst(block, value, to_type)
    elif from_type in [i1, i8, i16, i32, i64]:
        if isinstance(to_type, PointerType):
            return IntToPtrInst(block, value, to_type)
        elif to_type in [i1, i8, i16, i32, i64]:
            if src_size < dst_size:
                if signed:
                    return SExtInst(block, value, to_type)
                else:
                    return ZExtInst(block, value, to_type)
            else:
                return TruncInst(block, value, to_type)
        elif to_type in [f16, f32, f64, f128]:
            if signed:
                return SIToFPInst(block, value, to_type)
            else:
                return UIToFPInst(block, value, to_type)
        else:
            raise ValueError("Unsupporing cast")
    elif isinstance(from_type, PointerType):
        if isinstance(to_type, PointerType):
            return BitCastInst(block, value, to_type)
        elif to_type in [i1, i8, i16, i32, i64]:
            return PtrToIntInst(block, value, to_type)
        else:
            raise ValueError("Unsupporing cast")
    else:
        raise ValueError("Unsupporing cast")


def build_ir_expr_cmp_op(node, block, ctx):
    block, lhs = build_ir_expr(node.lhs, block, ctx)
    block, rhs = build_ir_expr(node.rhs, block, ctx)

    result_ty = ctx.get_ir_type(node.type)

    assert(lhs.ty == rhs.ty)

    if is_floating_type(node.lhs.type):
        if node.op == '==':
            op = 'oeq'
        elif node.op == '!=':
            op = 'one'
        elif node.op == '<':
            op = 'olt'
        elif node.op == '>':
            op = 'ogt'
        elif node.op == '<=':
            op = 'ole'
        elif node.op == '>=':
            op = 'oge'
        else:
            raise ValueError(
                "The compare node has invalid operator: " + node.op)

        return block, FCmpInst(block, op, lhs, rhs)

    elif is_integer_type(node.lhs.type):
        is_signed = is_signed_type(node.lhs.type)

        if node.op == '==':
            op = 'eq'
        elif node.op == '!=':
            op = 'ne'
        elif node.op == '<':
            op = 'slt' if is_signed else 'ult'
        elif node.op == '>':
            op = 'sgt' if is_signed else 'ugt'
        elif node.op == '<=':
            op = 'slte' if is_signed else 'ulte'
        elif node.op == '>=':
            op = 'sgte' if is_signed else 'ulte'
        else:
            raise ValueError(
                "The compare node has invalid operator: " + node.op)

        return block, CmpInst(block, op, lhs, rhs)

    raise ValueError("Invalid type to compare")


def build_ir_expr_logical_op(node, block, ctx):
    block, lhs = build_ir_expr(node.lhs, block, ctx)
    block, rhs = build_ir_expr(node.rhs, block, ctx)

    if node.op == '&&':
        op = 'and'
    elif node.op == '^^':
        op = 'xor'
    elif node.op == '||':
        op = 'or'
    else:
        raise ValueError(
            "The compare node has invalid operator: " + node.op)

    return block, BinaryInst(block, op, lhs, rhs)


def is_integer_type(ty):
    return ty.name in ["char", "unsigned char", "short", "unsigned short", "int", "unsigned int", "long", "unsigned long"]


def is_floating_type(ty):
    return ty.name in ["float", "double"]


def is_vector_of_integer_type(ty):
    return isinstance(ty, ast.types.VectorType) and is_integer_type(ty.elem_ty)


def is_vector_of_floating_type(ty):
    return isinstance(ty, ast.types.VectorType) and is_floating_type(ty.elem_ty)


def build_ir_expr_arith_op(node, block, ctx):
    block, lhs = build_ir_expr(node.lhs, block, ctx)
    block, rhs = build_ir_expr(node.rhs, block, ctx)

    if is_integer_type(node.type) or is_vector_of_integer_type(node.type):
        is_signed = is_signed_type(node.lhs.type)

        if node.op == '+':
            op = 'add'
        elif node.op == '-':
            op = 'sub'
        elif node.op == '*':
            op = 'mul'
        elif node.op == '/':
            op = 'sdiv' if is_signed else 'udiv'
        elif node.op == '>>':
            op = 'ashr' if is_signed else 'lshr'
        elif node.op == '<<':
            op = 'shl'
        elif node.op == '&':
            op = 'and'
        elif node.op == '|':
            op = 'or'
        else:
            raise ValueError(
                "The arithematic node has invalid operator: " + node.op)
    elif is_floating_type(node.type) or is_vector_of_floating_type(node.type):
        if node.op == '+':
            op = 'fadd'
        elif node.op == '-':
            op = 'fsub'
        elif node.op == '*':
            op = 'fmul'
        elif node.op == '/':
            op = 'fdiv'
        else:
            raise ValueError(
                "The arithematic node has invalid operator: " + node.op)
    else:
        raise ValueError(
            "Can't find the operator appropriating for the values")

    result_ty = ctx.get_ir_type(node.type)
    if result_ty != lhs.ty:
        if isinstance(result_ty, VectorType):
            if result_ty.elem_ty == lhs.ty:
                elem = lhs
                vec = InsertElementInst(block, get_constant_null_value(
                    result_ty), elem, ConstantInt(0, i32))

                for i in range(1, result_ty.size):
                    vec = InsertElementInst(
                        block, vec, elem, ConstantInt(i, i32))
                lhs = vec
            else:
                src_size = ctx.module.data_layout.get_type_alloc_size(lhs.ty)
                dst_size = ctx.module.data_layout.get_type_alloc_size(
                    result_ty)
                raise NotImplementedError()
        else:
            lhs = cast(block, lhs, result_ty, is_signed_type(node.type), ctx)

    if result_ty != rhs.ty:
        if isinstance(result_ty, VectorType):
            assert(result_ty.elem_ty == rhs.ty)

            elem = rhs
            vec = InsertElementInst(block, get_constant_null_value(
                result_ty), elem, ConstantInt(0, i32))

            for i in range(1, result_ty.size):
                vec = InsertElementInst(block, vec, elem, ConstantInt(i, i32))
            rhs = vec
        else:
            rhs = cast(block, rhs, result_ty, is_signed_type(node.type), ctx)

    return block, BinaryInst(block, op, lhs, rhs)


def build_ir_expr_binary_op(node, block, ctx):
    if node.op in ['=', '+=', '-=', '*=', '/=', '>>=', '<<=']:
        return build_ir_expr_assign_op(node, block, ctx)

    if node.op in ['+', '-', '*', '/', '%', '>>', '<<', '&', '|']:
        return build_ir_expr_arith_op(node, block, ctx)

    if node.op in ['==', '!=', '>', '<', '>=', '<=']:
        return build_ir_expr_cmp_op(node, block, ctx)

    if node.op in ['&&', '^^', '||']:
        return build_ir_expr_logical_op(node, block, ctx)

    raise Exception("Unreachable")


def build_ir_expr_conditional_expr(node, block, ctx):
    block, cond = build_ir_expr(node.cond_expr, block, ctx)

    if cond.ty != i1:
        cond = CmpInst(block, "ne", cond, ConstantInt(0, cond.ty))

    block_true = BasicBlock(block.func, block)
    block_false = BasicBlock(block.func, block_true)
    block_cont = BasicBlock(block.func, block_false)

    BranchInst(block, cond, block_true, block_false)

    ctx.push_branch_target(block_cont)

    block_true, true_value = build_ir_expr(node.true_expr, block_true, ctx)
    block_false, false_value = build_ir_expr(node.false_expr, block_false, ctx)

    ctx.pop_branch_target()

    JumpInst(block_true, block_cont)
    JumpInst(block_false, block_cont)

    values = [
        true_value, block_true,
        false_value, block_false,
    ]

    return block_cont, PHINode(block_cont, ctx.get_ir_type(node.type), values)


def build_ir_expr_post_op(node, block, ctx):
    if node.op in ["++", "--"]:
        block, mem = get_lvalue(node.expr, block, ctx)
        value = LoadInst(block, mem)

        one = ConstantInt(1, ctx.get_ir_type(node.expr.type))

        if node.op == "++":
            op = 'add'
        else:
            op = 'sub'

        inc_value = BinaryInst(block, op, value, one)
        StoreInst(block, inc_value, mem)

        return block, value

    raise Exception("Unreachable")


def build_ir_expr_unary_op(node, block, ctx):
    if node.op in ["++", "--"]:
        block, lhs_ptr = get_lvalue(node.expr, block, ctx)
        lhs = LoadInst(block, lhs_ptr)
        rhs = ConstantInt(1, ctx.get_ir_type(node.expr.type))

        if node.op == "++":
            op = 'add'
        elif node.op == "--":
            op = 'sub'

        val = BinaryInst(block, op, lhs, rhs)
        StoreInst(block, val, lhs_ptr)

        return block, val

    if node.op == "!":
        block, val = build_ir_expr(node.expr, block, ctx)

        op = 'xor'

        return block, BinaryInst(block, op, val, ConstantInt(1, val.ty))

    if node.op in ["-"]:
        block, rhs = build_ir_expr(node.expr, block, ctx)

        if is_integer_ty(rhs.ty):
            lhs = ConstantInt(0, rhs.ty)
            op = "sub"
        else:
            lhs = ConstantFP(0.0, rhs.ty)
            op = "fsub"

        expr_type = ctx.get_ir_type(node.expr.type)
        assert(expr_type == lhs.ty)

        return block, BinaryInst(block, op, lhs, rhs)

    if node.op == "*":
        block, val = build_ir_expr(node.expr, block, ctx)

        return block, LoadInst(block, val)

    if node.op == "&":
        block, val = get_lvalue(node.expr, block, ctx)

        return block, val

    raise Exception("Unreachable")


def build_ir_expr_ident(node, block, ctx):
    assert isinstance(node, (TypedIdentExpr,))

    if isinstance(node.type, ast.types.CompositeType):
        block, lval = get_lvalue(node, block, ctx)
        return block, LoadInst(block, lval)

    mem = ctx.named_values[node.val]
    return block, LoadInst(block, mem)


def get_func_name(name):
    from glsl.sema import FuncSignature

    if isinstance(name, FuncSignature):
        return name.name

    return name


def get_alloca_insert_pt(ctx):
    alloca_insert_pt = ctx.function.blocks[0]

    if len(ctx.function.blocks[0].insts) > 0:
        alloca_insert_pt = ctx.function.blocks[0].insts[0]

    last_inst = None
    for inst in reversed(ctx.function.blocks[0].insts):
        if isinstance(inst, AllocaInst):
            if last_inst:
                alloca_insert_pt = last_inst
                break

    return alloca_insert_pt


def build_ir_assign_op_formal_arg(lhs_node, rhs, block, ctx):
    if isinstance(lhs_node.type, ast.types.ArrayType):
        if isinstance(rhs.ty.elem_ty, PointerType):
            rhs = LoadInst(block, rhs)
            rhs = GetElementPtrInst(
                block, rhs, rhs.ty, ConstantInt(0, i32))
        else:
            rhs = GetElementPtrInst(
                block, rhs, rhs.ty, ConstantInt(0, i32), ConstantInt(0, i32))
    return build_ir_assign_op(lhs_node, rhs, block, ctx)


def build_ir_expr_func_call(node, block, ctx):
    from ir.types import PointerType

    if node.ident.name == "__va_start":
        block, va_list = build_ir_expr(node.params[0], block, ctx)
        va_list = BitCastInst(block, va_list, PointerType(i8, 0))
        va_start_func = ctx.get_va_start_func(va_list)
        return block, CallInst(block, va_start_func, [va_list])

    if node.ident.name == "__va_end":
        block, va_list = build_ir_expr(node.params[0], block, ctx)
        va_list = BitCastInst(block, va_list, PointerType(i8, 0))
        va_end_func = ctx.get_va_end_func(va_list)
        return block, CallInst(block, va_end_func, [va_list])

    if node.ident in ctx.defined_funcs:
        callee_func = ctx.defined_funcs[node.ident]

        test = node.ident.name in[
            # "add_v3v3",
            # "sub_v3v3",
            # "mul_v3v3",
            # "div_v3v3",
            # "dot_v3v3",
            # "cross_v3v3",

            # "normalize_v3",
            # "max_v3",
            # "min_v3",
            # "minus_v3",

            # "add_v3d",
            # "sub_v3d",
            # "mul_v3d",
            # "div_v3d",

            # "eval_r",
            # "intersect_sphere",
            # "intersect",
            # "_dorand48",
            # "erand48",
            # "ideal_specular_reflect",
            # "cosine_weighted_sample_on_hemisphere",
            # "schlick_reflectance",
            # "reflectance0",
            "ideal_specular_transmit",
        ]

        if "inline" in callee_func.proto.specs and callee_func.stmts:
            arg_values = []
            for arg in node.params:
                if isinstance(arg.type, ast.types.ArrayType):
                    block, arg_value = get_lvalue(arg, block, ctx)
                else:
                    block, arg_value = build_ir_expr(arg, block, ctx)
                arg_values.append(arg_value)

            ctx.begin_func()

            ctx.named_values.update(ctx.global_named_values)
            traverse_depth(callee_func, enter_func=build_ir_stack_alloc,
                           args=(block, ctx))

            alloca_insert_pt = get_alloca_insert_pt(ctx)

            for arg, formal_arg in zip(arg_values, callee_func.params):
                if isinstance(formal_arg.ty, ast.types.ArrayType):
                    formal_arg_ty = PointerType(
                        ctx.get_ir_type(formal_arg.ty.elem_ty), 0)
                else:
                    formal_arg_ty = ctx.get_ir_type(formal_arg.ty)
                mem = alloca_insert_pt = AllocaInst(
                    alloca_insert_pt, ConstantInt(1, i32), formal_arg_ty, 0)
                ctx.named_values[formal_arg] = mem

                block, _ = build_ir_assign_op_formal_arg(TypedIdentExpr(
                    formal_arg, formal_arg.ty), arg, block, ctx)

            return_ty = ctx.get_ir_type(callee_func.proto.type)
            if return_ty != void:
                return_value = alloca_insert_pt = AllocaInst(
                    alloca_insert_pt, ConstantInt(1, i32), return_ty, 0)

                ctx.return_value = return_value

            block_end = BasicBlock(block.func, block)
            if callee_func.stmts:
                ctx.push_return_target(block_end)
                for stmt in callee_func.stmts:
                    block = build_ir_stmt(stmt, block, ctx)
                ctx.push_return_target(block_end)

            ctx.end_func()

            JumpInst(block, block_end)

            if return_ty != void:
                return block_end, LoadInst(block_end, return_value)

            block_end.move(block)

            return block_end, None

    params = []
    func_info = compute_func_info(ctx, node.ident.ty)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        mem = ctx.named_values[node]
        params.append(mem)

        return_mem = mem

    for i, param in enumerate(node.params):
        param_ty = ctx.get_ir_type(param.type)

        if i < len(func_info.arguments):
            arg_info = func_info.arguments[i]
            ty = arg_info.ty
            kind = arg_info.kind
        else:
            assert(node.ident.ty.is_variadic)
            ty = param_ty
            src_size = ctx.module.data_layout.get_type_alloc_size(ty)
            dst_size = ctx.module.data_layout.get_type_alloc_size(i32)

            if src_size < dst_size:
                ty = i32
            kind = ABIArgKind.Direct

        if kind == ABIArgKind.Indirect:
            block, val = get_lvalue(param, block, ctx)
        else:
            if param_ty != ty:
                src_type = param_ty
                dst_type = ty

                src_size = ctx.module.data_layout.get_type_alloc_size(src_type)
                dst_size = ctx.module.data_layout.get_type_alloc_size(dst_type)

                if src_type in [i1, i8, i16, i32, i64] and dst_type in [i1, i8, i16, i32, i64]:
                    if src_size <= dst_size:
                        block, val = build_ir_expr(param, block, ctx)
                        val = ZExtInst(block, val, ty)
                    else:
                        raise NotImplementedError()
                elif src_type in [f16, f32, f64, f128] and dst_type in [f16, f32, f64, f128]:
                    if src_size <= dst_size:
                        block, val = build_ir_expr(param, block, ctx)
                        val = FPExtInst(block, val, dst_type)
                    else:
                        raise NotImplementedError()
                elif isinstance(src_type, PointerType) and isinstance(dst_type, PointerType):
                    block, val = get_lvalue(param, block, ctx)
                    val = BitCastInst(
                        block, val, PointerType(dst_type, 0))
                    val = LoadInst(block, val)
                elif isinstance(src_type, ArrayType) and isinstance(dst_type, PointerType):
                    block, val = get_lvalue(param, block, ctx)
                    if isinstance(val.ty.elem_ty, PointerType):
                        val = LoadInst(block, val)
                        val = GetElementPtrInst(
                            block, val, val.ty, ConstantInt(0, i32))
                    else:
                        val = GetElementPtrInst(
                            block, val, val.ty, ConstantInt(0, i32), ConstantInt(0, i32))
                else:

                    raise ValueError("Unsupporting cast.")

            else:
                block, val = build_ir_expr(param, block, ctx)

        params.append(val)

    call_inst = CallInst(block, ctx.funcs[node.ident.name], params)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        value = LoadInst(block, return_mem)
        return block, value

    return block, call_inst


def build_ir_expr_int_const(node, block, ctx):
    return block, ConstantInt(node.val, ctx.get_ir_type(node.type))


def build_ir_expr_float_const(node, block, ctx):
    return block, ConstantFP(node.val, ctx.get_ir_type(node.type))


def build_ir_expr_string_const(node, block, ctx):
    return get_lvalue(node, block, ctx)


def build_ir_expr_accessor(node, block, ctx):
    assert isinstance(node, (TypedAccessorOp,))

    block, lval = get_lvalue(node, block, ctx)
    return block, LoadInst(block, lval)


def build_ir_expr_indexer(node, block, ctx):
    assert isinstance(node, (TypedArrayIndexerOp,))

    block, lval = get_lvalue(node, block, ctx)
    return block, LoadInst(block, lval)


def build_ir_expr(node, block, ctx):
    if isinstance(node, TypedBinaryOp):
        return build_ir_expr_binary_op(node, block, ctx)

    if isinstance(node, TypedConditionalExpr):
        return build_ir_expr_conditional_expr(node, block, ctx)

    if isinstance(node, TypedPostOp):
        return build_ir_expr_post_op(node, block, ctx)

    if isinstance(node, TypedIdentExpr):
        return build_ir_expr_ident(node, block, ctx)

    if isinstance(node, IntegerConstantExpr):
        return build_ir_expr_int_const(node, block, ctx)

    if isinstance(node, FloatingConstantExpr):
        return build_ir_expr_float_const(node, block, ctx)

    if isinstance(node, StringLiteralExpr):
        return build_ir_expr_string_const(node, block, ctx)

    if isinstance(node, TypedAccessorOp):
        return build_ir_expr_accessor(node, block, ctx)

    if isinstance(node, TypedArrayIndexerOp):
        return build_ir_expr_indexer(node, block, ctx)

    if isinstance(node, TypedUnaryOp):
        return build_ir_expr_unary_op(node, block, ctx)

    if isinstance(node, TypedFunctionCall):
        return build_ir_expr_func_call(node, block, ctx)

    if isinstance(node, TypedCommaOp):
        result = None
        for expr in node.exprs:
            block, result = build_ir_expr(expr, block, ctx)
        return block, result

    if isinstance(node, TypedSizeOfExpr):
        if node.expr:
            ty = ctx.get_ir_type(node.expr.type)
        else:
            ty = ctx.get_ir_type(node.sized_type)
        size = ctx.module.data_layout.get_type_alloc_size(ty)

        return block, ConstantInt(size, ctx.get_ir_type(node.type))

    if isinstance(node, TypedInitializerList):
        exprs = []
        for _, expr in node.exprs:
            block, expr = build_ir_expr(expr, block, ctx)
            exprs.append(expr)
        return block, exprs

    if isinstance(node, CastExpr):
        result_ty = ctx.get_ir_type(node.type)

        block, value = build_ir_expr(node.expr, block, ctx)
        return block, cast(block, value, result_ty, is_signed_type(node.expr.type), ctx)

    raise Exception("Unreachable")


def build_ir_expr_stmt(node, block, ctx):
    if not node.expr:
        return block

    block, _ = build_ir_expr(node.expr, block, ctx)
    return block


def build_ir_if_stmt(node, block, ctx):
    block_then = BasicBlock(block.func, block)
    block_else = BasicBlock(block.func, block_then)
    block_cont = BasicBlock(block.func, block_else)

    block, cond = build_ir_expr(node.cond, block, ctx)

    if cond.ty != i1:
        cond = CmpInst(block, "ne", cond, ConstantInt(0, cond.ty))

    ctx.push_branch_target(block_cont)

    block_then_out = build_ir_stmt(node.then_stmt, block_then, ctx)
    block_else_out = build_ir_stmt(node.else_stmt, block_else, ctx)

    ctx.pop_branch_target()

    # BranchInst(block, cond, block_then, block_else)
    # JumpInst(block_then_out, block_cont)
    # JumpInst(block_else_out, block_cont)

    if block_then == block_then_out and len(block_then.insts) == 0:
        if block_else == block_else_out and len(block_else.insts) == 0:
            block_then.remove()
            block_else.remove()
            return block

        JumpInst(block_else_out, block_cont)
        BranchInst(block, cond, block_cont, block_else)
        block_then.remove()
    else:
        if block_else == block_else_out and len(block_else.insts) == 0:
            JumpInst(block_then_out, block_cont)
            BranchInst(block, cond, block_then, block_cont)
            block_else.remove()
        else:
            JumpInst(block_then_out, block_cont)
            JumpInst(block_else_out, block_cont)
            BranchInst(block, cond, block_then, block_else)

    return block_cont


def build_ir_while_stmt(node, block, ctx):
    # append 3 blocks
    block_cond = BasicBlock(block.func, block)
    block_then = BasicBlock(block.func, block_cond)
    block_cont = BasicBlock(block.func, block_then)

    JumpInst(block, block_cond)

    block_cond, cond = build_ir_expr(node.cond, block_cond, ctx)

    if cond.ty != i1:
        cond = CmpInst(block_cond, "ne", cond, ConstantInt(0, cond.ty))

    BranchInst(block_cond, cond, block_then, block_cont)

    ctx.push_break_target(block_cont)
    ctx.push_continue_target(block_cond)

    block_then = build_ir_stmt(node.stmt, block_then, ctx)
    JumpInst(block_then, block_cond)

    ctx.pop_break_target()
    ctx.pop_continue_target()

    return block_cont


def is_empty_stmt(node):
    if node is None:
        return True
    if isinstance(node, CompoundStmt) and len(node.stmts) > 0:
        return None

    return False


def build_ir_for_stmt(node, block, ctx):
    block = build_ir_stmt(node.init, block, ctx)

    # append 3 blocks
    block_cond = BasicBlock(block.func, block)
    block_then = BasicBlock(block.func, block_cond)
    block_cont = BasicBlock(block.func, block_then)

    JumpInst(block, block_cond)

    block_cond, cond = build_ir_expr(node.cond, block_cond, ctx)

    BranchInst(block_cond, cond, block_then, block_cont)

    ctx.push_break_target(block_cont)
    ctx.push_continue_target(block_cond)

    block_then = build_ir_stmt(node.stmt, block_then, ctx)
    block_then = build_ir_stmt(ExprStmt(node.loop), block_then, ctx)
    JumpInst(block_then, block_cond)

    ctx.pop_break_target()
    ctx.pop_continue_target()

    return block_cont


def build_ir_compound_stmt(node, block, ctx):
    if not node.stmts:
        return block

    for stmt in node.stmts:
        block = build_ir_stmt(stmt, block, ctx)

    return block


def build_ir_continue_stmt(node, block, ctx):
    JumpInst(block, ctx.get_continue_target())

    block = BasicBlock(block.func, block)
    return block


def build_ir_break_stmt(node, block, ctx):
    JumpInst(block, ctx.get_break_target())

    block = BasicBlock(block.func, block)
    return block


def build_ir_return_stmt(node, block, ctx):
    if ctx.in_inline:
        lhs = ctx.return_value
        rhs_ty = ctx.get_ir_type(node.expr.type)

        if lhs.ty.elem_ty != rhs_ty:
            if isinstance(lhs.ty.elem_ty, PointerType) and isinstance(rhs_ty, ArrayType):
                block, rhs_ptr = get_lvalue(node.expr, block, ctx)
                rhs = GetElementPtrInst(
                    block, rhs_ptr, rhs_ptr.ty, ConstantInt(0, i32), ConstantInt(0, i32))
            else:
                block, rhs = build_ir_expr(node.expr, block, ctx)
                rhs = cast(block, rhs, lhs.ty.elem_ty,
                           is_signed_type(node.expr.type), ctx)
        else:
            block, rhs = build_ir_expr(node.expr, block, ctx)

        StoreInst(block, rhs, lhs)
        JumpInst(block, ctx.get_return_target())

        block = BasicBlock(block.func, block)
        return block

    if not node.expr:
        JumpInst(block, ctx.get_return_target())
        block = BasicBlock(block.func, block)
        return block

    return_ty = ctx.get_ir_type(node.expr.type)

    func_info = ctx.func_info
    if func_info.return_info.kind == ABIArgKind.Indirect:
        block, rhs = get_lvalue(node.expr, block, ctx)
        lhs = ctx.return_value

        size, align = ctx.module.data_layout.get_type_size_in_bits(
            lhs.ty.elem_ty)
        align = 4
        rhs = BitCastInst(block, rhs, PointerType(i8, 0))
        lhs = BitCastInst(block, lhs, PointerType(i8, 0))

        memcpy = ctx.get_memcpy_func(lhs, rhs, size)
        CallInst(block, memcpy, [lhs, rhs, ConstantInt(
            int(size / 8), i32), ConstantInt(int(align / 8), i32), ConstantInt(0, i1)])

        JumpInst(block, ctx.get_return_target())
    else:
        lhs = ctx.return_value
        rhs_ty = ctx.get_ir_type(node.expr.type)

        if lhs.ty.elem_ty != rhs_ty:
            if isinstance(lhs.ty.elem_ty, PointerType) and isinstance(rhs_ty, ArrayType):
                block, rhs_ptr = get_lvalue(node.expr, block, ctx)
                rhs = GetElementPtrInst(
                    block, rhs_ptr, rhs_ptr.ty, ConstantInt(0, i32), ConstantInt(0, i32))
            else:
                block, rhs = build_ir_expr(node.expr, block, ctx)
                rhs = cast(block, rhs, lhs.ty.elem_ty,
                           is_signed_type(node.expr.type), ctx)
        else:
            block, rhs = build_ir_expr(node.expr, block, ctx)

        StoreInst(block, rhs, lhs)
        JumpInst(block, ctx.get_return_target())

    block = BasicBlock(block.func, block)
    return block


def build_ir_switch_stmt(node, block, ctx):
    assert(node.stmts)
    stmts = node.stmts.stmts

    blocks = {}
    cur_block = block
    for stmt in stmts:
        assert(isinstance(stmt, CaseLabelStmt))

        cur_block = BasicBlock(block.func, cur_block)
        blocks[stmt] = cur_block

    cont_block = BasicBlock(block.func, cur_block)

    block, cond = build_ir_expr(node.cond, block, ctx)
    assert(cond.ty in [i32])

    cases = []
    default_block = None
    for case_label, case_block in blocks.items():
        if not case_label.expr:
            default_block = case_block
            continue

        cases.append(evaluate_constant_expr(ctx, case_label.expr))
        cases.append(case_block)

    SwitchInst(block, cond, default_block, cases)

    for case_label in stmts:
        assert(isinstance(case_label, CaseLabelStmt))

        ctx.push_break_target(cont_block)

        blocks[case_label] = build_ir_stmt(
            case_label.stmt, blocks[case_label], ctx)

        ctx.pop_break_target()

    for i, case_label in enumerate(stmts):
        assert(isinstance(case_label, CaseLabelStmt))

        current_block = blocks[case_label]
        if i + 1 < len(stmts):
            next_block = blocks[stmts[i + 1]]
        else:
            next_block = cont_block

        JumpInst(current_block, next_block)

    return cont_block


def build_ir_asm_stmt(node, block, ctx):
    output_operands = node.operands[0]
    input_operands = node.operands[1]
    cobbers = node.operands[2]

    operands = []
    for constraint, name in input_operands:
        var = ctx.named_values[name]
        var = LoadInst(block, var)

        operands.append(var)

    result_ty_elems = []
    for constraint, name in output_operands:
        var = ctx.named_values[name]

        assert(isinstance(var.ty, PointerType))
        result_ty_elems.append(var.ty.elem_ty)

    result_ty = StructType("", result_ty_elems)
    func_ty = FunctionType("", result_ty, [operand.ty for operand in operands])

    def parse_constraint(constraint):
        m = re.match("([=+&%]?)(.*)", constraint)

        if not m:
            raise ValueError("Invalid constraint")

        modifier, constraint = m.groups()

        if constraint == "D":
            constraint = "{di}"
        elif constraint == "S":
            constraint = "{si}"
        elif constraint == "a":
            constraint = "{ax}"
        elif constraint == "b":
            constraint = "{bx}"
        elif constraint == "c":
            constraint = "{cx}"
        elif constraint == "d":
            constraint = "{dx}"
        elif constraint == "0":
            constraint = "0"
        elif constraint == "1":
            constraint = "1"
        else:
            raise ValueError("Invalid constraint")

        return modifier, constraint

    asm_string = node.template
    constraints = []
    for constraint, name in output_operands:
        constraint = constraint.value

        modifier, constraint = parse_constraint(constraint)
        constraints.append(f"{modifier}{constraint}")

    for constraint, name in input_operands:
        constraint = constraint.value

        modifier, constraint = parse_constraint(constraint)
        constraints.append(f"{modifier}{constraint}")

    for constraint, name in cobbers:
        constraint = constraint.value

        if constraint == "memory":
            constraints.append("~{memory}")
        elif constraint == "cc":
            constraints.append("~{cc}")
        else:
            raise ValueError("Invalid constraint")

    asm = InlineAsm(func_ty, asm_string, ",".join(constraints), True)

    asm_result = CallInst(block, asm, operands)

    for i, (constraint, name) in enumerate(output_operands):
        var = ctx.named_values[name]
        value = ExtractValueInst(block, asm_result, i)
        StoreInst(block, value, var)

    return block


def build_ir_stmt(node, block, ctx):
    # Allocate local variables.
    if isinstance(node, TypedVariable):
        for variable, _ in node.idents:
            size = ConstantInt(1, i32)
            ty = ctx.get_ir_type(variable.type)

            if "static" in node.storage_class:
                if variable.val not in ctx.named_values:
                    func_name = ctx.function.name
                    var_name = f"{func_name}.{variable.val.name}"

                    init = get_constant_null_value(ty)
                    # init = evaluate_constant_expr(ctx, node.init)

                    gv = GlobalVariable(
                        ty, False, GlobalLinkage.Internal, var_name, initializer=init)
                    ctx.module.add_global_variable(gv)
                    ctx.named_values[variable.val] = gv

                continue

            alloca_insert_pt = get_alloca_insert_pt(ctx)

            mem = alloca_insert_pt = AllocaInst(alloca_insert_pt, size, ty, 0)

            ctx.named_values[variable.val] = mem

    if isinstance(node, IfStmt):
        return build_ir_if_stmt(node, block, ctx)

    if isinstance(node, WhileStmt):
        return build_ir_while_stmt(node, block, ctx)

    if isinstance(node, ForStmt):
        return build_ir_for_stmt(node, block, ctx)

    if isinstance(node, CompoundStmt):
        return build_ir_compound_stmt(node, block, ctx)

    if isinstance(node, ExprStmt):
        return build_ir_expr_stmt(node, block, ctx)

    if isinstance(node, ContinueStmt):
        return build_ir_continue_stmt(node, block, ctx)

    if isinstance(node, BreakStmt):
        return build_ir_break_stmt(node, block, ctx)

    if isinstance(node, ReturnStmt):
        return build_ir_return_stmt(node, block, ctx)

    if isinstance(node, TypedVariable):
        for variable, initializer in node.idents:
            if initializer is not None:
                block, init_expr = build_ir_expr(initializer, block, ctx)
                block, _ = build_ir_assign_op(variable, init_expr, block, ctx)

        return block

    if isinstance(node, SwitchStmt):
        return build_ir_switch_stmt(node, block, ctx)

    if isinstance(node, AsmStmt):
        return build_ir_asm_stmt(node, block, ctx)

    return block


def build_ir_stack_alloc(node, block, ctx):
    alloca_insert_pt = get_alloca_insert_pt(ctx)

    # Allocate params of all callee functions.
    if isinstance(node, TypedFunctionCall):
        for i, (param_type, _, _) in enumerate(node.ident.ty.params):
            if ctx.get_ir_type(param_type) == void:
                continue

            mem = alloca_insert_pt = AllocaInst(alloca_insert_pt, ConstantInt(1, i32),
                                                ctx.get_ir_type(param_type), 0)
            ctx.named_values[(node, i)] = mem

    # Allocate return value of all callee functions.
    if isinstance(node, TypedFunctionCall):
        return_ty = ctx.get_ir_type(node.ident.ty.return_ty)
        func_info = compute_func_info(ctx, node.ident.ty)
        if func_info.return_info.kind == ABIArgKind.Indirect:
            mem = alloca_insert_pt = AllocaInst(alloca_insert_pt,
                                                ConstantInt(1, i32), return_ty, 0)
            ctx.named_values[node] = mem


def build_ir_func_header(node, func, ctx):
    func_info = compute_func_info(ctx, node.ident.ty)

    return_ty = ctx.get_ir_type(node.type)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        ir_arg = Argument(PointerType(return_ty, 0))
        ir_arg.add_attribute(Attribute(AttributeKind.StructRet))
        func.add_arg(ir_arg)

    for (arg_ty, arg_quals, arg_name), arg_info in zip(node.params, func_info.arguments):
        if arg_info.kind == ABIArgKind.Ignore:
            continue

        param_ty = arg_info.ty
        if arg_info.kind == ABIArgKind.Indirect:
            ir_arg = ir_arg_ptr = Argument(PointerType(param_ty, 0), arg_name)
        else:
            ir_arg = Argument(param_ty, arg_name)
        func.add_arg(ir_arg)


def build_ir_func(node, block, ctx):
    func_info = compute_func_info(ctx, node.proto.ident.ty)
    ctx.func_info = func_info
    alloca_insert_pt = get_alloca_insert_pt(ctx)

    return_ty = ctx.get_ir_type(node.proto.type)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        ir_arg = Argument(PointerType(return_ty, 0))
        ir_arg.add_attribute(Attribute(AttributeKind.StructRet))
        ctx.return_value = ir_arg
        block.func.add_arg(ir_arg)

    for arg, arg_info in zip(node.params, func_info.arguments):
        if isinstance(arg.ty, ast.types.ArrayType):
            param_ty = PointerType(ctx.get_ir_type(arg.ty.elem_ty), 0)
        else:
            param_ty = ctx.get_ir_type(arg.ty)

        coerced_param_ty = arg_info.ty

        if arg_info.kind == ABIArgKind.Ignore:
            continue

        if arg_info.kind == ABIArgKind.Indirect:
            ir_arg = ir_arg_ptr = Argument(
                PointerType(param_ty, 0), arg.name)
        else:
            ir_arg = Argument(coerced_param_ty, arg.name)
            ir_arg_ptr = alloca_insert_pt = AllocaInst(
                alloca_insert_pt, ConstantInt(1, i32), param_ty, 0)

            ir_arg_mem_ptr = ir_arg_ptr

            if coerced_param_ty != param_ty:
                ir_arg_mem_ptr = BitCastInst(
                    block, ir_arg_mem_ptr, PointerType(coerced_param_ty, 0))
            StoreInst(block, ir_arg, ir_arg_mem_ptr)

        ctx.named_values[arg] = ir_arg_ptr
        block.func.add_arg(ir_arg)

    block_end = BasicBlock(block.func)

    if func_info.return_info.kind in [ABIArgKind.Indirect, ABIArgKind.Ignore]:
        pass
    else:
        ctx.return_value = alloca_insert_pt = AllocaInst(
            alloca_insert_pt, ConstantInt(1, i32), return_ty, 0)

    traverse_depth(node, enter_func=build_ir_stack_alloc, args=(block, ctx))

    ctx.push_return_target(block_end)
    for stmt in node.stmts:
        block = build_ir_stmt(stmt, block, ctx)
        if len(ctx.branch_target) == 0 and isinstance(stmt, ReturnStmt):
            break
    ctx.pop_return_target()

    if func_info.return_info.kind in [ABIArgKind.Indirect, ABIArgKind.Ignore]:
        ReturnInst(block_end, None)
    else:
        if func_info.return_info.kind == ABIArgKind.Direct:
            if func_info.return_info.ty == return_ty:
                return_value = LoadInst(block_end, ctx.return_value)
            else:
                src_size = ctx.module.data_layout.get_type_alloc_size(
                    return_ty)
                dst_size = ctx.module.data_layout.get_type_alloc_size(
                    func_info.return_info.ty)
                if src_size >= dst_size:
                    coerced_value = BitCastInst(
                        block_end, ctx.return_value, PointerType(func_info.return_info.ty, 0))
                    return_value = LoadInst(block_end, coerced_value)
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
        assert(return_value.ty == func_info.return_info.ty)
        ReturnInst(block_end, return_value)

    block_end.move(block.func.blocks[-1])

    JumpInst(block, block_end)
    return block_end


class FunctionInfo:
    def __init__(self):
        pass


class ABIInfo:
    def __init__(self):
        pass


class ABIArgKind(Enum):
    Direct = auto()
    Extend = auto()
    Indirect = auto()
    Ignore = auto()
    Expand = auto()
    CoerceAndExpand = auto()
    InAlloca = auto()


class ABIArgInfo:
    def __init__(self, kind: ABIArgKind, ty: Type = None):
        self.kind = kind
        self.ty = ty

    @property
    def is_direct(self):
        return self.kind == ABIArgKind.Direct

    @property
    def is_indirect(self):
        return self.kind == ABIArgKind.Indirect

    @property
    def is_extend(self):
        return self.kind == ABIArgKind.Extend

    @property
    def is_coerce_and_expand(self):
        return self.kind == ABIArgKind.CoerceAndExpand

    @property
    def can_have_coerce_to_type(self):
        return self.is_direct or self.is_extend or self.is_coerce_and_expand


class TypeInfo:
    def __init__(self, width: int, align: int):
        self.witdh = width
        self.align = align


class CompositeTypeLayout:
    def __init__(self, size, alignment):
        self.size = size
        self.alignment = alignment


def align_to(value, align):
    return int(int((value + align - 1) / align) * align)


def compute_ast_composite_type_layout(ctx, ty):
    size = 0
    alignment = 0

    for field in ty.fields:
        field_ty, name, qual = field
        type_info = get_type_info(ctx, field_ty)
        field_size = type_info.witdh
        field_align = type_info.align

        alignment = max([alignment, field_align])

        size = align_to(size, field_align)
        size += field_size

    return CompositeTypeLayout(size, alignment)


def next_power_of_2(val):
    val |= (val >> 1)
    val |= (val >> 2)
    val |= (val >> 4)
    val |= (val >> 8)
    val |= (val >> 16)
    val |= (val >> 32)
    return val + 1


def is_power_of_2(val):
    return val & (val - 1) == 0


def get_type_info(ctx, ty):
    if isinstance(ty, ast.types.PrimitiveType):
        if ty.name in ["char", "unsigned char"]:
            return TypeInfo(8, 8)
        elif ty.name in ["short", "unsigned short"]:
            return TypeInfo(16, 16)
        elif ty.name in ["int", "unsigned int"]:
            return TypeInfo(32, 32)
        elif ty.name in ["long", "unsigned long"]:
            return TypeInfo(64, 64)
        elif ty.name == "float":
            return TypeInfo(32, 32)
        elif ty.name == "double":
            return TypeInfo(64, 64)

    if isinstance(ty, ast.types.VectorType):
        size = get_type_info(ctx, ty.elem_ty).witdh * ty.size
        align = size
        if not is_power_of_2(align):
            align = next_power_of_2(align)
            size = align_to(size, align)

        return TypeInfo(size, align)

    if isinstance(ty, ast.types.CompositeType):
        layout = compute_ast_composite_type_layout(ctx, ty)
        return TypeInfo(layout.size, layout.alignment)

    if isinstance(ty, ast.types.PointerType):
        return TypeInfo(64, 64)

    if isinstance(ty, ast.types.ArrayType):
        return get_type_info(ctx, ty.elem_ty)

    raise NotImplementedError()


def is_complex_type(ty):
    return isinstance(ty, ast.types.CompositeType)


def is_x86_mmx_type(ty):
    if isinstance(ty, VectorType):
        return get_primitive_size(ty) == 64 and is_integer_ty(ty.elem_ty)

    return False


class ABIClass(Enum):
    Integer = auto()
    SSE = auto()
    SSEUp = auto()
    X87 = auto()
    X87Up = auto()
    ComplexX87 = auto()
    NoClass = auto()
    Memory = auto()


def is_float_ty(ty):
    return ty == f32


class X86_64ABIInfo(ABIInfo):
    def __init__(self):
        super().__init__()

    def classify(self, ctx, ty):
        lohi = [ABIClass.NoClass, ABIClass.NoClass]
        idx = 0

        if isinstance(ty, ast.types.VoidType):
            lo = ABIClass.NoClass
        elif isinstance(ty, ast.types.PrimitiveType):
            if ty.name in ["bool", "uint", "int"]:
                lohi[idx] = ABIClass.Integer
            elif ty.name in ["float", "double"]:
                lohi[idx] = ABIClass.Integer
            else:
                raise ValueError("Invalid data type")
        elif isinstance(ty, ast.types.VectorType):
            size = get_type_info(ctx, ty).witdh

            if size == 128:
                lohi[0] = ABIClass.SSE
                lohi[1] = ABIClass.SSEUp
            else:
                raise ValueError("Invalid data size")
        else:
            raise ValueError("Invalid data type")

        return tuple(lohi)

    def contains_float_at_offset(self, ctx, ty, offset):
        if offset == 0 and is_float_ty(ty):
            return True

        if isinstance(ty, StructType):
            data_layout = ctx.module.data_layout
            elem_idx = data_layout.get_element_containing_offset(ty, offset)
            offset -= data_layout.get_elem_offset(ty, elem_idx)
            return self.contains_float_at_offset(ctx, ty, offset)

        if isinstance(ty, ArrayType):
            data_layout = ctx.module.data_layout
            elem_ty = ty.elem_ty
            elem_size = data_layout.get_type_alloc_size(elem_ty)
            offset -= int(offset / elem_size) * elem_size
            return self.contains_float_at_offset(ctx, elem_ty, offset)

        return False

    def get_sse_type(self, ctx, ty, offset):
        if self.contains_float_at_offset(ctx, ty, offset) and self.contains_float_at_offset(ctx, ty, offset + 4):
            return VectorType("", PrimitiveType("float"), 2)

        return PrimitiveType("double")

    def compute_arg_info(self, ctx, ty):
        if isinstance(ty, ast.types.VoidType):
            return ABIArgInfo(ABIArgKind.Ignore)

        if isinstance(ty, ast.types.PrimitiveType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.VectorType):
            ir_ty = ctx.get_ir_type(ty)
            if is_x86_mmx_type(ir_ty):
                return ABIArgInfo(ABIArgKind.Direct, get_integer_type(64))

            return ABIArgInfo(ABIArgKind.Direct)

        type_info = get_type_info(ctx, ty)
        width = type_info.witdh

        if is_complex_type(ty):
            if width > 64:
                return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))
            return ABIArgInfo(ABIArgKind.Direct, get_integer_type(width))

        raise NotImplementedError()


class WinX86_64ABIInfo(ABIInfo):
    def __init__(self):
        super().__init__()

    def compute_arg_info(self, ctx, ty: FunctionType):
        if isinstance(ty, ast.types.VoidType):
            return ABIArgInfo(ABIArgKind.Ignore)

        if isinstance(ty, ast.types.PrimitiveType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.PointerType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.ArrayType):
            return ABIArgInfo(ABIArgKind.Direct, PointerType(ctx.get_ir_type(ty.elem_ty), 0))

        if isinstance(ty, ast.types.VectorType):
            ir_ty = ctx.get_ir_type(ty)
            if is_x86_mmx_type(ir_ty):
                return ABIArgInfo(ABIArgKind.Direct, get_integer_type(64))

            return ABIArgInfo(ABIArgKind.Direct)

        type_info = get_type_info(ctx, ty)
        width = type_info.witdh

        if is_complex_type(ty):
            if width > 64:
                return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))
            return ABIArgInfo(ABIArgKind.Direct, get_integer_type(width))

        raise NotImplementedError()

    def compute_return_info(self, ctx, ty):
        return self.compute_arg_info(ctx, ty)


class EABIABIInfo(ABIInfo):
    def __init__(self):
        super().__init__()

    def compute_arg_info(self, ctx, ty: FunctionType):
        if isinstance(ty, ast.types.VoidType):
            return ABIArgInfo(ABIArgKind.Ignore)

        if isinstance(ty, ast.types.PrimitiveType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.VectorType):
            ir_ty = ctx.get_ir_type(ty)
            if is_x86_mmx_type(ir_ty):
                return ABIArgInfo(ABIArgKind.Direct, get_integer_type(64))

            return ABIArgInfo(ABIArgKind.Direct)

        type_info = get_type_info(ctx, ty)
        width = type_info.witdh

        if is_complex_type(ty):
            if width > 128:
                return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))
            return ABIArgInfo(ABIArgKind.Direct, get_integer_type(width))

        raise NotImplementedError()

    def compute_return_info(self, ctx, ty):
        return self.compute_arg_info(ctx, ty)


class RISCVABIInfo(ABIInfo):
    def __init__(self, xlen):
        super().__init__()

        self.xlen = xlen

    def classify_arg_type(self, ctx, ty):
        if isinstance(ty, ast.types.VoidType):
            return ABIArgInfo(ABIArgKind.Ignore)

        if isinstance(ty, ast.types.PrimitiveType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.VectorType):
            type_info = get_type_info(ctx, ty)
            width = type_info.witdh

            ir_ty = ctx.get_ir_type(ty)
            if width <= self.xlen * 2:
                if width <= self.xlen:
                    return ABIArgInfo(ABIArgKind.Direct, get_integer_type(self.xlen))
                if width <= self.xlen * 2:
                    return ABIArgInfo(ABIArgKind.Direct, get_array_type(get_integer_type(self.xlen), 2))

            return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))

        type_info = get_type_info(ctx, ty)
        width = type_info.witdh

        if is_complex_type(ty):
            ir_ty = ctx.get_ir_type(ty)
            if width <= self.xlen * 2:
                if width <= self.xlen:
                    return ABIArgInfo(ABIArgKind.Direct, get_integer_type(self.xlen))
                if width <= self.xlen * 2:
                    return ABIArgInfo(ABIArgKind.Direct, get_array_type(get_integer_type(self.xlen), 2))

            return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))

        raise NotImplementedError()

    def compute_arg_info(self, ctx, ty):
        return self.classify_arg_type(ctx, ty)

    def classify_return_type(self, ctx, ty):
        if isinstance(ty, ast.types.VoidType):
            return ABIArgInfo(ABIArgKind.Ignore)

        if isinstance(ty, ast.types.PrimitiveType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.VectorType):
            type_info = get_type_info(ctx, ty)
            width = type_info.witdh

            ir_ty = ctx.get_ir_type(ty)
            if width <= self.xlen * 2:
                if width <= self.xlen:
                    return ABIArgInfo(ABIArgKind.Direct, get_integer_type(self.xlen))
                if width <= self.xlen * 2:
                    return ABIArgInfo(ABIArgKind.Direct, get_array_type(get_integer_type(self.xlen), 2))

            return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))

        type_info = get_type_info(ctx, ty)
        width = type_info.witdh

        if is_complex_type(ty):
            if width <= self.xlen * 2:
                if width <= self.xlen:
                    return ABIArgInfo(ABIArgKind.Direct, get_integer_type(self.xlen))
                if width <= self.xlen * 2:
                    return ABIArgInfo(ABIArgKind.Direct, get_array_type(get_integer_type(self.xlen), 2))

            return ABIArgInfo(ABIArgKind.Indirect, get_integer_type(width))

        raise NotImplementedError()

    def compute_return_info(self, ctx, ty):
        return self.classify_return_type(ctx, ty)


def compute_arg_result_info(ctx, func_ty):
    return_ty = func_ty.return_ty

    func_info = FunctionInfo()

    func_info.return_info = ctx.abi_info.compute_return_info(ctx, return_ty)

    return_info = func_info.return_info
    if return_info.can_have_coerce_to_type and return_info.ty == None:
        return_info.ty = ctx.get_ir_type(return_ty)
        assert(return_info.ty is not None)

    return func_info


def compute_func_info(ctx, func_ty):
    func_info = compute_arg_result_info(ctx, func_ty)

    func_info.arguments = []
    for (param_ty, _, _) in func_ty.params:
        arg_info = ctx.abi_info.compute_arg_info(ctx, param_ty)

        if arg_info.can_have_coerce_to_type and arg_info.ty == None:
            arg_info.ty = ctx.get_ir_type(param_ty)
            assert(arg_info.ty is not None)
        func_info.arguments.append(arg_info)

    return func_info


class ABIType(Enum):
    WinX86_64 = auto()
    X86_64 = auto()
    EABI = auto()
    RISCV32 = auto()
    RISCV64 = auto()


def emit_ir(ast, abi, module):
    ctx = Context(module)
    if abi == ABIType.WinX86_64:
        ctx.abi_info = WinX86_64ABIInfo()
    elif abi == ABIType.X86_64:
        ctx.abi_info = X86_64ABIInfo()
    elif abi == ABIType.EABI:
        ctx.abi_info = EABIABIInfo()
    elif abi == ABIType.RISCV32:
        ctx.abi_info = RISCVABIInfo(32)
    elif abi == ABIType.RISCV64:
        ctx.abi_info = RISCVABIInfo(64)
    else:
        raise ValueError("Invalid abi type.")

    funcs = {}

    for decl in ast:
        if isinstance(decl, TypedFunction):
            funcs[decl.proto.ident] = decl

    ctx.defined_funcs = funcs

    global_named_values = {}
    ctx.global_named_values = global_named_values
    for decl in ast:
        if isinstance(decl, TypedVariable):
            for ident, init_expr in decl.idents:
                ty = ctx.get_ir_type(ident.type)
                if not init_expr:
                    init = get_constant_null_value(ty)
                else:
                    init = evaluate_constant_expr(ctx, init_expr)

                linkage = GlobalLinkage.Global
                thread_local = ThreadLocalMode.NotThreadLocal

                if "extern" in decl.storage_class:
                    linkage = GlobalLinkage.External
                    init = None

                global_named_values[ident.val] = module.add_global_variable(
                    GlobalVariable(ty, False, linkage, ident.val.name, thread_local, init))

        if isinstance(decl, TypedFunctionProto):
            func_ty = ctx.get_ir_type(decl.ident.ty)
            func = module.add_func(
                Function(module, func_ty, str(decl.ident.name)))
            ctx.funcs[decl.ident.name] = func

            build_ir_func_header(decl, func, ctx)

        if isinstance(decl, TypedFunction):
            func_ty = ctx.get_ir_type(decl.proto.ident.ty)
            func_name = str(decl.proto.ident.name)

            func = module.add_func(
                Function(module, func_ty, func_name))

            if "inline" in decl.proto.specs:
                comdat = module.add_comdat(func_name, ComdatKind.Any)
                func.comdat = comdat

            if decl.stmts:
                ctx.begin_func()
                ctx.ast_func = decl
                ctx.function = func
                ctx.funcs[func.name] = func
                ctx.current_block = BasicBlock(func)

                ctx.named_values.update(global_named_values)
                block = ctx.current_block
                build_ir_func(decl, block, ctx)

                ctx.end_func()
            else:
                if decl.proto.ident.name not in ctx.funcs:
                    build_ir_func_header(decl.proto, func, ctx)

                ctx.funcs[decl.proto.ident.name] = func

    return module
