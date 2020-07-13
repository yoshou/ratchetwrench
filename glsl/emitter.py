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

    raise NotImplementedError()


class Context:
    def __init__(self, module):
        self.module = module

        self.continue_target = []
        self.break_target = []
        self.return_target = []
        self.branch_target = []
        self.preds = []
        self.named_values = {}

        self.funcs = {}

    def begin_func(self, func):
        self.function = func

        self.continue_target = []
        self.break_target = []
        self.return_target = []
        self.branch_target = []
        self.preds = []
        self.named_values = {}

        self.funcs[func.name] = func

        self.current_block = BasicBlock(self.function)

    def end_func(self):
        pass

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
        if isinstance(ast_ty, ast.types.PrimitiveType):
            if ast_ty.name in ["bool"]:
                return i1
            if ast_ty.name in ["ushort", "short"]:
                return i16
            if ast_ty.name in ["uint", "int"]:
                return i32
            if ast_ty.name in ["float"]:
                return f32
            if ast_ty.name in ["double"]:
                return f64
        if isinstance(ast_ty, ast.types.CompositeType):
            if self.module.contains_struct_type(ast_ty.name):
                return self.module.structs[ast_ty.name]

            fields = [self.get_ir_type(ty) for ty, name, arr in ast_ty.fields]
            ty = StructType(ast_ty.name, fields)
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

            return FunctionType(return_ty, params_ty)

        if isinstance(ast_ty, ast.types.VectorType):
            elem_ty = self.get_ir_type(ast_ty.elem_ty)
            return VectorType(ast_ty.name, elem_ty, ast_ty.size)

        if isinstance(ast_ty, ast.types.ArrayType):
            elem_ty = self.get_ir_type(ast_ty.elem_ty)
            return ArrayType(elem_ty, ast_ty.size)

        raise Exception

    def get_memcpy_func(self, rd, rs, size):
        arg_tys = [rd.ty, rs.ty, i32, i32, i1]
        return self.module.get_or_declare_intrinsic_func("llvm.memcpy", arg_tys)


def get_lvalue(node, block, ctx):
    if isinstance(node, TypedIdentExpr):
        return ctx.named_values[node.val]

    if isinstance(node, TypedAccessorOp):
        ptr = get_lvalue(node.obj, block, ctx)
        ty = ctx.get_ir_type(node.obj.type)

        if isinstance(ty, VectorType):
            if node.field.val == "x":
                idx = 0
            elif node.field.val == "xy":
                idx = 0
            elif node.field.val == "y":
                idx = 1
            elif node.field.val == "z":
                idx = 2
            elif node.field.val == "w":
                idx = 3
            else:
                raise ValueError()
            inst = GetElementPtrInst(
                block, ptr, ptr.ty, ConstantInt(0, i32), ConstantInt(idx, i32))
        elif isinstance(ty, CompositeType):
            idx = node.obj.type.get_field_idx(node.field.val)
            inst = GetElementPtrInst(
                block, ptr, ptr.ty, ConstantInt(0, i32), ConstantInt(idx, i32))
        else:
            raise ValueError()

        ctx.named_values[node] = inst

        return inst

    if isinstance(node, TypedArrayIndexerOp):
        if node in ctx.named_values:
            return ctx.named_values[node]

        ptr = get_lvalue(node.arr, block, ctx)
        ty = ctx.get_ir_type(node.arr.type)

        if isinstance(ty, ArrayType):
            block, idx = build_ir_expr(node.idx, block, ctx)
            inst = GetElementPtrInst(
                block, ptr, ptr.ty, ConstantInt(0, i32), idx)
        else:
            raise ValueError()

        ctx.named_values[node] = inst

        return inst

    if isinstance(node, (TypedBinaryOp, TypedFunctionCall)):
        block, rhs = build_ir_expr(node, block, ctx)
        mem = AllocaInst(
            ctx.function.bbs[0].insts[0], ConstantInt(1, i32), rhs.ty, 0)
        StoreInst(block, rhs, mem)
        return mem

    print(node)
    raise Exception("Unreachable")


def build_ir_assign_op(lhs_node, rhs, block, ctx):
    lhs = get_lvalue(lhs_node, block, ctx)

    if lhs.ty.elem_ty != rhs.ty:
        src_size = ctx.module.data_layout.get_type_alloc_size(rhs.ty)
        dst_size = ctx.module.data_layout.get_type_alloc_size(
            lhs.ty.elem_ty)
        if src_size <= dst_size:
            lhs = BitCastInst(block, lhs, PointerType(rhs.ty, 0))
        else:
            raise NotImplementedError()
    return block, StoreInst(block, rhs, lhs)


def build_ir_expr_assign_op(node, block, ctx):
    if node.op in ["+=", "-=", "*=", "/="]:
        block, lhs = build_ir_expr(node.lhs, block, ctx)
        block, rhs = build_ir_expr(node.rhs, block, ctx)

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
            op = "div"

        op = prefix + op
        rhs = BinaryInst(block, op, lhs, rhs)
    else:
        assert(node.op == "=")
        block, rhs = build_ir_expr(node.rhs, block, ctx)

    return build_ir_assign_op(node.lhs, rhs, block, ctx)

    # node_ty = ctx.get_ir_type(node.type)
    # if require_struct_returning(node_ty, ctx):
    #     rhs = get_lvalue(node.rhs, block, ctx)
    #     lhs = get_lvalue(node.lhs, block, ctx)

    #     size, align = ctx.module.data_layout.get_type_size_in_bits(
    #         lhs.ty.elem_ty)
    #     rhs = BitCastInst(block, rhs, PointerType(i8))
    #     lhs = BitCastInst(block, lhs, PointerType(i8))

    #     memcpy = ctx.get_memcpy_func(lhs, rhs, size)
    #     return CallInst(block, memcpy, [lhs, rhs, ConstantInt(int(size / 8), i32), ConstantInt(int(align / 8), i32), ConstantInt(0, i1)])
    # else:
    #     rhs = build_ir_expr(node.rhs, block, ctx)
    #     lhs = get_lvalue(node.lhs, block, ctx)

    #     if lhs.ty.elem_ty != rhs.ty:
    #         src_size = ctx.module.data_layout.get_type_alloc_size(rhs.ty)
    #         dst_size = ctx.module.data_layout.get_type_alloc_size(
    #             lhs.ty.elem_ty)
    #         if src_size <= dst_size:
    #             lhs = BitCastInst(block, lhs, PointerType(rhs.ty))
    #         else:
    #             raise NotImplementedError()
    #     return StoreInst(block, rhs, lhs)


def is_signed_type(ty):
    if isinstance(ty, ast.types.PrimitiveType):
        if ty.name == 'uint':
            return False

        elif ty.name == 'int':
            return True

        raise NotImplementedError()
    return False


def build_ir_expr_cmp_op(node, block, ctx):
    block, lhs = build_ir_expr(node.lhs, block, ctx)
    block, rhs = build_ir_expr(node.rhs, block, ctx)

    if is_floating_type(node.lhs.type):
        if node.op == '==':
            op = 'eq'
        elif node.op == '!=':
            op = 'ne'
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
    return ty.name in ["int", "uint"]


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
            op = 'div'
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
            raise NotImplementedError()

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
            raise NotImplementedError()

    return block, BinaryInst(block, op, lhs, rhs)


def build_ir_expr_binary_op(node, block, ctx):
    if node.op in ['=', '+=', '-=', '*=', '/=']:
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
        mem = get_lvalue(node.expr, block, ctx)
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
        block, lhs = build_ir_expr(node.expr, block, ctx)
        rhs = ConstantInt(1, ctx.get_ir_type(node.expr.type))

        if node.op == "++":
            op = 'add'
        elif node.op == "--":
            op = 'sub'

        return block, BinaryInst(block, op, lhs, rhs)

    if node.op == "!":
        block, val = build_ir_expr(node.expr, block, ctx)

        op = 'xor'

        return block, BinaryInst(block, op, val, ConstantInt(1, i32))

    if node.op in ["-"]:
        expr_type = ctx.get_ir_type(node.expr.type)
        if is_integer_ty(expr_type):
            lhs = ConstantInt(0, expr_type)
            op = "sub"
        else:
            lhs = ConstantFP(0.0, expr_type)
            op = "fsub"

        block, rhs = build_ir_expr(node.expr, block, ctx)

        return block, BinaryInst(block, op, lhs, rhs)

    raise Exception("Unreachable")


def build_ir_expr_ident(node, block, ctx):
    assert isinstance(node, (TypedIdentExpr,))

    if isinstance(node.type, ast.types.CompositeType):
        lval = get_lvalue(node, block, ctx)
        return block, LoadInst(block, lval)

    mem = ctx.named_values[node.val]
    return block, LoadInst(block, mem)


def get_func_name(name):
    from glsl.sema import FuncSignature

    if isinstance(name, FuncSignature):
        return name.name

    return name


def build_ir_expr_func_call(node, block, ctx):
    if node.ident.name == "memoryBarrier":
        return block, FenceInst(block, AtomicOrdering.AcquireRelease, SyncScope.SingleThread)

    # if ctx.funcs[node.ident.name].name == "vec3_1":
    #     result_ty = ctx.get_ir_type(node.type)

    #     block, elem = build_ir_expr(node.params[0], block, ctx)
    #     vec = InsertElementInst(block, get_constant_null_value(
    #         result_ty), elem, ConstantInt(0, i32))

    #     for i in range(1, 3):
    #         vec = InsertElementInst(block, vec, elem, ConstantInt(i, i32))

    #     return block, vec

    # if ctx.funcs[node.ident.name].name == "vec3_3":
    #     result_ty = ctx.get_ir_type(node.type)

    #     block, elem = build_ir_expr(node.params[0], block, ctx)
    #     vec = InsertElementInst(block, get_constant_null_value(
    #         result_ty), elem, ConstantInt(0, i32))

    #     for i in range(1, 3):
    #         block, elem = build_ir_expr(node.params[i], block, ctx)
    #         vec = InsertElementInst(block, vec, elem, ConstantInt(i, i32))

    #     return block, vec

    params = []

    func_info = compute_func_info(ctx, node.ident.ty)

    return_ty = ctx.get_ir_type(node.type)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        mem = ctx.named_values[node]
        params.append(mem)

        return_mem = mem

    for i, param in enumerate(node.params):
        arg_info = func_info.arguments[i]
        param_ty = ctx.get_ir_type(param.type)

        if arg_info.kind == ABIArgKind.Indirect:
            val = get_lvalue(param, block, ctx)
        else:
            if param_ty != arg_info.ty:
                src_size = ctx.module.data_layout.get_type_alloc_size(
                    arg_info.ty)
                dst_size = ctx.module.data_layout.get_type_alloc_size(
                    param_ty)
                if src_size <= dst_size:
                    val = get_lvalue(param, block, ctx)
                    val = BitCastInst(block, val, PointerType(arg_info.ty, 0))
                    val = LoadInst(block, val)
                else:
                    raise NotImplementedError()
            else:
                block, val = build_ir_expr(param, block, ctx)

        params.append(val)

    call_inst = CallInst(block, ctx.funcs[str(
        node.ident.name)].func_ty, ctx.funcs[str(node.ident.name)], params)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        value = LoadInst(block, return_mem)
        return block, value

    return block, call_inst


def build_ir_expr_int_const(node, block, ctx):
    return block, ConstantInt(node.val, ctx.get_ir_type(node.type))


def build_ir_expr_float_const(node, block, ctx):
    return block, ConstantFP(node.val, ctx.get_ir_type(node.type))


def build_ir_expr_accessor(node, block, ctx):
    assert isinstance(node, (TypedAccessorOp,))

    lval = get_lvalue(node, block, ctx)
    return block, LoadInst(block, lval)


def build_ir_expr_indexer(node, block, ctx):
    assert isinstance(node, (TypedArrayIndexerOp,))

    lval = get_lvalue(node, block, ctx)
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

    if isinstance(node, TypedAccessorOp):
        return build_ir_expr_accessor(node, block, ctx)

    if isinstance(node, TypedArrayIndexerOp):
        return build_ir_expr_indexer(node, block, ctx)

    if isinstance(node, TypedUnaryOp):
        return build_ir_expr_unary_op(node, block, ctx)

    if isinstance(node, TypedFunctionCall):
        return build_ir_expr_func_call(node, block, ctx)

    raise Exception("Unreachable")


def build_ir_expr_stmt(node, block, ctx):
    block, build_ir_expr(node.expr, block, ctx)
    return block


TARGET_GPU = 0

if TARGET_GPU == 1:

    def build_ir_if_stmt(node, block, ctx):
        nested = len(ctx.preds) > 0
        if nested:
            pred = PropPredInst(block, node.cond, ctx.preds[-1])
        else:
            pred = PropPredInst(block, node.cond)

        ctx.preds.append(pred)

        block = build_ir_stmt(node.then_stmt, block, ctx)

        InvPredInst(block, pred)

        block = build_ir_stmt(node.else_stmt, block, ctx)

        RestorePredInst(block, pred)
        ctx.preds.pop()

        return block


else:
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


if TARGET_GPU == 1:

    def build_ir_while_stmt(node, block, ctx):
        nested = len(ctx.preds) > 0
        if nested:
            pred = PushPredInst(block, ctx.preds[-1])
        else:
            pred = PushPredInst(block)

        # append 3 blocks
        block_cond = BasicBlock(block.func, block)
        block_then = BasicBlock(block.func, block_cond)
        block_cont = BasicBlock(block.func, block_then)

        # save pred
        ctx.preds.append(pred)

        JumpInst(block, block_cond)

        block, cond = build_ir_expr(node.cond, block_cond, ctx)

        BranchPredInst(block_cond, cond, block_then, block_cont, pred)

        ctx.push_break_target(block_cont)
        ctx.push_continue_target(block_cond)

        block_then = build_ir_stmt(node.stmt, block_then, ctx)
        JumpInst(block_then, block_cond)

        ctx.pop_break_target()
        ctx.pop_continue_target()

        # restore pred
        RestorePredInst(block, pred)
        ctx.preds.pop()

        return block_cont


else:
    def build_ir_while_stmt(node, block, ctx):
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


if TARGET_GPU == 1:

    def build_ir_for_stmt(node, block, ctx):
        nested = len(ctx.preds) > 0
        if nested:
            pred = PushPredInst(block, ctx.preds[-1])
        else:
            pred = PushPredInst(block)

        block = build_ir_stmt(node.init, block, ctx)

        # append 3 blocks
        block_cond = BasicBlock(block.func, block)
        block_then = BasicBlock(block.func, block_cond)
        block_cont = BasicBlock(block.func, block_then)

        # save pred
        ctx.preds.append(pred)

        JumpInst(block, block_cond)

        block_cond, cond = build_ir_expr(node.cond, block_cond, ctx)

        BranchPredInst(block_cond, cond, block_then, block_cont, pred)

        ctx.push_break_target(block_cont)
        ctx.push_continue_target(block_cond)

        block_then = build_ir_stmt(node.stmt, block_then, ctx)
        build_ir_expr(node.loop, block_then, ctx)
        JumpInst(block_then, block_cond)

        ctx.pop_break_target()
        ctx.pop_continue_target()

        # restore pred
        RestorePredInst(block_cont, pred)
        ctx.preds.pop()

        return block_cont
else:
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
        block_then, _ = build_ir_expr(node.loop, block_then, ctx)
        JumpInst(block_then, block_cond)

        ctx.pop_break_target()
        ctx.pop_continue_target()

        return block_cont


def build_ir_compound_stmt(node, block, ctx):
    for stmt in node.stmts:
        block = build_ir_stmt(stmt, block, ctx)

    return block


if TARGET_GPU == 1:
    pass
else:
    def build_ir_continue_stmt(node, block, ctx):
        JumpInst(block, ctx.get_continue_target())

        block = BasicBlock(block.func, block)
        return block


if TARGET_GPU == 1:
    pass
else:
    def build_ir_break_stmt(node, block, ctx):
        JumpInst(block, ctx.get_break_target())

        block = BasicBlock(block.func, block)
        return block


if TARGET_GPU == 1:
    def build_ir_return_stmt(node, block, ctx):
        if len(ctx.preds) > 0:
            ReturnPredInst(block, node.expr, ctx.preds[-1])
        else:
            ReturnPredInst(block, node.expr)
        return block
else:
    def build_ir_return_stmt(node, block, ctx):
        func_info = ctx.func_info
        if func_info.return_info.kind == ABIArgKind.Ignore:
            JumpInst(block, ctx.get_return_target())

            block = BasicBlock(block.func, block)
            return block

        if func_info.return_info.kind == ABIArgKind.Indirect:
            rhs = get_lvalue(node.expr, block, ctx)
            lhs = ctx.return_value

            size, align = ctx.module.data_layout.get_type_size_in_bits(
                lhs.ty.elem_ty)
            align = 4
            rhs = BitCastInst(block, rhs, PointerType(i8, 0))
            lhs = BitCastInst(block, lhs, PointerType(i8, 0))

            memcpy = ctx.get_memcpy_func(lhs, rhs, size)
            CallInst(block, memcpy.func_ty, memcpy, [lhs, rhs, ConstantInt(
                int(size / 8), i32), ConstantInt(int(align / 8), i32), ConstantInt(0, i1)])

            JumpInst(block, ctx.get_return_target())
        else:
            block, rhs = build_ir_expr(node.expr, block, ctx)
            lhs = ctx.return_value

            if lhs.ty.elem_ty != rhs.ty:
                src_size = ctx.module.data_layout.get_type_alloc_size(rhs.ty)
                dst_size = ctx.module.data_layout.get_type_alloc_size(
                    lhs.ty.elem_ty)
                if src_size <= dst_size:
                    lhs = BitCastInst(block, lhs, PointerType(rhs.ty, 0))
                else:
                    raise NotImplementedError()

            StoreInst(block, rhs, lhs)
            JumpInst(block, ctx.get_return_target())

        block = BasicBlock(block.func, block)
        return block


def build_ir_switch_stmt(node, block, ctx):
    stmts = node.stmts

    blocks = {}
    cur_block = block
    for stmt in stmts:
        if isinstance(stmt, CaseLabel):
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

    ctx.push_branch_target(cont_block)

    cur_label = None
    for stmt in stmts:
        if isinstance(stmt, CaseLabel):
            cur_label = stmt
        else:
            assert(cur_block is not None)
            blocks[cur_label] = build_ir_stmt(stmt, blocks[cur_label], ctx)

    ctx.pop_branch_target()

    for case_block in blocks.values():
        JumpInst(case_block, cont_block)

    return cont_block


def build_ir_stmt(node, block, ctx):
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
                build_ir_assign_op(variable, init_expr, block, ctx)

        return block

    if isinstance(node, SwitchStmt):
        return build_ir_switch_stmt(node, block, ctx)

    return block


def build_ir_stack_alloc(node, block, ctx):
    # Allocate local variables.
    if isinstance(node, TypedVariable):
        for variable, _ in node.idents:
            size = ConstantInt(1, i32)
            ty = variable.type
            if isinstance(variable.type, ast.types.ArrayType):
                block, size = build_ir_expr(ty.size, block, ctx)
                ty = ty.elem_ty

            mem = AllocaInst(block, size,
                             ctx.get_ir_type(ty), 0)

            ctx.named_values[variable.val] = mem

    # Allocate params of all callee functions.
    if isinstance(node, TypedFunctionCall):
        for i, (param_type, _, _) in enumerate(node.ident.ty.params):
            mem = AllocaInst(block, ConstantInt(1, i32),
                             ctx.get_ir_type(param_type), 0)
            ctx.named_values[(node, i)] = mem

    # Allocate return value of all callee functions.
    if isinstance(node, TypedFunctionCall):
        return_ty = ctx.get_ir_type(node.ident.ty.return_ty)
        func_info = compute_func_info(ctx, node.ident.ty)
        if func_info.return_info.kind == ABIArgKind.Indirect:
            mem = AllocaInst(block, ConstantInt(1, i32), return_ty, 0)
            ctx.named_values[node] = mem


def build_ir_func_header(node, func, ctx):
    func_info = compute_func_info(ctx, node.ident.ty)

    return_ty = ctx.get_ir_type(node.type)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        ir_arg = Argument(PointerType(return_ty, 0))
        ir_arg.add_attribute(Attribute(AttributeKind.StructRet))
        ctx.return_value = ir_arg
        func.add_arg(ir_arg)

    for (arg_ty, arg_quals, arg_name), arg_info in zip(node.params, func_info.arguments):
        param_ty = ctx.get_ir_type(arg_ty)
        coerced_param_ty = arg_info.ty
        if arg_info.kind == ABIArgKind.Indirect:
            ir_arg = ir_arg_ptr = Argument(
                PointerType(param_ty, 0), arg_name)
        else:
            ir_arg = Argument(coerced_param_ty, arg_name)
        func.add_arg(ir_arg)


def build_ir_func(node, block, ctx):
    func_info = compute_func_info(ctx, node.proto.ident.ty)
    ctx.func_info = func_info

    return_ty = ctx.get_ir_type(node.proto.type)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        ir_arg = Argument(PointerType(return_ty, 0))
        ir_arg.add_attribute(Attribute(AttributeKind.StructRet))
        ctx.return_value = ir_arg
        block.func.add_arg(ir_arg)

    for arg, arg_info in zip(node.params, func_info.arguments):
        param_ty = ctx.get_ir_type(arg.ty)
        coerced_param_ty = arg_info.ty
        if arg_info.kind == ABIArgKind.Indirect:
            ir_arg = ir_arg_ptr = Argument(
                PointerType(param_ty, 0), arg.name)
        else:
            ir_arg = Argument(coerced_param_ty, arg.name)
            ir_arg_ptr = AllocaInst(block, ConstantInt(1, i32), param_ty, 0)

            ir_arg_mem_ptr = ir_arg_ptr

            if coerced_param_ty != param_ty:
                ir_arg_mem_ptr = BitCastInst(
                    block, ir_arg_mem_ptr, PointerType(coerced_param_ty, 0))
            StoreInst(block, ir_arg, ir_arg_mem_ptr)

        ctx.named_values[arg] = ir_arg_ptr
        block.func.add_arg(ir_arg)

    block_end = BasicBlock(block.func)

    if func_info.return_info.kind in [ABIArgKind.Indirect, ABIArgKind.Ignore]:
        ReturnInst(block_end, None)
    else:
        ctx.return_value = AllocaInst(block, ConstantInt(1, i32), return_ty, 0)
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

    traverse_depth(node, enter_func=build_ir_stack_alloc, args=(block, ctx))

    ctx.push_return_target(block_end)
    for stmt in node.stmts:
        block = build_ir_stmt(stmt, block, ctx)
        if len(ctx.branch_target) == 0 and isinstance(stmt, ReturnStmt):
            break
    ctx.pop_return_target()

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
        if ty.name in ["int", "uint"]:
            return TypeInfo(32, 32)
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

    global_named_values = {}
    for decl in ast:
        if isinstance(decl, TypedVariable):
            for ident, init_expr in decl.idents:
                ty = ctx.get_ir_type(ident.type)
                if not init_expr:
                    init = get_constant_null_value(ty)
                else:
                    init = evaluate_constant_expr(ctx, init_expr)

                linkage = GlobalLinkage.Global
                if "uniform" in ident.val.ty_qual:
                    linkage = GlobalLinkage.Global

                thread_local = ThreadLocalMode.GeneralDynamicTLSModel
                if "shared" in ident.val.ty_qual:
                    thread_local = ThreadLocalMode.NotThreadLocal

                name = ident.val.name
                global_named_values[ident.val] = module.add_global(name,
                                                                   GlobalVariable(ty, False, linkage, name, thread_local, init))

        if isinstance(decl, TypedFunctionProto):
            func_ty = ctx.get_ir_type(decl.ident.ty)
            func_name = str(decl.ident.name)
            func = module.add_func(func_name,
                                   Function(module, func_ty, GlobalLinkage.Global, func_name))
            ctx.funcs[func_name] = func

            build_ir_func_header(decl, func, ctx)

        if isinstance(decl, TypedFunction):
            func_ty = ctx.get_ir_type(decl.proto.ident.ty)
            func_name = str(decl.proto.ident.name)

            func = module.add_func(func_name,
                                   Function(module, func_ty, GlobalLinkage.Global, func_name))

            ctx.begin_func(func)

            ctx.named_values.update(global_named_values)
            block = ctx.current_block
            build_ir_func(decl, block, ctx)

            ctx.end_func()

    return module
