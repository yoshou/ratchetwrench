#!/usr/bin/env python
# -*- coding: utf-8 -*-

from c.sema import is_pointer_type
from c.sema import get_type_of
from c.symtab import FunctionSymbol, VariableSymbol
from ast.node import *
from ir.types import *
from ir.values import *
from ir.data_layout import *
import ast


def evaluate_constant_expr(ctx, expr):
    if isinstance(expr, FloatingConstantExpr):
        return ConstantFP(expr.val, ctx.get_ir_type(expr.type))

    if isinstance(expr, IntegerConstantExpr):
        return ConstantInt(expr.val, ctx.get_ir_type(expr.type))

    if isinstance(expr, TypedUnaryOp):
        op = expr.op

        val = evaluate_constant_expr(ctx, expr.expr)
        if op == "-":
            def calc(a): return -a
        if op == "~":
            def calc(a): return ((1 << 64) - 1) ^ a

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

        if isinstance(lhs_val, (ConstantInt, ConstantFP)) and isinstance(rhs_val, (ConstantInt, ConstantFP)):
            if isinstance(lhs_val, (ConstantInt, CastInst)):
                def construct_val(value, ty):
                    return ConstantInt(value, ty)
            elif isinstance(lhs_val, ConstantFP):
                def construct_val(value, ty):
                    return ConstantFP(value, ty)

            if op == "+":
                def calc(a, b): return construct_val(a.value + b.value, a.ty)
            elif op == "-":
                def calc(a, b): return construct_val(a.value - b.value, a.ty)
            elif op == "*":
                def calc(a, b): return construct_val(a.value * b.value, a.ty)
            elif op == "/":
                def calc(a, b): return construct_val(a.value / b.value, a.ty)
            elif op == "<<":
                def calc(a, b): return construct_val(a.value << b.value, a.ty)
            elif op == ">>":
                def calc(a, b): return construct_val(a.value >> b.value, a.ty)
            elif op == "&":
                def calc(a, b): return construct_val(a.value & b.value, a.ty)
            elif op == "|":
                def calc(a, b): return construct_val(a.value | b.value, a.ty)
            elif op == "^":
                def calc(a, b): return construct_val(a.value ^ b.value, a.ty)

            result = calc(lhs_val, rhs_val)

            return result
        elif isinstance(lhs_val, IntToPtrInst) and isinstance(rhs_val, (ConstantInt, ConstantFP)):
            def construct_val(value, ty):
                return ConstantInt(value, ty)

            lhs_size = ctx.module.data_layout.get_type_alloc_size(
                lhs_val.rs.ty)
            rhs_size = ctx.module.data_layout.get_type_alloc_size(rhs_val.ty)

            if lhs_size >= rhs_size:
                result_ty = lhs_val.rs.ty
            else:
                result_ty = rhs_val.ty

            if op == "+":
                def calc(a, b): return construct_val(
                    a.value + b.value, result_ty)
            elif op == "-":
                def calc(a, b): return construct_val(
                    a.value - b.value, result_ty)
            elif op == "*":
                def calc(a, b): return construct_val(
                    a.value * b.value, result_ty)
            elif op == "/":
                def calc(a, b): return construct_val(
                    a.value / b.value, result_ty)
            elif op == "<<":
                def calc(a, b): return construct_val(
                    a.value << b.value, result_ty)
            elif op == ">>":
                def calc(a, b): return construct_val(
                    a.value >> b.value, result_ty)
            elif op == "&":
                def calc(a, b): return construct_val(
                    a.value & b.value, result_ty)
            elif op == "|":
                def calc(a, b): return construct_val(
                    a.value | b.value, result_ty)
            elif op == "^":
                def calc(a, b): return construct_val(
                    a.value ^ b.value, result_ty)

            result = calc(lhs_val.rs, rhs_val)
            result_ty = ctx.get_ir_type(expr.type)
            return IntToPtrInst(None, result, result_ty)
        return result

    if isinstance(expr, StringLiteralExpr):
        if expr in ctx.named_values:
            return block, ctx.named_values[expr]

        values = []
        for value in expr.val:
            values.append(ConstantInt(ord(value), i8))
        values.append(ConstantInt(0, i8))

        value = ConstantArray(values, ctx.get_ir_type(expr.type))

        return value

    if isinstance(expr, TypedIdentExpr):
        if not isinstance(get_type_of(expr), ast.types.EnumType):
            _, value = get_lvalue(expr, None, ctx)
            assert(isinstance(value, (GlobalVariable, Function)))
            return value

        value = expr.type.values[expr.val.name]

        return ConstantInt(value, i32)

    if isinstance(expr, TypedInitializerList):
        field_idx = 0
        if isinstance(get_type_of(expr), ast.types.CompositeType):
            field_vals = []
            for field_ty, field_name, _ in expr.type.fields:
                designator, field_val = expr.exprs[field_idx]

                if designator:
                    raise NotImplementedError()

                field_idx += 1
                field_vals.append(evaluate_constant_expr(ctx, field_val))
            return ConstantStruct(field_vals, ctx.get_ir_type(expr.type))
        elif isinstance(get_type_of(expr), (ast.types.ArrayType, ast.types.PointerType)):
            values = []

            ty = ctx.get_ir_type(expr.type)
            elem_ty = ctx.get_ir_type(expr.type.elem_ty)

            for designators, elem_expr in expr.exprs:
                if designators:
                    value = evaluate_constant_expr(ctx, elem_expr)
                    for designator in designators:
                        if isinstance(designator, IntegerConstantExpr):
                            designator_idx = designator.val
                        else:
                            assert(isinstance(designator, TypedIdentExpr))
                            designator_idx = evaluate_constant_expr(
                                ctx, designator).value

                        if designator_idx > len(values):
                            values.extend([get_constant_null_value(
                                ty.elem_ty)] * (designator_idx - len(values)))

                        if elem_ty != value.ty:
                            if isinstance(elem_expr, StringLiteralExpr):
                                str_const = GlobalVariable(
                                    value.ty, True, GlobalLinkage.Private, f".str{len(ctx.module.globals)}", initializer=value)

                                ctx.module.add_global(
                                    f".str{len(ctx.module.globals)}", str_const)

                                value = ctx.named_values[elem_expr] = str_const

                                value = GetElementPtrInst(
                                    None, value, value.ty, ConstantInt(0, i32), ConstantInt(0, i32))
                            else:
                                value = cast(None, value, elem_ty,
                                             is_signed_type(elem_expr), ctx)
                        values.insert(designator_idx, value)
                else:
                    if isinstance(elem_ty, PointerType) and isinstance(elem_expr, StringLiteralExpr):
                        _, value = get_lvalue(elem_expr, None, ctx)
                    else:
                        value = evaluate_constant_expr(ctx, elem_expr)

                    if elem_ty != value.ty:
                        value = cast(None, value, elem_ty,
                                     is_signed_type(elem_expr), ctx)
                    values.append(value)

            if ty.size > len(values):
                values.extend([get_constant_null_value(ty.elem_ty)]
                              * (ty.size - len(values)))

            ty = ArrayType(ty.elem_ty, len(values))

            return ConstantArray(values, ty)

    def sign_extend(value, bits):
        sign_bit = 1 << (bits - 1)
        return (value & (sign_bit - 1)) - (value & sign_bit)

    if isinstance(expr, TypedCastExpr):
        ty = ctx.get_ir_type(expr.type)

        result = evaluate_constant_expr(ctx, expr.expr)
        if ty != result.ty:
            if isinstance(result.ty, ArrayType):
                pass
            else:
                result = cast(None, result,
                              ty, is_signed_type(expr.expr.type), ctx)

        if isinstance(result, SExtInst):
            src_size = ctx.module.data_layout.get_type_alloc_size(result.rs.ty)
            dst_size = ctx.module.data_layout.get_type_alloc_size(result.ty)

            assert(isinstance(result.rs, ConstantInt))

            bits = dst_size * 8
            value = sign_extend(result.rs.value, bits)
            return ConstantInt(value, result.ty)
        elif isinstance(result, TruncInst):
            src_size = ctx.module.data_layout.get_type_alloc_size(result.rs.ty)
            dst_size = ctx.module.data_layout.get_type_alloc_size(result.ty)

            assert(isinstance(result.rs, ConstantInt))

            bits = dst_size * 8
            value = result.rs.value & ((1 << bits) - 1)
            return ConstantInt(value, result.ty)
        elif isinstance(result, IntToPtrInst):
            assert(isinstance(result.rs, ConstantInt))

            return result
        elif isinstance(result, BitCastInst):
            src_size = ctx.module.data_layout.get_type_alloc_size(result.rs.ty)
            dst_size = ctx.module.data_layout.get_type_alloc_size(result.ty)

            assert(src_size == dst_size)

            return result
        elif isinstance(result, Constant):
            return result

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
        self.goto_jump_query = []
        self.goto_targets = {}

    def query_goto_jump(self, jump_from, goto_label):
        if goto_label in self.goto_targets:
            JumpInst(jump_from, self.goto_targets[goto_label])
            return

        self.goto_jump_query.append((jump_from, goto_label))

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
            return {}

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
        if isinstance(ast_ty, ast.types.QualType):
            return self.get_ir_type(ast_ty.ty)

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
                ty = self.module.structs[ast_ty.name]
                return ty

            if not ast_ty.fields:
                # Incomplete type
                return PrimitiveType("i8")

            ty = StructType(name=ast_ty.name, is_packed=ast_ty.is_packed)
            self.module.add_struct_type(ast_ty.name, ty)

            fields = []
            bitpos = -1
            for field_ty, name, bit in ast_ty.fields:
                field_type = self.get_ir_type(field_ty)
                field_size, _ = self.module.data_layout.get_type_size_in_bits(
                    field_type)

                if not bit:
                    fields.append(field_type)
                    bitpos = -1
                else:
                    assert(isinstance(bit, IntegerConstantExpr))
                    bitval = bit.val

                    assert(bitval < field_size)

                    if bitval == 0:
                        bitpos = -1
                    else:
                        if bitpos == -1 or (bitpos + bitval) > field_size:
                            fields.append(field_type)
                            bitpos = 0

                        bitpos += bitval

            ty.fields = fields
            assert(ast_ty.name)
            return ty

        if isinstance(ast_ty, ast.types.VoidType):
            return void

        if isinstance(ast_ty, ast.types.FunctionType):
            func_info = compute_func_info(self, ast_ty)

            params_ty = []

            return_ty = func_info.return_info.ty
            if func_info.return_info.kind == ABIArgKind.Direct:
                pass
            elif func_info.return_info.kind == ABIArgKind.Indirect:
                params_ty.append(return_ty)
                return_ty = void
            elif func_info.return_info.kind == ABIArgKind.Ignore:
                return_ty = void
            else:
                raise NotImplementedError()

            params_ty.extend([self.get_ir_type(ast_param_ty)
                              for (ast_param_ty, _, _) in ast_ty.params if not isinstance(ast_param_ty, ast.types.VoidType)])

            return FunctionType(return_ty, params_ty, ast_ty.is_variadic)

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

    def get_returnaddress_func(self, level):
        arg_tys = [i32]
        return self.module.get_or_declare_intrinsic_func("llvm.returnaddress", arg_tys)


def get_lvalue(node, block, ctx):
    if isinstance(node, TypedIdentExpr):
        if node.val in ctx.named_values:
            return block, ctx.named_values[node.val]

        return block, ctx.global_named_values[node.val]

    if isinstance(node, TypedUnaryOp):
        if node.op == "*":
            block, expr = build_ir_expr(node.expr, block, ctx)
            return block, expr
            return block, LoadInst(block, expr)
        if node.op == "&":
            return get_lvalue(node.expr, block, ctx)
        if node.op in ["++", "--"]:
            block, mem = get_lvalue(node.expr, block, ctx)
            value = LoadInst(block, mem)

            if node.op == "++":
                op = 'add'
            else:
                op = 'sub'

            if isinstance(get_type_of(node.expr), ast.types.PointerType):
                inc_value = GetElementPtrInst(
                    block, value, value.ty, ConstantInt(1, i64))
            else:
                one = ConstantInt(1, ctx.get_ir_type(node.expr.type))
                inc_value = BinaryInst(block, op, value, one)

            StoreInst(block, inc_value, mem)

            return block, mem

    if isinstance(node, TypedPostOp):
        if node.op in ["++", "--"]:
            block, mem = get_lvalue(node.expr, block, ctx)
            value = LoadInst(block, mem)

            if node.op == "++":
                op = 'add'
            else:
                op = 'sub'

            if isinstance(get_type_of(node.expr), ast.types.PointerType):
                if op == "add":
                    inc_value = GetElementPtrInst(
                        block, value, value.ty, ConstantInt(1, i32))
                else:
                    inc_value = GetElementPtrInst(
                        block, value, value.ty, ConstantInt(-1, i32))
            else:
                one = ConstantInt(1, ctx.get_ir_type(node.expr.type))
                inc_value = BinaryInst(block, op, value, one)

            StoreInst(block, inc_value, mem)

            return block, mem

    if isinstance(node, TypedCommaOp):
        return block, get_lvalue(node.exprs[-1], block, ctx)

    def get_field_ptr(ty, field):
        assert(isinstance(ty, ast.types.CompositeType))

        if ty.contains_field(field):
            field_idx = ty.get_field_idx(field)
            if ty.is_union:
                return [(None, ty.get_field_type_by_name(field))]
            return [(field_idx, None)]

        for idx, (field_ty, field_name, field_bit) in enumerate(ty.fields):
            if field_bit:
                raise NotImplementedError()

            if not field_name:
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

            idx_or_casts = get_field_ptr(
                get_type_of(node.obj).elem_ty, node.field.val)

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
            idx_or_casts = get_field_ptr(get_type_of(node.obj), node.field.val)

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
                block, ptr, ptr.ty, ConstantInt(0, idx.ty), idx)
        elif isinstance(ptr.ty, PointerType):
            block, idx = build_ir_expr(node.idx, block, ctx)
            ptr = LoadInst(block, ptr)
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
            value.ty, True, GlobalLinkage.Private, f".str{len(ctx.module.globals)}", initializer=value)

        ctx.module.add_global(value.name, value)

        ctx.named_values[node] = value

        return block, value

    if isinstance(node, TypedConditionalExpr):
        block, cond = build_ir_expr(node.cond_expr, block, ctx)

        if cond.ty != i1:
            cond = CmpInst(block, "ne", cond, ConstantInt(0, cond.ty))

        block_true = BasicBlock(block.func, block)
        block_false = BasicBlock(block.func, block_true)
        block_cont = BasicBlock(block.func, block_false)

        BranchInst(block, cond, block_true, block_false)

        ctx.push_branch_target(block_cont)

        block_true, true_value = get_lvalue(node.true_expr, block_true, ctx)
        block_false, false_value = get_lvalue(
            node.false_expr, block_false, ctx)

        ctx.pop_branch_target()

        JumpInst(block_true, block_cont)
        JumpInst(block_false, block_cont)

        values = [
            true_value, block_true,
            false_value, block_false,
        ]

        return block_cont, PHINode(block_cont, true_value.ty, values)

    if isinstance(node, TypedCastExpr):
        if isinstance(get_type_of(node.expr), ast.types.PointerType):
            block, value = get_lvalue(node.expr, block, ctx)
            value = BitCastInst(block, value, PointerType(
                ctx.get_ir_type(node.type), 0))
            return block, value
        elif isinstance(get_type_of(node.expr), ast.types.ArrayType):
            block, value = build_ir_expr(node.expr, block, ctx)

            alloca_insert_pt = get_alloca_insert_pt(ctx)
            mem = alloca_insert_pt = AllocaInst(
                alloca_insert_pt, ConstantInt(1, i32), value.ty, 0)
            StoreInst(block, value, mem)

            mem = BitCastInst(block, mem, PointerType(
                ctx.get_ir_type(node.type), 0))
            return block, mem

    if isinstance(node, (TypedBinaryOp, TypedFunctionCall, IntegerConstantExpr)):
        alloca_insert_pt = get_alloca_insert_pt(ctx)

        block, rhs = build_ir_expr(node, block, ctx)
        mem = alloca_insert_pt = AllocaInst(
            alloca_insert_pt, ConstantInt(1, i32), rhs.ty, 0)
        StoreInst(block, rhs, mem)
        return block, mem

    print(node)
    raise Exception("Unreachable")


def build_ir_assign_op_init_list(lhs_node, rhs_lst, block, ctx):
    if isinstance(get_type_of(lhs_node), ast.types.CompositeType):
        lhs_node_ty = get_type_of(lhs_node)
        if lhs_node_ty.is_union:
            lhs_node_ty, _, _ = lhs_node_ty.fields[0]

        field_idx = 0
        for field_ty, field_name, _ in lhs_node_ty.fields:
            field_val = TypedAccessorOp(lhs_node, Ident(field_name), field_ty)

            block, _ = build_ir_assign_op(
                field_val, rhs_lst[field_idx], block, ctx)

            field_idx += 1

        return block, None

    if isinstance(get_type_of(lhs_node), ast.types.ArrayType):
        elem_idx = 0
        elem_ty = get_type_of(lhs_node).elem_ty

        for i in range(lhs_node.type.size):
            elem_val = ast.node.TypedArrayIndexerOp(
                lhs_node, ast.node.IntegerConstantExpr(i, ast.types.PrimitiveType("int")), elem_ty)

            block, _ = build_ir_assign_op(
                elem_val, rhs_lst[elem_idx], block, ctx)

            elem_idx += 1

        return block, None

    raise Exception("Unreachable")


def get_qualifier(lvalue):
    if isinstance(lvalue, TypedIdentExpr):
        if isinstance(lvalue.val.ty, ast.types.QualType):
            return lvalue.val.ty.quals
    if isinstance(lvalue, TypedAccessorOp):
        if isinstance(lvalue.type, ast.types.QualType):
            return lvalue.type.quals
    if isinstance(lvalue, TypedArrayIndexerOp):
        if isinstance(lvalue.type, ast.types.QualType):
            return lvalue.type.quals
    return ast.types.Qualifier.Undefined


def build_ir_assign_op(lhs_node, rhs, block, ctx):
    if isinstance(rhs, list):
        return build_ir_assign_op_init_list(lhs_node, rhs, block, ctx)

    block, lhs = get_lvalue(lhs_node, block, ctx)

    qual = get_qualifier(lhs_node)

    if lhs.ty.elem_ty != rhs.ty:
        if isinstance(lhs_node.type, ast.types.ArrayType):
            size, align = ctx.module.data_layout.get_type_size_in_bits(
                lhs.ty.elem_ty)
            align = 4

            rhs = BitCastInst(block, rhs, PointerType(i8, 0))

            assert(isinstance(lhs.ty.elem_ty, ArrayType))
            lhs = GetElementPtrInst(
                block, lhs, lhs.ty, ConstantInt(0, i32), ConstantInt(0, i32))
            lhs = BitCastInst(block, lhs, PointerType(i8, 0))

            memcpy = ctx.get_memcpy_func(lhs, rhs, size)
            CallInst(block, memcpy.func_ty, memcpy, [lhs, rhs, ConstantInt(
                int(size / 8), i32), ConstantInt(int(align / 8), i32), ConstantInt(0, i1)])
            return block, None

        rhs = cast(block, rhs, lhs.ty.elem_ty,
                   is_signed_type(lhs_node.type), ctx)

    is_volatile = False
    if qual & ast.types.Qualifier.Volatile:
        is_volatile = True

    StoreInst(block, rhs, lhs, is_volatile)
    return block, rhs


def build_ir_expr_assign_op(node, block, ctx):
    if node.op in ["+=", "-=", "*=", "/=", "<<=", ">>=", "&=", "|=", "^="]:
        block, lhs = build_ir_expr(node.lhs, block, ctx)
        block, rhs = build_ir_expr(node.rhs, block, ctx)

        if isinstance(node.lhs.type, ast.types.PointerType) and is_integer_type(node.rhs.type):
            if node.op == "-=":
                rhs = BinaryInst(block, "sub", ConstantInt(0, rhs.ty), rhs)

            assert(node.op in ["+=", "-="])
            rhs = GetElementPtrInst(block, lhs, lhs.ty, rhs)

            return build_ir_assign_op(node.lhs, rhs, block, ctx)

        if rhs.ty != lhs.ty:
            rhs = cast(block, rhs, lhs.ty, is_signed_type(node.lhs.type), ctx)

        if is_float_ty(lhs.ty):
            prefix = "f"
        else:
            prefix = ""

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
        elif node.op == "&=":
            op = "and"
        elif node.op == "|=":
            op = "or"
        elif node.op == "^=":
            op = "xor"

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

    src_size, _ = ctx.module.data_layout.get_type_size_in_bits(from_type)
    dst_size, _ = ctx.module.data_layout.get_type_size_in_bits(to_type)

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


def is_constant_zero(value):
    if not isinstance(value.ty, PrimitiveType):
        return False

    return value.value == 0


def build_ir_expr_cmp_op(node, block, ctx):
    block, lhs = build_ir_expr(node.lhs, block, ctx)
    block, rhs = build_ir_expr(node.rhs, block, ctx)

    result_ty = ctx.get_ir_type(get_type_of(node))

    if isinstance(lhs.ty, PointerType) and is_constant_zero(rhs):
        rhs = ConstantPointerNull(lhs.ty)

    assert(lhs.ty == rhs.ty)

    if is_floating_type(get_type_of(node.lhs)):
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

    elif is_integer_type(get_type_of(node.lhs)):
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
            op = 'sle' if is_signed else 'ule'
        elif node.op == '>=':
            op = 'sge' if is_signed else 'uge'
        else:
            raise ValueError(
                "The compare node has invalid operator: " + node.op)

        return block, CmpInst(block, op, lhs, rhs)

    elif isinstance(get_type_of(node.lhs), ast.types.PointerType):
        is_signed = False

        if node.op == '==':
            op = 'eq'
        elif node.op == '!=':
            op = 'ne'
        elif node.op == '<':
            op = 'slt' if is_signed else 'ult'
        elif node.op == '>':
            op = 'sgt' if is_signed else 'ugt'
        elif node.op == '<=':
            op = 'sle' if is_signed else 'ule'
        elif node.op == '>=':
            op = 'sge' if is_signed else 'uge'
        else:
            raise ValueError(
                "The compare node has invalid operator: " + node.op)

        return block, CmpInst(block, op, lhs, rhs)

    raise ValueError("Invalid type to compare")


def build_ir_expr_logical_op(node, block, ctx):
    block, lhs = build_ir_expr(node.lhs, block, ctx)

    if lhs.ty != i1:
        if isinstance(lhs.ty, PointerType):
            cond = PtrToIntInst(block, lhs, i64)
            lhs = CmpInst(block, "ne", lhs, ConstantInt(0, i64))
        else:
            lhs = cast(block, lhs, i32, is_signed_type(node.lhs.type), ctx)
            lhs = CmpInst(block, "ne", lhs, ConstantInt(0, i32))

    rhs_block = BasicBlock(block.func, block)
    rhs_block_entry = rhs_block
    rhs_block, rhs = build_ir_expr(node.rhs, rhs_block, ctx)

    if rhs.ty != i1:
        if isinstance(rhs.ty, PointerType):
            cond = PtrToIntInst(rhs_block, rhs, i64)
            lhs = CmpInst(rhs_block, "ne", lhs, ConstantInt(0, i64))
        else:
            rhs = cast(rhs_block, rhs, i32, is_signed_type(node.rhs.type), ctx)
            rhs = CmpInst(rhs_block, "ne", rhs, ConstantInt(0, i32))

    cont_block = BasicBlock(block.func, rhs_block)

    if node.op == '&&':
        BranchInst(block, lhs, rhs_block_entry, cont_block)
        JumpInst(rhs_block, cont_block)
        result = PHINode(cont_block, i1, [lhs, block, rhs, rhs_block])
    elif node.op == '^^':
        op = 'xor'
        raise ValueError(
            "The compare node has invalid operator: " + node.op)
    elif node.op == '||':
        BranchInst(block, lhs, cont_block, rhs_block_entry)
        JumpInst(rhs_block, cont_block)
        result = PHINode(cont_block, i1, [lhs, block, rhs, rhs_block])
    else:
        raise ValueError(
            "The compare node has invalid operator: " + node.op)

    return cont_block, result


def is_integer_type(ty):
    if not isinstance(ty, ast.types.PrimitiveType):
        return False

    return ty.name in ["char", "unsigned char", "short", "unsigned short", "int", "unsigned int", "long", "unsigned long"]


def is_floating_type(ty):
    if not isinstance(ty, ast.types.PrimitiveType):
        return False

    return ty.name in ["float", "double"]


def is_vector_of_integer_type(ty):
    return isinstance(ty, ast.types.VectorType) and is_integer_type(ty.elem_ty)


def is_vector_of_floating_type(ty):
    return isinstance(ty, ast.types.VectorType) and is_floating_type(ty.elem_ty)


def build_ir_expr_arith_op(node, block, ctx):
    if is_integer_type(node.type) or is_vector_of_integer_type(node.type):
        block, lhs = build_ir_expr(node.lhs, block, ctx)
        block, rhs = build_ir_expr(node.rhs, block, ctx)

        is_signed = is_signed_type(node.lhs.type)

        if node.op == '+':
            op = 'add'
        elif node.op == '-':
            op = 'sub'
        elif node.op == '*':
            op = 'mul'
        elif node.op == '/':
            op = 'sdiv' if is_signed else 'udiv'
        elif node.op == '%':
            op = 'srem' if is_signed else 'urem'
        elif node.op == '>>':
            op = 'ashr' if is_signed else 'lshr'
        elif node.op == '<<':
            op = 'shl'
        elif node.op == '&':
            op = 'and'
        elif node.op == '|':
            op = 'or'
        elif node.op == '^':
            op = 'xor'
        else:
            raise ValueError(
                "The arithematic node has invalid operator: " + node.op)
    elif is_floating_type(node.type) or is_vector_of_floating_type(node.type):
        block, lhs = build_ir_expr(node.lhs, block, ctx)
        block, rhs = build_ir_expr(node.rhs, block, ctx)

        if node.op == '+':
            op = 'fadd'
        elif node.op == '-':
            op = 'fsub'
        elif node.op == '*':
            op = 'fmul'
        elif node.op == '/':
            op = 'fdiv'
        elif node.op == '%':
            op = 'frem'
        else:
            raise ValueError(
                "The arithematic node has invalid operator: " + node.op)
    elif isinstance(get_type_of(node.lhs), ast.types.PointerType) and is_integer_type(get_type_of(node.rhs)):
        block, ptr = build_ir_expr(node.lhs, block, ctx)
        block, idx = build_ir_expr(node.rhs, block, ctx)

        if node.op == '-':
            if isinstance(idx, ConstantInt):
                offset = ConstantInt(-idx.value, idx.ty)
            else:
                offset = BinaryInst(block, "sub", ConstantInt(0, idx.ty), idx)
        else:
            assert(node.op == '+')
            offset = idx

        return block, GetElementPtrInst(block, ptr, ptr.ty, offset)
    elif isinstance(get_type_of(node.lhs), ast.types.PointerType) and isinstance(get_type_of(node.rhs), ast.types.PointerType):
        block, lhs_ptr = build_ir_expr(node.lhs, block, ctx)
        block, rhs_ptr = build_ir_expr(node.rhs, block, ctx)

        lhs_ptr = BitCastInst(block, lhs_ptr, i64)
        rhs_ptr = BitCastInst(block, rhs_ptr, i64)

        elem_size = ctx.module.data_layout.get_type_alloc_size(
            ctx.get_ir_type(node.lhs.type.elem_ty))

        diff = BinaryInst(block, "sub", lhs_ptr, rhs_ptr)
        diff = BinaryInst(block, "udiv", diff,
                          ConstantInt(elem_size, lhs_ptr.ty))

        return block, diff
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
    if node.op in ['=', '+=', '-=', '*=', '/=', '>>=', '<<=', '&=', '|=', '^=']:
        return build_ir_expr_assign_op(node, block, ctx)

    if node.op in ['+', '-', '*', '/', '%', '>>', '<<', '&', '|', '^']:
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

    return block_cont, PHINode(block_cont, true_value.ty, values)


def build_ir_expr_post_op(node, block, ctx):
    if node.op in ["++", "--"]:
        block, mem = get_lvalue(node.expr, block, ctx)
        value = LoadInst(block, mem)

        if node.op == "++":
            op = 'add'
        else:
            op = 'sub'

        if isinstance(get_type_of(node.expr), ast.types.PointerType):
            if op == "add":
                inc_value = GetElementPtrInst(
                    block, value, value.ty, ConstantInt(1, i64))
            else:
                inc_value = GetElementPtrInst(
                    block, value, value.ty, ConstantInt(-1, i64))
        else:
            one = ConstantInt(1, ctx.get_ir_type(node.expr.type))
            inc_value = BinaryInst(block, op, value, one)

        StoreInst(block, inc_value, mem)

        return block, value

    raise Exception("Unreachable")


def build_ir_expr_unary_op(node, block, ctx):
    if node.op in ["++", "--"]:
        block, mem = get_lvalue(node.expr, block, ctx)
        value = LoadInst(block, mem)

        if node.op == "++":
            op = 'add'
        elif node.op == "--":
            op = 'sub'

        if is_pointer_type(get_type_of(node.expr)):
            inc_value = GetElementPtrInst(
                block, value, value.ty, ConstantInt(1, i64))
        else:
            one = ConstantInt(1, ctx.get_ir_type(node.expr.type))
            inc_value = BinaryInst(block, op, value, one)

        StoreInst(block, inc_value, mem)

        return block, inc_value

    if node.op == "!":
        block, val = build_ir_expr(node.expr, block, ctx)

        if isinstance(val.ty, PointerType):
            return block, CmpInst(block, "eq", val, ConstantPointerNull(val.ty))

        assert(isinstance(val.ty, PrimitiveType))
        return block, CmpInst(block, "eq", val, ConstantInt(0, val.ty))

    if node.op == "~":
        block, val = build_ir_expr(node.expr, block, ctx)

        op = 'xor'

        return block, BinaryInst(block, op, val, ConstantInt(-1, val.ty))

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

    if isinstance(get_type_of(node), ast.types.CompositeType):
        block, lval = get_lvalue(node, block, ctx)
        return block, LoadInst(block, lval)

    if isinstance(node.val, FunctionSymbol):
        return block, ctx.funcs[node.val.name]

    if isinstance(get_type_of(node), ast.types.EnumType):
        value = node.type.values[node.val.name]
        assert(isinstance(value, int))
        return block, ConstantInt(value, i32)

    mem = ctx.named_values[node.val]

    if isinstance(get_type_of(node), ast.types.ArrayType):
        return block, GetElementPtrInst(block, mem, mem.ty, ConstantInt(0, i32), ConstantInt(0, i32))

    is_volatile = False
    quals = get_qualifier(node)
    if quals & ast.types.Qualifier.Volatile:
        is_volatile = True

    return block, LoadInst(block, mem, is_volatile)


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
    if isinstance(get_type_of(lhs_node), ast.types.ArrayType):
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

    if isinstance(node.ident, FunctionSymbol):
        if node.ident.name == "__va_start":
            block, va_list = build_ir_expr(node.params[0], block, ctx)
            va_list = BitCastInst(block, va_list, PointerType(i8, 0))
            va_start_func = ctx.get_va_start_func(va_list)
            return block, CallInst(block, va_start_func.func_ty, va_start_func, [va_list])

        if node.ident.name == "__va_end":
            block, va_list = build_ir_expr(node.params[0], block, ctx)
            va_list = BitCastInst(block, va_list, PointerType(i8, 0))
            va_end_func = ctx.get_va_end_func(va_list)
            return block, CallInst(block, va_end_func.func_ty, va_end_func, [va_list])

        if node.ident.name == "__builtin_return_address":
            block, level = build_ir_expr(node.params[0], block, ctx)
            returnaddress_func = ctx.get_returnaddress_func(level)
            return block, CallInst(block, returnaddress_func.func_ty, returnaddress_func, [level])

        if node.ident.name == "__builtin_va_start":
            block, va_list = get_lvalue(node.params[0], block, ctx)
            va_list = GetElementPtrInst(
                block, va_list, va_list.ty, ConstantInt(0, i32), ConstantInt(0, i32))
            va_list = BitCastInst(block, va_list, PointerType(i8, 0))
            va_start_func = ctx.get_va_start_func(va_list)
            return block, CallInst(block, va_start_func.func_ty, va_start_func, [va_list])

        if node.ident.name == "__builtin_va_arg_char":
            block, va_list = get_lvalue(node.params[0], block, ctx)
            return block, VAArgInst(block, va_list, PrimitiveType("i8"))

        if node.ident.name == "__builtin_va_arg_int":
            block, va_list = get_lvalue(node.params[0], block, ctx)
            return block, VAArgInst(block, va_list, PrimitiveType("i32"))

        if node.ident.name == "__builtin_va_arg_uint":
            block, va_list = get_lvalue(node.params[0], block, ctx)
            return block, VAArgInst(block, va_list, PrimitiveType("i32"))

        if node.ident.name == "__builtin_va_arg_ulong":
            block, va_list = get_lvalue(node.params[0], block, ctx)
            return block, VAArgInst(block, va_list, PrimitiveType("i64"))

        if node.ident.name == "__builtin_va_arg_ptr":
            block, va_list = get_lvalue(node.params[0], block, ctx)
            return block, VAArgInst(block, va_list, PointerType(PrimitiveType("i8"), 0))

        no_inline = True

        if not no_inline and node.ident in ctx.defined_funcs:
            callee_func = ctx.defined_funcs[node.ident]

            if callee_func.proto.specs.func_spec_inline and callee_func.stmts:
                arg_values = []
                for arg in node.params:
                    if isinstance(get_type_of(arg), ast.types.ArrayType):
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

    if isinstance(node.ident, FunctionSymbol):
        func_ty = node.ident.ty.elem_ty
    elif isinstance(node.ident, VariableSymbol):
        func_ty = node.ident.ty.elem_ty
    else:
        func_ty = node.ident.type.elem_ty

    func_info = compute_func_info(ctx, func_ty)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        mem = ctx.named_values[node]
        params.append(mem)

        return_mem = mem

    for i, param in enumerate(node.params):
        param_ty = ctx.get_ir_type(param.type)

        if isinstance(param_ty, ArrayType):
            param_ty = PointerType(param_ty.elem_ty, 0)

        if i < len(func_info.arguments):
            arg_info = func_info.arguments[i]
            ty = arg_info.ty
            kind = arg_info.kind
        else:
            assert(func_ty.is_variadic)
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

                if isinstance(src_type, FunctionType) and isinstance(dst_type, PointerType):
                    block, val = get_lvalue(param, block, ctx)
                    print("")

                src_size = ctx.module.data_layout.get_type_alloc_size(src_type)
                dst_size = ctx.module.data_layout.get_type_alloc_size(dst_type)

                if src_type in [i1, i8, i16, i32, i64] and dst_type in [i1, i8, i16, i32, i64]:
                    if src_size <= dst_size:
                        block, val = build_ir_expr(param, block, ctx)
                        val = ZExtInst(block, val, dst_type)
                    else:
                        block, val = build_ir_expr(param, block, ctx)
                        val = TruncInst(block, val, dst_type)
                elif src_type in [f16, f32, f64, f128] and dst_type in [f16, f32, f64, f128]:
                    if src_size <= dst_size:
                        block, val = build_ir_expr(param, block, ctx)
                        val = FPExtInst(block, val, dst_type)
                    else:
                        block, val = build_ir_expr(param, block, ctx)
                        val = FPTruncInst(block, val, dst_type)
                elif src_type in [i1, i8, i16, i32, i64] and isinstance(dst_type, PointerType):
                    block, val = build_ir_expr(param, block, ctx)
                    val = IntToPtrInst(block, val, dst_type)
                elif isinstance(src_type, PointerType) and isinstance(dst_type, PointerType):
                    block, val = build_ir_expr(param, block, ctx)
                    val = BitCastInst(block, val, dst_type)
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

    if isinstance(node.ident, FunctionSymbol):
        callee = ctx.funcs[node.ident.name]
    elif isinstance(node.ident, VariableSymbol):
        callee = ctx.named_values[node.ident]
        callee = LoadInst(block, callee)
    else:
        block, callee = build_ir_expr(node.ident, block, ctx)

    call_inst = CallInst(block, ctx.get_ir_type(func_ty), callee, params)

    if func_info.return_info.kind == ABIArgKind.Indirect:
        value = LoadInst(block, return_mem)
        return block, value

    return block, call_inst


def build_ir_expr_int_const(node, block, ctx):
    return block, ConstantInt(node.val, ctx.get_ir_type(node.type))


def build_ir_expr_float_const(node, block, ctx):
    return block, ConstantFP(node.val, ctx.get_ir_type(node.type))


def build_ir_expr_string_const(node, block, ctx):
    block, value = get_lvalue(node, block, ctx)
    return block, GetElementPtrInst(block, value, value.ty, ConstantInt(0, i32), ConstantInt(0, i32))


def build_ir_expr_accessor(node, block, ctx):
    assert isinstance(node, (TypedAccessorOp,))

    block, lval = get_lvalue(node, block, ctx)
    if isinstance(lval.ty.elem_ty, ArrayType):
        return block, GetElementPtrInst(block, lval, lval.ty, ConstantInt(0, i32), ConstantInt(0, i32))
    return block, LoadInst(block, lval)


def build_ir_expr_indexer(node, block, ctx):
    assert isinstance(node, (TypedArrayIndexerOp,))

    block, lval = get_lvalue(node, block, ctx)

    is_volatile = False
    quals = get_qualifier(node)
    if quals & ast.types.Qualifier.Volatile:
        is_volatile = True

    return block, LoadInst(block, lval, is_volatile)


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
        for designator, expr in node.exprs:
            block, expr = build_ir_expr(expr, block, ctx)
            exprs.append(expr)
        return block, exprs

    if isinstance(node, TypedCastExpr):
        result_ty = ctx.get_ir_type(node.type)

        if isinstance(get_type_of(node.expr), ast.types.ArrayType) and isinstance(get_type_of(node), ast.types.PointerType):
            block, val = get_lvalue(node.expr, block, ctx)
            val = GetElementPtrInst(
                block, val, val.ty, ConstantInt(0, i32), ConstantInt(0, i32))
            return block, BitCastInst(block, val, result_ty)

        block, value = build_ir_expr(node.expr, block, ctx)
        if isinstance(node.expr, TypedInitializerList):
            return block, value

        return block, cast(block, value, result_ty, is_signed_type(node.expr.type), ctx)

    print(node)
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
        if isinstance(cond.ty, PointerType):
            cond = PtrToIntInst(block, cond, i32)
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

    block_cond_entry = block_cond
    block_cond, cond = build_ir_expr(node.cond, block_cond, ctx)

    if cond.ty != i1:
        if isinstance(cond.ty, PointerType):
            cond = PtrToIntInst(block_cond, cond, i32)
        cond = CmpInst(block_cond, "ne", cond, ConstantInt(0, cond.ty))

    BranchInst(block_cond, cond, block_then, block_cont)

    ctx.push_break_target(block_cont)
    ctx.push_continue_target(block_cond_entry)

    block_then = build_ir_stmt(node.stmt, block_then, ctx)
    JumpInst(block_then, block_cond_entry)

    ctx.pop_break_target()
    ctx.pop_continue_target()

    return block_cont


def build_ir_do_while_stmt(node, block, ctx):
    # append 3 blocks
    block_iter = BasicBlock(block.func, block)
    block_iter_entry = block_iter
    block_cont = BasicBlock(block.func, block_iter)

    JumpInst(block, block_iter_entry)

    ctx.push_break_target(block_cont)
    ctx.push_continue_target(block_iter_entry)

    block_iter = build_ir_stmt(node.stmt, block_iter, ctx)

    ctx.pop_break_target()
    ctx.pop_continue_target()

    block_iter, cond = build_ir_expr(node.cond, block_iter, ctx)

    if cond.ty != i1:
        if isinstance(cond.ty, PointerType):
            cond = CmpInst(block_iter, "ne", cond,
                           ConstantPointerNull(cond.ty))
        else:
            cond = CmpInst(block_iter, "ne", cond, ConstantInt(0, cond.ty))

    BranchInst(block_iter, cond, block_iter_entry, block_cont)

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
    block_loop = BasicBlock(block.func, block_then)
    block_cont = BasicBlock(block.func, block_loop)

    JumpInst(block, block_cond)

    block_cond_entry = block_cond
    block_loop_entry = block_loop

    if node.cond:
        block_cond, cond = build_ir_expr(node.cond, block_cond, ctx)

        if cond.ty != i1:
            if isinstance(cond.ty, PointerType):
                cond = CmpInst(block_cond, "ne", cond,
                               ConstantPointerNull(cond.ty))
            else:
                cond = CmpInst(block_cond, "ne", cond, ConstantInt(0, cond.ty))

        BranchInst(block_cond, cond, block_then, block_cont)
    else:
        JumpInst(block_cond, block_then)

    ctx.push_break_target(block_cont)
    ctx.push_continue_target(block_loop_entry)

    block_then = build_ir_stmt(node.stmt, block_then, ctx)

    ctx.pop_break_target()
    ctx.pop_continue_target()

    block_loop = build_ir_stmt(ExprStmt(node.loop), block_loop, ctx)

    JumpInst(block_then, block_loop_entry)
    JumpInst(block_loop, block_cond_entry)

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


def build_ir_goto_stmt(node, block, ctx):
    assert(isinstance(node.label, Ident))
    ctx.query_goto_jump(block, node.label.val)

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
        CallInst(block, memcpy.func_ty, memcpy, [lhs, rhs, ConstantInt(
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
            if lhs.ty.elem_ty != rhs.ty:
                rhs = cast(block, rhs, lhs.ty.elem_ty,
                           rhs.ty != i1 and is_signed_type(node.expr.type), ctx)

        StoreInst(block, rhs, lhs)
        JumpInst(block, ctx.get_return_target())

    block = BasicBlock(block.func, block)
    return block


def build_ir_switch_stmt(node, block, ctx):
    assert(node.stmts)
    stmts = node.stmts.stmts

    blocks = {}
    cur_block = block
    default_block = None
    for stmt in stmts:
        if not isinstance(stmt, CaseLabelStmt):
            continue

        if not stmt.expr:
            continue

        cur_block = BasicBlock(block.func, cur_block)
        blocks[stmt] = cur_block

    cur_block = BasicBlock(block.func, cur_block)
    default_block = cur_block

    cont_block = BasicBlock(block.func, cur_block)

    block, cond = build_ir_expr(node.cond, block, ctx)
    if cond.ty in [i1, i8, i16]:
        cond = cast(block, cond, i32, is_signed_type(node.cond.type), ctx)
    assert(cond.ty in [i32, i64])

    label_ty = cond.ty

    cases = []
    for case_label, case_block in blocks.items():
        if not case_label.expr:
            continue

        exprs = []
        while isinstance(case_label, CaseLabelStmt):
            exprs.append(case_label.expr)

            case_label = case_label.stmt

        for expr in exprs:
            label = evaluate_constant_expr(ctx, expr)
            if label.ty in [i1, i8, i16, i32, i64] and label.ty != label_ty:
                label = ConstantInt(label.value, label_ty)
            assert(label.ty == label_ty)

            if label.ty in [i1, i8, i16]:
                label = ConstantInt(label.value, label_ty)

            cases.append(label)
            cases.append(case_block)

    SwitchInst(block, cond, default_block, cases)

    cur_label = None
    for stmt in stmts:
        if isinstance(stmt, CaseLabelStmt):
            cur_label = stmt

        if not cur_label.expr:
            ctx.push_break_target(cont_block)

            default_block = build_ir_stmt(
                stmt, default_block, ctx)

            ctx.pop_break_target()
            continue

        ctx.push_break_target(cont_block)

        blocks[cur_label] = build_ir_stmt(
            stmt, blocks[cur_label], ctx)

        ctx.pop_break_target()

    prev_block = None
    for i, case_label in enumerate(stmts):
        if not isinstance(case_label, CaseLabelStmt):
            continue

        if not case_label.expr:
            continue

        cur_block = blocks[case_label]

        if prev_block:
            JumpInst(prev_block, cur_block)
        prev_block = cur_block

    if prev_block:
        JumpInst(prev_block, default_block)
    prev_block = default_block

    if prev_block:
        JumpInst(prev_block, cont_block)

    return cont_block


def build_ir_asm_stmt(node, block, ctx):
    output_operands = node.operands[0]
    input_operands = node.operands[1]
    cobbers = node.operands[2]

    result_elems = []
    operands = []

    for constraint, name in output_operands:
        if not name:
            continue

        if constraint.value.startswith("+"):
            block, var = get_lvalue(name, block, ctx)
            operands.append(var)
        else:
            block, var = get_lvalue(name, block, ctx)

            assert(isinstance(var.ty, PointerType))
            result_elems.append(var)

    for constraint, name in input_operands:
        block, var = build_ir_expr(name, block, ctx)

        operands.append(var)

    if len(result_elems) == 1:
        result_ty = result_elems[0].ty.elem_ty
    elif result_elems:
        result_ty = StructType(
            fields=[elem.ty.elem_ty for elem in result_elems])
    else:
        result_ty = void

    func_ty = FunctionType(result_ty, [operand.ty for operand in operands])

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
        elif constraint == "r":
            constraint = "r"
        elif constraint == "m":
            constraint = "m"
        else:
            raise ValueError("Invalid constraint")

        return modifier, constraint

    asm_string = node.template
    constraints = []
    for constraint, name in output_operands:
        if not name:
            continue

        constraint = constraint.value

        modifier, constraint = parse_constraint(constraint)

        if modifier == "+":
            modifier = "=*"

        constraints.append(f"{modifier}{constraint}")

    for constraint, name in input_operands:
        if not name:
            continue

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

    constraints.append("~{dirflag}")
    constraints.append("~{fpsr}")
    constraints.append("~{flags}")

    for i in range(10):
        asm_string = asm_string.replace(f"%{i}", f"${i}")
    asm_string = asm_string.replace("%%", "%")

    asm = InlineAsm(func_ty, asm_string, ",".join(constraints), True)

    asm_result = CallInst(block, asm.func_ty, asm, operands)

    if len(result_elems) == 1:
        result = result_elems[0]

        if result:
            StoreInst(block, asm_result, result)
    elif result_elems:
        for i, result in enumerate(result_elems):
            if not result:
                continue

            value = ExtractValueInst(block, asm_result, i)
            StoreInst(block, value, result)

    return block


def build_ir_stmt(node, block, ctx):
    # Allocate local variables.
    if isinstance(node, TypedVariable):
        for variable, _ in node.idents:
            size = ConstantInt(1, i32)
            ty = ctx.get_ir_type(variable.type)

            if "static" in node.storage_class or "extern" in node.storage_class:
                continue

            alloca_insert_pt = get_alloca_insert_pt(ctx)

            mem = alloca_insert_pt = AllocaInst(alloca_insert_pt, size, ty, 0)

            ctx.named_values[variable.val] = mem

    if isinstance(node, IfStmt):
        return build_ir_if_stmt(node, block, ctx)

    if isinstance(node, WhileStmt):
        return build_ir_while_stmt(node, block, ctx)

    if isinstance(node, DoWhileStmt):
        return build_ir_do_while_stmt(node, block, ctx)

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
        if "static" in node.storage_class:
            emit_global_variable(ctx.function.name, node, ctx)
            return block
        if "extern" in node.storage_class:
            emit_global_variable("", node, ctx)
            return block

        for variable, initializer in node.idents:
            if initializer is not None:
                block, init_expr = build_ir_expr(initializer, block, ctx)
                block, _ = build_ir_assign_op(variable, init_expr, block, ctx)

        return block

    if isinstance(node, SwitchStmt):
        return build_ir_switch_stmt(node, block, ctx)

    if isinstance(node, CaseLabelStmt):
        return build_ir_stmt(node.stmt, block, ctx)

    if isinstance(node, AsmStmt):
        return build_ir_asm_stmt(node, block, ctx)

    if isinstance(node, GotoStmt):
        return build_ir_goto_stmt(node, block, ctx)

    if isinstance(node, LabelStmt):
        cont_block = BasicBlock(block.func, block)
        JumpInst(block, cont_block)

        assert(isinstance(node.label, Ident))
        ctx.goto_targets[node.label.val] = cont_block

        remains = []
        for block_from, target_label in ctx.goto_jump_query:
            if node.label.val != target_label:
                remains.append((block_from, target_label))
                continue

            JumpInst(block_from, cont_block)

        ctx.goto_jump_query = remains

        cont_block = build_ir_stmt(node.stmt, cont_block, ctx)

        return cont_block

    return block


def build_ir_stack_alloc(node, block, ctx):
    alloca_insert_pt = get_alloca_insert_pt(ctx)

    # Allocate return value of all callee functions.
    from c.symtab import FunctionSymbol, VariableSymbol

    if isinstance(node, TypedFunctionCall):
        return_ty = ctx.get_ir_type(node.type)

        if isinstance(node.ident, FunctionSymbol):
            func_ty = node.ident.ty.elem_ty
        elif isinstance(node.ident, VariableSymbol):
            func_ty = node.ident.ty.elem_ty
        else:
            func_ty = node.ident.type.elem_ty

        func_info = compute_func_info(ctx, func_ty)
        if func_info.return_info.kind == ABIArgKind.Indirect:
            mem = alloca_insert_pt = AllocaInst(alloca_insert_pt,
                                                ConstantInt(1, i32), return_ty, 0)
            ctx.named_values[node] = mem


def build_ir_func_header(node, func, ctx):
    func_info = compute_func_info(ctx, node.ident.ty.elem_ty)

    return_ty = ctx.get_ir_type(node.type)

    func.args = []

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
    func_info = compute_func_info(ctx, node.proto.ident.ty.elem_ty)
    ctx.func_info = func_info
    alloca_insert_pt = get_alloca_insert_pt(ctx)

    return_ty = ctx.get_ir_type(node.proto.type)

    func = block.func
    func.args = []

    if func_info.return_info.kind == ABIArgKind.Indirect:
        ir_arg = Argument(PointerType(return_ty, 0))
        ir_arg.add_attribute(Attribute(AttributeKind.StructRet))
        ctx.return_value = ir_arg
        func.add_arg(ir_arg)

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
        func.add_arg(ir_arg)

    block_end = BasicBlock(func)

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

    assert(not ctx.goto_jump_query)
    ctx.goto_targets = {}

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

    @ property
    def is_direct(self):
        return self.kind == ABIArgKind.Direct

    @ property
    def is_indirect(self):
        return self.kind == ABIArgKind.Indirect

    @ property
    def is_extend(self):
        return self.kind == ABIArgKind.Extend

    @ property
    def is_coerce_and_expand(self):
        return self.kind == ABIArgKind.CoerceAndExpand

    @ property
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
    return ty in [f32, f64]


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

        if isinstance(ty, ast.types.PointerType):
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


class AArch64ABIInfo(ABIInfo):
    def __init__(self):
        super().__init__()

    def is_homogeneous_aggregate_small_enough(self, ty, members):
        return members <= 4

    def is_homogeneous_aggregate(self, ty):
        def is_homogeneous_aggregate_rec(ty, base, members):
            
            if isinstance(ty, ast.types.CompositeType):
                for field in ty.fields:
                    field_ty, _, _ = field

                    field_is_ha, base, field_members = is_homogeneous_aggregate_rec(field_ty, base, 0)

                    if not field_is_ha:
                        return False

                    if ty.is_union:
                        members = max(members, field_members)
                    else:
                        members += field_members

                if not base:
                    return False
            else:
                members = 1
                base = ty

            return members > 0 and self.is_homogeneous_aggregate_small_enough(base, members), base, members

        result, _, _ = is_homogeneous_aggregate_rec(ty, None, 0)
        return result

    def compute_arg_info(self, ctx, ty):
        if isinstance(ty, ast.types.VoidType):
            return ABIArgInfo(ABIArgKind.Ignore)

        if isinstance(ty, ast.types.PrimitiveType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.PointerType):
            return ABIArgInfo(ABIArgKind.Direct)

        if isinstance(ty, ast.types.VectorType):
            ir_ty = ctx.get_ir_type(ty)
            if is_x86_mmx_type(ir_ty):
                return ABIArgInfo(ABIArgKind.Direct, get_integer_type(64))

            return ABIArgInfo(ABIArgKind.Direct)

        if self.is_homogeneous_aggregate(ty):
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

        if isinstance(ty, ast.types.PointerType):
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

        if isinstance(ty, ast.types.PointerType):
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
    AArch64 = auto()
    RISCV32 = auto()
    RISCV64 = auto()


def emit_global_variable(prefix, decl, ctx):
    module = ctx.module

    for ident, init_expr in decl.idents:
        ty = ctx.get_ir_type(ident.type)
        if not init_expr:
            init = get_constant_null_value(ty)
        else:
            init = evaluate_constant_expr(ctx, init_expr)
            assert(ident.type == init_expr.type)

        linkage = GlobalLinkage.Global
        thread_local = ThreadLocalMode.NotThreadLocal

        if "extern" in decl.storage_class:
            linkage = GlobalLinkage.External
            init = None
        elif "static" in decl.storage_class:
            linkage = GlobalLinkage.Internal

        if "thread_local" in decl.storage_class:
            thread_local = ThreadLocalMode.LocalExecTLSModel

        name = ident.val.name
        if prefix:
            name = f"{prefix}.{name}"

        if init:
            ty = init.ty

        ctx.global_named_values[ident.val] = module.add_global(name,
                                                               GlobalVariable(ty, False, linkage, name, thread_local, init))
        ctx.named_values[ident.val] = ctx.global_named_values[ident.val]


def emit_ir(ast, abi, module):
    ctx = Context(module)
    if abi == ABIType.WinX86_64:
        ctx.abi_info = WinX86_64ABIInfo()
    elif abi == ABIType.X86_64:
        ctx.abi_info = X86_64ABIInfo()
    elif abi == ABIType.EABI:
        ctx.abi_info = EABIABIInfo()
    elif abi == ABIType.AArch64:
        ctx.abi_info = AArch64ABIInfo()
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
            emit_global_variable("", decl, ctx)

        if isinstance(decl, TypedFunctionProto):
            func_decl = decl.ident

            func_ty = ctx.get_ir_type(func_decl.ty.elem_ty)
            func_name = str(func_decl.name)

            linkage = GlobalLinkage.Global

            from c.parse import StorageClass

            if decl.specs.storage_class_spec == StorageClass.Extern:
                linkage = GlobalLinkage.External
            elif decl.specs.storage_class_spec == StorageClass.Static:
                linkage = GlobalLinkage.Internal

            func = module.add_func(
                func_name, Function(module, func_ty, linkage, func_name))

            func.attributes = set(
                [Attribute(AttributeKind.NoRedZone), Attribute(AttributeKind.NoImplicitFloat)])

            assert(isinstance(func_decl, FunctionSymbol))
            ctx.funcs[func_decl.name] = func
            ctx.global_named_values[func_decl] = func

            build_ir_func_header(decl, func, ctx)

        if isinstance(decl, TypedFunction):
            func_decl = decl.proto.ident

            func_ty = ctx.get_ir_type(func_decl.ty.elem_ty)
            func_name = str(func_decl.name)

            linkage = GlobalLinkage.Global

            from c.parse import StorageClass

            if decl.proto.specs.storage_class_spec == StorageClass.Extern:
                linkage = GlobalLinkage.External
            elif decl.proto.specs.storage_class_spec == StorageClass.Static:
                linkage = GlobalLinkage.Internal

            func = module.add_func(
                func_name, Function(module, func_ty, linkage, func_name))

            func.attributes = set(
                [Attribute(AttributeKind.NoRedZone), Attribute(AttributeKind.NoImplicitFloat)])

            assert(isinstance(func_decl, FunctionSymbol))
            ctx.funcs[func_decl.name] = func
            ctx.global_named_values[func_decl] = func

            no_common = False

            if not no_common and decl.proto.specs.func_spec_inline:
                comdat = module.add_comdat(func_name, ComdatKind.Any)
                func.comdat = comdat

            if decl.stmts:
                ctx.begin_func()
                ctx.ast_func = decl
                ctx.function = func
                ctx.current_block = BasicBlock(func)

                ctx.named_values.update(global_named_values)
                block = ctx.current_block
                build_ir_func(decl, block, ctx)

                ctx.end_func()
            else:
                if func_decl.name not in ctx.funcs:
                    build_ir_func_header(decl.proto, func, ctx)

                ctx.funcs[func_decl.name] = func

    return module
