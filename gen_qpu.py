#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid

import re
import sys

from ast.node import *
from glsl.symtab import PrimitiveType, CompositeType, Symbol
from ast.printer import print_ast_expr
from ir.values import PropPredInst, InvPredInst, RestorePredInst, ReturnPredInst, PushPredInst, BranchPredInst
from ir.values import JumpInst, LoadInst, BinaryInst, ReturnInst

arith_insts_f = {
    '+': 'fadd',
    '-': 'fsub',
    '*': 'fmul',
}

arith_insts_i = {
    '+': 'iadd',
    '-': 'isub',
    '*': 'imul24',
    '^': 'bxor',
    '|': 'bor',
    '&': 'band',
    '>>': 'asr',
    '<<': 'shl',
}

arith_insts_ui = {
    '+': 'iadd',
    '-': 'isub',
    '*': 'imul24',
    '^': 'bxor',
    '|': 'bor',
    '&': 'band',
    '>>': 'shr',
    '<<': 'shl',
}


def emit_arith(inst, rd, rs, rt, env, output, cond='zc', sf=0):
    if isinstance(rd, list):
        assert(len(rd) == 1)
        rd = rd[0]
    if isinstance(rs, list):
        assert(len(rs) == 1)
        rs = rs[0]
    if isinstance(rt, list):
        assert(len(rt) == 1)
        rt = rt[0]

    output.append(f"{inst}({rd}, {rs}, {rt}, cond='{cond}', set_flags={sf})")


def emit_mov(rd, rs, env, output, cond='zc', sf=0):
    if isinstance(rd, list):
        assert(len(rd) == 1)
        rd = rd[0]
    if isinstance(rs, list):
        assert(len(rs) == 1)
        rs = rs[0]

    output.append(f"mov({rd}, {rs}, cond='{cond}', set_flags={sf})")


def emit_ldi(rd, imm, env, output, cond='zc', sf=0):
    if isinstance(rd, list):
        assert(len(rd) == 1)
        rd = rd[0]

    output.append(f"ldi({rd}, {imm}, cond='{cond}', set_flags={sf})")


def emit_nop(env, output):
    output.append('nop()')


def emit_clz(rd, rs, env, output, cond='zc', sf=0):
    output.append(f"clz({rd}, {rs}, cond='{cond}', set_flags={sf})")

# Pseudocodes for comparation.
# Use r1 for a working register.


def emit_eq(rd, rs, rt, env, output):
    emit_arith('bxor', 'r1', rs, rt, env, output)
    # (clz(r1) == 32) <=> (r1 == 0)
    emit_clz('r1', 'r1', env, output)
    emit_arith('shr', rd, 'r1', 5, env, output)


def emit_ne(rd, rs, rt, env, output):
    emit_eq('r1', rs, rt, env, output)
    emit_arith('bxor', rd, 'r1', 1, env, output)


def emit_lte(rd, rs, rt, env, output):
    # (rs <= rt) <=> (r1 = rs - rt <= 0)
    emit_arith('isub', 'r1', rs, rt, env, output)
    # clz(r1) == 0 or 32 <=> (r1 <= 0)
    emit_clz('r1', 'r1', env, output)
    # (1 >> r1) <=> (1 if r1 == 0 || r1 == 32 else 0)
    emit_arith('shr', rd, 1, 'r1', env, output)


def emit_gte(rd, rs, rt, env, output):
    emit_lte(rd, rt, rs, env, output)


def emit_lt(rd, rs, rt, env, output):
    # (rs < rt) <=> (rs - rt < 0) <=> not(rs - rt >= 0)
    emit_gte('r1', rs, rt, env, output)
    emit_arith('bxor', rd, 'r1', 1, env, output)


def emit_gt(rd, rs, rt, env, output):
    emit_lt(rd, rt, rs, env, output)


def emit_setp(rd, rs, env, output):
    # rs > 0 <=> clz(rs) != 32 <=> r1 == 0
    emit_clz('r1', rs, env, output)
    emit_arith('shr', 'r1', 'r1', 5, env, output)
    # (1 if r1 == 0 else 0)
    emit_arith('bxor', rd, 'r1', 1, env, output)


def emit_not(rd, rs, env, output):
    # rs > 0 <=> clz(rs) != 32 <=> r1 == 0
    emit_clz('r1', rs, env, output)
    emit_arith('shr', rd, 'r1', 5, env, output)


def emit_bin_op_int(node, env, output, unsigned):
    insts = arith_insts_ui if unsigned else arith_insts_i
    if node.op in insts:
        inst = insts[node.op]

        # compute lhs
        emit_expr(node.lhs, env, output)

        # alloc mem
        reg_mem = env.allocate_register_annonymous(node.lhs.type)

        # push r0
        emit_mov(reg_mem, 'r0', env, output)

        # compute rhs
        emit_expr(node.rhs, env, output)

        # compute binary operation
        emit_arith(inst, 'r0', reg_mem, 'r0', env, output)

        # free reg
        env.free_register(reg_mem)

        return ['r0']

    if node.op in ["==", "!=", "<", ">", "<=", ">="]:

        # compute lhs
        reg_lhs = emit_expr(node.lhs, env, output)

        # alloc mem
        reg_mem = env.allocate_register_annonymous(node.lhs.type)

        # push r0
        emit_mov(reg_mem, reg_lhs, env, output)

        # compute rhs
        reg_rhs = emit_expr(node.rhs, env, output)

        assert(reg_rhs == 'r0')

        if node.op == "==":
            emit_eq(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == "!=":
            emit_ne(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == "<":
            emit_lt(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == ">":
            emit_gt(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == "<=":
            emit_lte(reg_rhs, reg_mem, reg_rhs, env, output)
        else:  # node.op == ">=":
            emit_gte(reg_rhs, reg_mem, reg_rhs, env, output)

        # free reg
        env.free_register(reg_mem)

        return ['r0']

    print(node)
    raise Exception


def emit_bin_op_float(node, env, output):
    if node.op in arith_insts_f:
        inst = arith_insts_f[node.op]

        # compute lhs
        emit_expr(node.lhs, env, output)

        # alloc reg
        reg_mem = env.allocate_register_annonymous(node.lhs.type)

        # push r0
        emit_mov(reg_mem, 'r0', env, output)

        # compute rhs
        emit_expr(node.rhs, env, output)

        # compute binary operation
        emit_arith(inst, 'r0', reg_mem, 'r0', env, output)

        # free reg
        env.free_register(reg_mem)

        return ['r0']

    raise Exception


def emit_assign_op(node, env, output):
    if node.op == '=':
        if isinstance(node.lhs, TypedIdentExpr):
            reg_lhs = env.get_register(node.lhs.val, node.lhs.type)
            reg_rhs = emit_expr(node.rhs, env, output)

            assert(len(reg_lhs) == len(reg_rhs))

            for a, b in zip(reg_lhs, reg_rhs):
                emit_mov(a, b, env, output)
            emit_nop(env, output)
            return reg_lhs
        elif isinstance(node.lhs, TypedAccessorOp):
            if not isinstance(node.lhs.obj, TypedIdentExpr):
                raise NotImplementedError()

            reg_lhs = env.get_elem_register(
                node.lhs.obj.val, node.lhs.obj.type, node.lhs.field.val)
            reg_rhs = emit_expr(node.rhs, env, output)

            assert(len(reg_lhs) == len(reg_rhs))

            for a, b in zip(reg_lhs, reg_rhs):
                emit_mov(a, b, env, output)
            emit_nop(env, output)
            return reg_lhs

    raise Exception


def emit_compare_op(node, env, output):
    if node.op in ['==', '!=', '<', '>', '<=', '>=']:
        # compute lhs
        reg_lhs = emit_expr(node.lhs, env, output)

        # alloc mem
        reg_mem = env.allocate_register_annonymous(node.lhs.type)

        # push r0
        emit_mov(reg_mem, reg_lhs, env, output)

        # compute rhs
        reg_rhs = emit_expr(node.rhs, env, output)

        if node.op == "==":
            emit_eq(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == "!=":
            emit_ne(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == "<":
            emit_lt(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == ">":
            emit_gt(reg_rhs, reg_mem, reg_rhs, env, output)
        elif node.op == "<=":
            emit_lte(reg_rhs, reg_mem, reg_rhs, env, output)
        else:  # node.op == ">=":
            emit_gte(reg_rhs, reg_mem, reg_rhs, env, output)

        # free reg
        env.free_register(reg_mem)

        return ['r0']


def emit_bin_op(node, env, output):
    if node.op == '=':
        return emit_assign_op(node, env, output)
    if node.op in ['==', '!=', '<', '>', '<=', '>=']:
        return emit_compare_op(node, env, output)

    typename = node.type.name
    if typename == 'int':
        return emit_bin_op_int(node, env, output, False)
    if typename == 'uint':
        return emit_bin_op_int(node, env, output, True)
    elif typename == 'vec3':
        return emit_bin_op_float(node, env, output)
    elif typename == 'float':
        return emit_bin_op_float(node, env, output)

    print(node)
    raise Exception


def emit_ident(node, env, output):
    if node.type.name in buildin_type_reg_size:
        reg = env.get_register(node.val, node.type)
        emit_mov('r0', reg, env, output)
        return ['r0']
    else:
        ld_regs = env.get_register(node.val, node.type)
        st_regs = env.allocate_register_annonymous(node.type)
        for a, b in zip(st_regs, ld_regs):
            emit_mov(a, b, env, output)
        return st_regs


def emit_lit(node, env, output):
    emit_ldi('r0', node.val, env, output)
    return ['r0']


def emit_unary(node, env, output):
    if isinstance(node.expr, TypedIdentExpr):
        reg = env.get_register(node.expr.val)

        if node.type.specifier in ['int', 'uint']:
            if node.op == '++':
                emit_mov('r0', reg, env, output)
                emit_arith('iadd', 'r0', 'r0', 1, env, output)
                emit_mov(reg, 'r0', env, output)
                emit_nop(env, output)
                return ['r0']
            if node.op == '--':
                emit_mov('r0', reg, env, output)
                emit_arith('isub', 'r0', 'r0', 1, env, output)
                emit_mov(reg, 'r0', env, output)
                emit_nop(env, output)
                return ['r0']
        if node.type.specifier == 'vec3':
            if node.op == '++':
                emit_mov('r0', reg, env, output)
                emit_arith('fadd', 'r0', 'r0', 1, env, output)
                emit_mov(reg, 'r0', env, output)
                emit_nop(env, output)
                return ['r0']
            if node.op == '--':
                emit_mov('r0', reg, env, output)
                emit_arith('fsub', 'r0', 'r0', 1, env, output)
                emit_mov(reg, 'r0', env, output)
                emit_nop(env, output)
                return ['r0']
    elif isinstance(node, TypedUnaryOp):
        emit_expr(node.expr, env, output)
        emit_not('r0', 'r0', env, output)
        return ['r0']

    raise Exception


def emit_post(node, env, output):
    if isinstance(node.expr, TypedIdentExpr):
        reg = env.get_register(node.expr.val, node.expr.type)
    else:
        raise Exception

    if node.type.name in ['int', 'uint']:
        if node.op == '++':
            emit_mov('r0', reg, env, output)
            emit_arith('iadd', reg, reg, 1, env, output)
            emit_nop(env, output)
            return ['r0']
        if node.op == '--':
            emit_mov('r0', reg, env, output)
            emit_arith('isub', reg, reg, 1, env, output)
            emit_nop(env, output)
            return ['r0']
    if node.type.name == 'vec3':
        if node.op == '++':
            emit_mov('r0', reg, env, output)
            emit_arith('fadd', reg, reg, 1, env, output)
            emit_nop(env, output)
            return ['r0']
        if node.op == '--':
            emit_mov('r0', reg, env, output)
            emit_arith('fsub', reg, reg, 1, env, output)
            emit_nop(env, output)
            return ['r0']

    raise Exception


def emit_accessor(node, env, output):
    reg = env.get_elem_register(
        node.obj.val, node.obj.type, node.field.val)
    emit_mov('r0', reg, env, output)
    return ['r0']


def emit_expr(node, env, output):
    if isinstance(node, TypedBinaryOp):
        return emit_bin_op(node, env, output)

    if isinstance(node, TypedIdentExpr):
        return emit_ident(node, env, output)

    if isinstance(node, IntegerConstantExpr):
        return emit_lit(node, env, output)

    if isinstance(node, TypedUnaryOp):
        return emit_unary(node, env, output)

    if isinstance(node, TypedPostOp):
        return emit_post(node, env, output)

    if isinstance(node, TypedFunctionCall):
        return emit_func_call(node, env, output)

    if isinstance(node, TypedAccessorOp):
        return emit_accessor(node, env, output)

    assert False, f"Unknown node type {node}"


pred_type = PrimitiveType('int', None)


def is_empty_stmt(node):
    if node is None:
        return True
    if isinstance(node, CompoundStmt) and len(node.stmts) > 0:
        return None

    return False


def emit_label(label, env, output):
    output.append(f"L.{label}")


indent = '    '


def emit_floatBitsToUInt(node, env, output):
    assert(len(node.params) == 1)
    param = node.params[0]

    reg_rhs = emit_expr(param, env, output)
    emit_mov('r0', reg_rhs, env, output)

    return ['r0']


def emit_floatBitsToInt(node, env, output):
    assert(len(node.params) == 1)
    param = node.params[0]

    reg_rhs = emit_expr(param, env, output)
    emit_mov('r0', reg_rhs, env, output)

    return ['r0']


buildin_functions = {
    'floatBitsToUInt': emit_floatBitsToUInt,
    'floatBitsToInt': emit_floatBitsToInt,
}


def emit_func_call(node, env, output):
    func_node = None
    for func in env.funcs:
        if node.ident.val == func.proto.ident.val:
            func_node = func

    if func_node is None:
        if node.ident.val in buildin_functions:
            return buildin_functions[node.ident.val](node, env, output)

        raise Exception(
            f"The function \"{node.ident.val}\" isn't found in the current scope")

    reg = env.allocate_register_annonymous(pred_type)[0]  # save pred
    env.conds.append(reg)

    output.append(f"mov({reg}, r3, set_flags=1)")

    return_reg = env.allocate_register_annonymous(node.type)
    env.return_regs.append(return_reg)

    # evaluate parameters
    assert(len(func_node.proto.params) == len(node.params))
    for param, arg in zip(func_node.proto.params, node.params):
        reg_lhs = env.allocate_register(Symbol(
            param.ident.val, arg.type, [], node.definition.scope), arg.type)
        reg_rhs = emit_expr(arg, env, output)
        emit_mov(reg_lhs, reg_rhs, env, output)

    env.push_scope()

    output.append('# {0}'.format(func_node.proto.ident.val))
    # for stmt in func_node.stmts:
    #     emit_stmt(stmt, env, output)
    emit_ir_func(func_node, env, output, True)
    output.append('# end {0}'.format(func_node.proto.ident.val))

    env.pop_scope()

    output.append(f"mov(r3, {reg}, set_flags=1)")
    env.free_register(reg)
    env.conds.pop()

    return return_reg


buildin_funcs = [
    "radians",
    "degrees",
    "sin",
    "cos",
    "asin",
    "acos",

    "pow",
    "exp",
    "log",
    "exp2",
    "log2",
    "sqrt",
    "inversesqrt",

    "abs",
    "sign",
    "floor",
    "trunc",
    "round",
    "ceil",
    "mod",
    "min",
    "max",
    "clamp",

    "length",
    "dot",
    "normalize",
    "cross",
]


def is_buildin_func(ident):
    return ident in buildin_funcs


buildin_type_reg_size = {
    'int': 1,
    'uint': 1,
    'float': 1,
}

# A register has 4 * 16 * 8 bits.
# Use as 128 bits * 4 vectors


class Env:
    def __init__(self):
        self.stack = [{}]
        self.free = list(reversed([f"rb{i}" for i in range(
            0, 32)])) + list(reversed([f"ra{i}" for i in range(0, 32)]))

        self.inputs = []
        self.inputs_idx = {}
        self.inputs_consumed = 0

        self.outputs = []
        self.outputs_idx = {}
        self.outputs_consumed = 0

        self.conds = []
        self.return_regs = []

        self.pred_map = {}

        self.labels = {}

    def push_scope(self):
        self.stack.append({})

    def pop_scope(self):
        dealloc = self.stack.pop()
        for ident, (regs, ty) in dealloc.items():
            for reg in regs:
                self.free.append(reg)

    def allocate_register(self, var, ty):
        assert(isinstance(var, (Symbol)))
        assert(isinstance(ty, (PrimitiveType, CompositeType)))

        static = False
        if 'in' in var.ty_qual:
            self.allocate_input(var, ty)
            static = True
        if 'out' in var.ty_qual:
            self.allocate_output(var, ty)
            static = True

        size = self.compute_register_size(ty)

        eytries = self.stack[0] if static else self.stack[-1]
        free = self.free

        if var not in eytries:
            regs = []
            for _ in range(size):
                assert(len(free) > 0)
                reg = free.pop()
                regs.append(reg)
            eytries[var] = (regs, ty)
            return regs

        raise Exception(
            "{0} is already exisiting in the scope.".format(var.name))

    def allocate_register_annonymous(self, ty):
        return self.allocate_register(Symbol(uuid.uuid4().hex, ty, [], None), ty)

    def free_register(self, regs):
        stack = self.stack
        free = self.free

        keys = list(stack[-1].keys())
        values = [regs for regs, ty in stack[-1].values()]

        if regs in values:
            ident = keys[values.index(regs)]
            stack[-1].pop(ident)
            for reg in regs:
                free.append(reg)

        if [regs] in values:
            regs = [regs]
            ident = keys[values.index(regs)]
            stack[-1].pop(ident)
            for reg in regs:
                free.append(reg)

    def get_register(self, ident, ty):
        assert(isinstance(ident, (Symbol)))
        stack = self.stack

        for entries in reversed(stack):
            if ident in entries:
                regs, ty = entries[ident]
                return regs

        return self.allocate_register(ident, ty)

    def compute_register_size(self, ty):
        if isinstance(ty, PrimitiveType):
            return buildin_type_reg_size[ty.name]
        elif isinstance(ty, CompositeType):
            size = 0
            for field_name, field_type in ty.fields.items():
                field_ty, arr = field_type
                if arr is not None:
                    raise NotImplementedError
                size += self.compute_register_size(field_ty)
            return size

        raise NotImplementedError

    def compute_register_offset(self, ty, field):
        if isinstance(ty, CompositeType):
            size = 0
            for field_name, field_type in ty.fields.items():
                field_ty, arr = field_type
                if arr is not None:
                    raise NotImplementedError
                field_size = self.compute_register_size(field_ty)

                if field_name == field:
                    return (size, field_size)

                size += field_size

    def get_elem_register(self, ident, ty, field):
        assert(isinstance(ident, Symbol))
        stack = self.stack

        for entries in reversed(stack):
            if ident in entries:
                regs, ty = entries[ident]
                offset, size = self.compute_register_offset(ty, field)
                return regs[offset:offset+size]

        self.allocate_register(ident, ty)
        return self.get_elem_register(ident, ty, field)

    def allocate_input(self, ident, ty):
        if ident not in self.inputs_idx:
            print(f"input {ident.name}")
            self.inputs_idx[ident] = len(self.inputs)
            self.inputs.append((ident, ty))

    def allocate_output(self, ident, ty):
        if ident not in self.outputs_idx:
            print(f"output {ident.name}")
            self.outputs_idx[ident] = len(self.outputs)
            self.outputs.append((ident, ty))

    def push_cond_scope(self, reg):
        self.conds.append(reg)

    def pop_cond_scope(self):
        self.conds.pop()

    def get_cond_depth(self):
        return len(self.conds)


def emit_inputs_load(env, output):
    mov_code = []
    for var, ty in env.inputs:
        regs = env.get_register(var, var)
        for reg in regs:
            mov_code.append(f'mov({reg}, vpm)')

    dma_code = [
        'setup_dma_load(nrows={0})'.format(len(mov_code)),
        'start_dma_load(uniform)',
        'wait_dma_load()',
    ]

    output.extend(dma_code)
    output.append('setup_vpm_read(nrows={0})'.format(len(mov_code)))
    output.extend(mov_code)


def emit_outputs_store(env, output):
    output.append('setup_vpm_write()')

    mov_code = []
    for var, ty in env.outputs:
        regs = env.get_register(var, ty)
        for reg in regs:
            mov_code.append(f'mov(vpm, {reg})')

    output.extend(mov_code)

    dma_code = [
        'setup_dma_store(nrows={0})'.format(len(mov_code)),
        'start_dma_store(uniform)',
        'wait_dma_store()',
    ]

    output.extend(dma_code)


def emit_ir_prop_pred_inst(inst, env, output):
    reg = env.allocate_register_annonymous(pred_type)[0]

    env.pred_map[inst] = reg

    # execute cond
    output.append(f"# if ({print_ast_expr(inst.cond)}) {{")
    emit_expr(inst.cond, env, output)

    # condition push
    emit_setp('r0', 'r0', env, output)
    output.append(f"mov({reg}, r3, set_flags=0)")
    output.append(f"band(r3, r3, r0, set_flags=1)")


def emit_ir_return_pred_inst(inst, env, output):
    if inst.expr is not None:
        ld_regs = emit_expr(inst.expr, env, output)
        st_regs = env.return_regs[-1]
        for a, b in zip(st_regs, ld_regs):
            emit_mov(a, b, env, output)

    pred = inst.pred
    while pred is not None:
        reg = env.pred_map[pred]
        output.append(f"mov({reg}, 0, cond='zc', set_flags=0)")
        pred = pred.pred
    output.append(f"mov(r3, 0, cond='zc', set_flags=1)")


def emit_ir_inv_pred_inst(inst, env, output):
    output.append("# } else {")
    # condition flip
    reg = env.pred_map[inst.pred]
    output.append(f"bxor(r3, {reg}, r3, set_flags=1)")


def emit_ir_restore_pred_inst(inst, env, output):
    output.append("# }")

    # condition pop
    reg = env.pred_map[inst.pred]
    output.append(f"mov(r3, {reg}, set_flags=1)")


def emit_ir_push_pred_inst(inst, env, output):
    reg = env.allocate_register_annonymous(pred_type)[0]  # for predication

    env.pred_map[inst] = reg

    # condition push
    output.append(f"mov({reg}, r3, set_flags=1)")


def emit_ir_branch_pred_inst(inst, env, output):
    output.append(
        f"# if ({print_ast_expr(inst.cond)}) goto L.{env.labels[inst.then_target]}")
    emit_expr(inst.cond, env, output)

    # condition set
    emit_setp('r0', 'r0', env, output)
    output.append(f"band(r3, r3, r0, set_flags=1)")

    output.append(f"jzc_any(L.{env.labels[inst.then_target]})")
    emit_nop(env, output)
    emit_nop(env, output)
    emit_nop(env, output)

    output.append(f"jzs(L.{env.labels[inst.else_target]})")
    emit_nop(env, output)
    emit_nop(env, output)
    emit_nop(env, output)


def emit_ir_jump_inst(inst, env, output):
    if inst.goto_target != inst.block.next_block:
        output.append(f"jmp(L.{env.labels[inst.goto_target]})")
        emit_nop(env, output)
        emit_nop(env, output)
        emit_nop(env, output)


def get_rhs_reg(inst, env, output):
    if isinstance(inst.rs, (TypedIdentExpr,)):
        return env.get_register(inst.rs.val, inst.rs.type)
    elif isinstance(inst.rs, (TypedAccessorOp)):
        return env.get_elem_register(inst.rs.obj.val, inst.rs.obj.type, inst.rs.field.val)
    else:
        return emit_expr(inst.rs, env, output)


def emit_ir_assign_inst(inst, env, output):
    if isinstance(inst.rd, TypedIdentExpr):
        reg_lhs = env.get_register(inst.rd.val, inst.rd.type)

        if isinstance(inst.rs, (IntegerConstantExpr, FloatingConstantExpr)):
            emit_ldi(reg_lhs, inst.rs.val, env, output)
            return
        else:
            reg_rhs = get_rhs_reg(inst, env, output)

            assert(len(reg_lhs) == len(reg_rhs))

            for a, b in zip(reg_lhs, reg_rhs):
                emit_mov(a, b, env, output)
            emit_nop(env, output)
            return
    elif isinstance(inst.rd, TypedAccessorOp):
        if not isinstance(inst.rd.obj, TypedIdentExpr):
            raise NotImplementedError()

        reg_lhs = env.get_elem_register(
            inst.rd.obj.val, inst.rd.obj.type, inst.rd.field.val)

        if isinstance(inst.rs, (IntegerConstantExpr, FloatingConstantExpr)):
            emit_ldi(reg_lhs, inst.rs.val, env, output)
            return
        else:
            reg_rhs = get_rhs_reg(inst, env, output)

            assert(len(reg_lhs) == len(reg_rhs))

            for a, b in zip(reg_lhs, reg_rhs):
                emit_mov(a, b, env, output)
            emit_nop(env, output)
            return

    assert False, "Unreachable"


def emit_ir_inst(inst, env, output):
    # if isinstance(inst, ExprInst):
    #     output.append(f"# {print_ast_expr(inst.expr)}")
    #     emit_expr(inst.expr, env, output)
    #     return
    if isinstance(inst, PropPredInst):
        emit_ir_prop_pred_inst(inst, env, output)
        return
    if isinstance(inst, ReturnPredInst):
        emit_ir_return_pred_inst(inst, env, output)
        return
    if isinstance(inst, InvPredInst):
        emit_ir_inv_pred_inst(inst, env, output)
        return
    if isinstance(inst, RestorePredInst):
        emit_ir_restore_pred_inst(inst, env, output)
        return
    if isinstance(inst, PushPredInst):
        emit_ir_push_pred_inst(inst, env, output)
        return
    if isinstance(inst, BranchPredInst):
        emit_ir_branch_pred_inst(inst, env, output)
        return
    if isinstance(inst, JumpInst):
        emit_ir_jump_inst(inst, env, output)
        return

    print(inst)
    raise Exception


def emit_ir_block(block, env, output):
    emit_label(env.labels[block], env, output)
    for inst in block.insts:
        emit_ir_inst(inst, env, output)


def emit_ir_func(func, env, output, call):
    env.push_scope()

    if func.proto.ident.val == 'main':
        output.append('@qpu')
        output.append('def {0}(asm):'.format(func.proto.ident.val))

        func_insts = []

        # init predicate register
        func_insts.append('mov(r3, 1, set_flags=1)')

        func_insts.append('# main')
        for block in func.blocks:
            emit_ir_block(block, env, func_insts)
        func_insts.append('# end main')

        startup_insts = []
        startup_insts.append('# load input variables')
        emit_inputs_load(env, startup_insts)

        terminate_insts = []
        terminate_insts.append('# store output variables')
        emit_outputs_store(env, terminate_insts)
        terminate_insts.append('exit()')

        startup_insts = map(lambda l: indent + l, startup_insts)
        terminate_insts = map(lambda l: indent + l, terminate_insts)
        func_output = map(lambda l: indent + l, func_insts)

        output.extend(startup_insts)
        output.extend(func_output)
        output.extend(terminate_insts)
    elif call:
        for block in func.blocks:
            emit_ir_block(block, env, output)

    env.pop_scope()


def emit_ir_module(module):
    env = Env()
    env.funcs = module.funcs

    output = []

    for func in module.funcs:
        for block in func.blocks:
            env.labels[block] = f"L{len(env.labels)}"
        emit_ir_func(func, env, output, False)

    return output


def emit(module):
    return emit_ir_module(module)
