#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from ir.types import *
from ir.values import *
from codegen.dag import *
from codegen.spec import *
from codegen.mir_emitter import compute_value_types


class FunctionLoweringInfo:
    def __init__(self, func, calling_conv):
        self.func = func
        self.calling_conv = calling_conv

        self.can_lower_return = calling_conv.can_lower_return(func)
        self.frame_map = {}
        self.reg_value_map = {}

    def get_frame_idx(self, ir_value):
        if ir_value in self.frame_map:
            return self.frame_map[ir_value]

        return None


def align_offset(offset, align):
    return int(int((offset + align - 1) / align) * align)


def get_copy_from_parts_vector(parts, part_vt, value_vt, dag):
    from codegen.mir_emitter import get_vector_ty

    if len(parts) > 1:
        vec_vt = get_vector_ty(part_vt.value_type, len(parts))
        assert(vec_vt == value_vt.value_type)

        ops = parts

        return DagValue(dag.add_node(VirtualDagOps.BUILD_VECTOR, [value_vt], *ops), 0)

    raise NotImplementedError()


def get_copy_from_parts(parts, part_vt, value_vt, dag):
    if value_vt.is_vector:
        return get_copy_from_parts_vector(parts, part_vt, value_vt, dag)

    if len(parts) > 1:
        raise NotImplementedError()

    return parts[0]


def get_copy_to_parts_vector(value, parts, part_vt, value_vt, chain, dag):
    from codegen.mir_emitter import get_vector_ty

    if len(parts) > 1:
        vec_vt = get_vector_ty(part_vt.value_type, len(parts))
        assert(vec_vt == value_vt.value_type)

        for idx, part in enumerate(parts):
            idx_val = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I32), ConstantInt(idx, i32)), 0)

            elem = DagValue(dag.add_node(VirtualDagOps.EXTRACT_VECTOR_ELT, [
                part_vt], value, idx_val), 0)
            chain = DagValue(
                dag.add_copy_to_reg_node(part, elem), 0)

        return chain

    raise NotImplementedError()


def get_copy_to_parts(value, parts, part_vt, chain, dag, flags=None):
    value_vt = value.ty

    if value_vt.is_vector:
        return get_copy_to_parts_vector(value, parts, part_vt, value_vt, chain, dag)

    if len(parts) > 1:
        raise NotImplementedError()

    part = parts[0]
    if part.ty != value.ty:
        if part.ty.get_size_in_bits() > value.ty.get_size_in_bits():
            value = DagValue(dag.add_node(
                VirtualDagOps.ZERO_EXTEND, [part.ty], value), 0)
        else:
            value = DagValue(dag.add_node(
                VirtualDagOps.TRUNCATE, [part.ty], value), 0)

    if flags:
        vts = [MachineValueType(ValueType.OTHER),
               MachineValueType(ValueType.GLUE)]
        chain = DagValue(
            dag.add_node(VirtualDagOps.COPY_TO_REG, vts, chain, parts[0], value, flags), 0)
        return chain, chain.get_value(1)
    else:
        vts = [MachineValueType(ValueType.OTHER),
               MachineValueType(ValueType.GLUE)]
        chain = DagValue(
            dag.add_node(VirtualDagOps.COPY_TO_REG, vts, chain, parts[0], value), 0)

    return chain


def get_parts_to_copy_vector(value, num_parts, part_vt, value_vt, dag):
    from codegen.mir_emitter import get_vector_ty

    parts = []

    if num_parts > 1:
        vec_vt = get_vector_ty(part_vt.value_type, num_parts)
        assert(vec_vt == value_vt.value_type)

        for idx in range(num_parts):
            idx_val = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I32), ConstantInt(idx, i32)), 0)

            elem = DagValue(dag.add_node(VirtualDagOps.EXTRACT_VECTOR_ELT, [
                part_vt], value, idx_val), 0)

            parts.append(elem)

        return parts

    raise NotImplementedError()


def get_parts_to_copy(value, num_parts, part_vt, dag):
    value_vt = value.ty

    if value_vt.is_vector:
        return get_parts_to_copy_vector(value, num_parts, part_vt, value_vt, dag)

    if num_parts > 1:
        raise NotImplementedError()

    return [value]


class DagBuilder:
    def __init__(self, mfunc, mbb_map, data_layout: DataLayout, func_info):
        self.data_layout = data_layout
        self.inst_map = {}
        self.mbb_map = mbb_map
        self.g = Dag(mfunc)
        self.condcodes = {}
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.reg_info = mfunc.target_info.get_register_info()
        self.func_info = func_info

    def set_inst_value(self, inst, value):
        assert(isinstance(value, DagValue))
        self.inst_map[inst] = value

    @property
    def graph(self):
        return self.g

    @property
    def root(self):
        return self.g.root

    @property
    def control_root(self):
        return self.g.control_root

    @root.setter
    def root(self, value: DagValue):
        assert(isinstance(value, DagValue))
        self.g.root = value

    def get_cond_code(self, cond_code: CondCode):
        if cond_code in self.condcodes:
            return self.condcodes[cond_code]

        node = self.g.add_condition_code_node(cond_code)
        value = DagValue(node, 0)
        self.condcodes[cond_code] = value

        return value

    def get_or_create_constant(self, value: ConstantInt, is_target=False):
        assert(isinstance(value, (ConstantInt, ConstantFP,
                                  ConstantVector, ConstantPointerNull)))

        vt = compute_value_types(value.ty, self.data_layout)
        assert(len(vt) == 1)

        if isinstance(value, ConstantFP):
            if is_target:
                node = self.g.add_target_constant_fp_node(vt[0], value)
            else:
                node = self.g.add_constant_fp_node(vt[0], value)
            return DagValue(node, 0)

        if isinstance(value, ConstantVector):
            elem_values = []
            for elem_value in value.values:
                elem_values.append(
                    self.get_or_create_constant(elem_value, is_target))

            node = self.g.add_node(
                VirtualDagOps.BUILD_VECTOR, vt, *elem_values)
            return DagValue(node, 0)

        if isinstance(value, ConstantPointerNull):
            if is_target:
                node = self.g.add_target_constant_node(vt[0], 0)
            else:
                node = self.g.add_constant_node(vt[0], 0)
            return DagValue(node, 0)

        assert(isinstance(value, ConstantInt))
        if is_target:
            node = self.g.add_target_constant_node(vt[0], value.value)
        else:
            node = self.g.add_constant_node(vt[0], value.value)
        return DagValue(node, 0)

    def get_or_create_global_address(self, value: GlobalValue, target=False):
        vt = compute_value_types(value.ty, self.data_layout)

        node = self.g.add_global_address_node(vt[0], value, target)

        return DagValue(node, 0)

    def get_or_create_frame_idx(self, value):
        if value in self.func_info.frame_map:
            frame_idx = self.func_info.frame_map[value]
            ptr_ty = self.target_lowering.get_frame_index_type(
                self.data_layout)
            node = self.g.add_frame_index_node(ptr_ty, frame_idx)
            return DagValue(node, 0)

        raise NotImplementedError()

    def get_value(self, ir_value):
        if ir_value in self.inst_map:
            return self.inst_map[ir_value]

        if ir_value in self.func_info.reg_value_map:
            vregs = self.func_info.reg_value_map[ir_value]

            vts = compute_value_types(ir_value.ty, self.data_layout)

            values = []

            vreg_idx = 0
            for vt in vts:
                regs = []

                reg_vt = self.target_lowering.get_register_type(vt)
                reg_count = self.target_lowering.get_register_count(vt)

                for _ in range(reg_count):
                    vreg = vregs[vreg_idx]
                    reg = DagValue(self.g.add_register_node(reg_vt, vreg), 0)
                    reg = DagValue(
                        self.g.add_copy_from_reg_node(reg.ty, reg), 0)
                    regs.append(reg)

                    vreg_idx += 1

                values.append(get_copy_from_parts(
                    regs, reg_vt, vt, self.g))

            self.inst_map[ir_value] = self.g.add_merge_values(values)

            return self.inst_map[ir_value]

        if isinstance(ir_value, ConstantInt):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, ConstantFP):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, ConstantPointerNull):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, ConstantVector):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, GlobalVariable):
            return self.get_or_create_global_address(ir_value)

        if isinstance(ir_value, AllocaInst):
            return self.get_or_create_frame_idx(ir_value)

        if isinstance(ir_value, Function):
            return self.get_or_create_global_address(ir_value)

        raise NotImplementedError()

    def get_basic_block(self, ir_bb: BasicBlock):
        mbb = self.mbb_map[ir_bb]
        return DagValue(self.g.add_basic_block_node(mbb), 0)

    @staticmethod
    def get_dag_op_from_ir_op(op):
        IR_OP_HANDLE_MAP = {
            "add": VirtualDagOps.ADD,
            "sub": VirtualDagOps.SUB,
            "mul": VirtualDagOps.MUL,
            "sdiv": VirtualDagOps.SDIV,
            "udiv": VirtualDagOps.UDIV,

            "and": VirtualDagOps.AND,
            "or": VirtualDagOps.OR,
            "xor": VirtualDagOps.XOR,

            "shl": VirtualDagOps.SHL,
            "lshr": VirtualDagOps.SRL,
            "ashr": VirtualDagOps.SRA,

            "fadd": VirtualDagOps.FADD,
            "fsub": VirtualDagOps.FSUB,
            "fmul": VirtualDagOps.FMUL,
            "fdiv": VirtualDagOps.FDIV,
        }
        if op in IR_OP_HANDLE_MAP:
            return IR_OP_HANDLE_MAP[op]

        print(op)
        raise NotImplementedError()

    def visit_alloca(self, inst: AllocaInst):
        if len(inst.uses) == 0:
            return

        if self.func_info.get_frame_idx(inst) is not None:
            return

        raise NotImplementedError

        count = self.get_value(inst.count)
        elem_size = self.get_value(ConstantInt(
            compute_type_size(inst.alloca_ty), i32))
        size = self.g.add_node(VirtualDagOps.MUL, [], DagValue(
            count, 0), DagValue(elem_size, 0))

        node = self.g.add_node(
            VirtualDagOps.DYNAMIC_STACKALLOC, [], DagValue(size, 0))
        self.set_inst_value(inst, DagValue(node, 0))

    def visit_load(self, inst: LoadInst):
        rs = self.get_value(inst.rs)
        offsets = []
        vts = compute_value_types(inst.ty, self.data_layout, offsets)
        ptr_ty = self.target_lowering.get_pointer_type(
            self.data_layout, inst.rs.ty.addr_space)

        values = []
        for i, vt in enumerate(vts):
            offset_val = offsets[i]

            offset = self.get_or_create_constant(
                ConstantInt(offset_val, ptr_ty.get_ir_type()))

            assert(ptr_ty == rs.ty)
            assert(ptr_ty == offset.ty)

            if offset_val != 0:
                addr = DagValue(self.g.add_node(
                    VirtualDagOps.ADD, [rs.ty], rs, offset), 0)
            else:
                addr = rs

            value = DagValue(self.g.add_load_node(
                vt, self.root, addr, False), 0)
            values.append(value)

        node = self.g.add_merge_values(values)
        self.set_inst_value(inst, node)

    def visit_store(self, inst: StoreInst):
        rs = self.get_value(inst.rs)
        rd = self.get_value(inst.rd)
        offsets = []
        vts = compute_value_types(inst.rs.ty, self.data_layout, offsets)
        ptr_ty = self.target_lowering.get_pointer_type(
            self.data_layout, inst.rd.ty.addr_space)

        chains = []
        for i, vt in enumerate(vts):
            offset_val = offsets[i]

            offset = self.get_or_create_constant(
                ConstantInt(offset_val, ptr_ty.get_ir_type()))

            assert(ptr_ty == rd.ty)
            assert(ptr_ty == offset.ty)

            if offset_val != 0:
                addr = DagValue(self.g.add_node(
                    VirtualDagOps.ADD, [ptr_ty], rd, offset), 0)
            else:
                addr = rd

            value = DagValue(self.g.add_store_node(
                self.root, addr, DagValue(rs.node, rs.index+i), mem_value_ty=vt), 0)
            chains.append(value)

        node = self.g.add_node(VirtualDagOps.TOKEN_FACTOR, [
                               MachineValueType(ValueType.OTHER)], *chains)

    def visit_fence(self, inst: FenceInst):
        ordering = DagValue(self.g.add_constant_node(
            MachineValueType(ValueType.I32), inst.ordering.value), 0)

        sync_scope = DagValue(self.g.add_constant_node(
            MachineValueType(ValueType.I32), inst.syncscope.value.id), 0)

        vts = [MachineValueType(ValueType.OTHER)]
        node = self.g.add_node(VirtualDagOps.ATOMIC_FENCE,
                               vts, self.root, ordering, sync_scope)

        self.root = DagValue(node, 0)

    def visit_get_element_ptr(self, inst: GetElementPtrInst):
        ptr = self.get_value(inst.rs)
        ptr_ty = self.target_lowering.get_pointer_type(
            self.data_layout, inst.ty.addr_space)

        ty = inst.rs.ty
        for idx in inst.idx:
            if isinstance(idx, ConstantInt):
                offset, field_size = self.data_layout.get_elem_offset_in_bits(
                    ty, idx.value)
                offset_value = self.get_value(ConstantInt(
                    int(offset / 8), ptr_ty.get_ir_type()))
                if isinstance(ty, StructType):
                    ty = ty.fields[idx.value]
                elif isinstance(ty, PointerType):
                    ty = ty.elem_ty
                elif isinstance(ty, VectorType):
                    ty = ty.elem_ty
                elif isinstance(ty, ArrayType):
                    ty = ty.elem_ty
                else:
                    raise ValueError("Can't compute the offset.")

            else:
                offset = -1
                if isinstance(ty, (ArrayType, PointerType)):
                    elem_size = self.data_layout.get_type_alloc_size(
                        ty.elem_ty)
                    elem_size_value = self.get_value(
                        ConstantInt(elem_size, ptr_ty.get_ir_type()))

                    idx_value = self.get_value(idx)

                    if idx_value.ty != ptr_ty:
                        if ptr_ty.get_size_in_bits() > idx_value.ty.get_size_in_bits():
                            idx_value = DagValue(self.g.add_node(
                                VirtualDagOps.ZERO_EXTEND, [ptr_ty], idx_value), 0)
                        else:
                            idx_value = DagValue(self.g.add_node(
                                VirtualDagOps.TRUNCATE, [ptr_ty], idx_value), 0)

                    offset_value = DagValue(self.g.add_node(
                        VirtualDagOps.MUL, [ptr_ty], elem_size_value, idx_value), 0)

                    ty = ty.elem_ty
                else:
                    raise ValueError("Can't compute the offset.")

            if offset == 0:
                continue

            if offset_value.ty != ptr.ty:
                if ptr.ty.get_size_in_bits() > offset_value.ty.get_size_in_bits():
                    offset_value = DagValue(self.g.add_node(
                        VirtualDagOps.ZERO_EXTEND, [ptr_ty], offset_value), 0)
                else:
                    offset_value = DagValue(self.g.add_node(
                        VirtualDagOps.TRUNCATE, [ptr_ty], offset_value), 0)

            ptr = DagValue(self.g.add_node(
                VirtualDagOps.ADD, [ptr_ty], ptr, offset_value), 0)

        self.set_inst_value(inst, ptr)

        return

    def visit_insert_element(self, inst: InsertElementInst):
        vec = self.get_value(inst.vec)
        elem = self.get_value(inst.elem)
        idx = self.get_value(inst.idx)

        value = DagValue(self.g.add_node(
            VirtualDagOps.INSERT_VECTOR_ELT, vec.node.value_types, vec, elem, idx), 0)

        self.set_inst_value(inst, value)

    def compute_linear_index(self, ty, indices, idx=0):
        if isinstance(ty, StructType):
            for i, field_ty in enumerate(ty.fields):
                if indices and i == indices[0]:
                    return idx
                idx = self.compute_linear_index(field_ty, indices[1:], idx)

            return idx
        elif isinstance(ty, ArrayType):
            size = ty.size
            elem_size = self.compute_linear_index(ty.elem_ty, None)

            if indices:
                idx += elem_size * indices[0]
                idx = self.compute_linear_index(ty.elem_ty, indices[1:], idx)
                return idx

            idx += size * elem_size
            return idx

        return idx + 1

    def visit_extract_element(self, inst: ExtractElementInst):
        vec = self.get_value(inst.vec)
        idx = self.get_value(inst.idx)

        vts = compute_value_types(inst.ty, self.data_layout)

        value = DagValue(self.g.add_node(
            VirtualDagOps.EXTRACT_VECTOR_ELT, vts, vec, idx), 0)

        self.set_inst_value(inst, value)

    def visit_extract_value(self, inst: ExtractValueInst):
        vec = self.get_value(inst.value)
        indices = inst.idx

        linear_index = self.compute_linear_index(inst.value.ty, indices)

        vts = compute_value_types(inst.ty, self.data_layout)

        value = vec.get_value(linear_index)

        self.set_inst_value(inst, value)

    def visit_binary(self, inst: BinaryInst):
        rs_value = self.get_value(inst.rs)
        rt_value = self.get_value(inst.rt)

        op = self.get_dag_op_from_ir_op(inst.op)

        vts = compute_value_types(inst.rs.ty, self.data_layout)

        node = self.g.add_node(op, vts, rs_value, rt_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_unary(self, inst: UnaryInst):
        if inst.op == "sub":
            rs_value = self.get_value(ConstantInt(0, inst.rs.ty))
            rt_value = self.get_value(inst.rs)

            op = self.get_dag_op_from_ir_op(inst.op)

            vts = compute_value_types(inst.rs.ty, self.data_layout)

            node = self.g.add_node(op, vts, rs_value, rt_value)

            self.set_inst_value(inst, DagValue(node, 0))
        elif inst.op == "not":
            rs_value = self.get_value(ConstantInt(0, inst.rs.ty))
            rt_value = self.get_value(inst.rs)

            op = self.get_dag_op_from_ir_op(inst.op)

            vts = compute_value_types(inst.rs.ty, self.data_layout)

            node = self.g.add_node(op, vts, rs_value, rt_value)

            self.set_inst_value(inst, DagValue(node, 0))
        else:
            print(inst.op)
            raise NotImplementedError

    @staticmethod
    def get_condcode_from_op(op):
        TABLE = {
            "eq": CondCode.SETEQ,
            "ne": CondCode.SETNE,
            "sgt": CondCode.SETGT,
            "sge": CondCode.SETGE,
            "slt": CondCode.SETLT,
            "sle": CondCode.SETLE,
            "ugt": CondCode.SETUGT,
            "uge": CondCode.SETUGE,
            "ult": CondCode.SETULT,
            "ule": CondCode.SETULE,
        }

        if op in TABLE:
            return TABLE[op]

        raise NotImplementedError

    @staticmethod
    def get_fcmp_condcode_from_op(op):
        TABLE = {
            "eq": CondCode.SETEQ,
            "ne": CondCode.SETNE,
            "oeq": CondCode.SETEQ,
            "one": CondCode.SETNE,
            "ogt": CondCode.SETOGT,
            "oge": CondCode.SETOGE,
            "olt": CondCode.SETOLT,
            "ole": CondCode.SETOLE,
            "ugt": CondCode.SETUGT,
            "uge": CondCode.SETUGE,
            "ult": CondCode.SETULT,
            "ule": CondCode.SETULE,
        }

        if op in TABLE:
            return TABLE[op]

        raise NotImplementedError

    def visit_cmp(self, inst: CmpInst):
        rs_value = self.get_value(inst.rs)
        rt_value = self.get_value(inst.rt)

        cond = self.get_cond_code(self.get_condcode_from_op(inst.op))

        node = self.g.add_node(VirtualDagOps.SETCC, [
            MachineValueType(ValueType.I1)], rs_value, rt_value, cond)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_fcmp(self, inst: FCmpInst):
        rs_value = self.get_value(inst.rs)
        rt_value = self.get_value(inst.rt)

        cond = self.get_cond_code(self.get_fcmp_condcode_from_op(inst.op))

        node = self.g.add_node(VirtualDagOps.SETCC, [
            MachineValueType(ValueType.I1)], rs_value, rt_value, cond)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_trunc(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.TRUNCATE, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_zext(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        if rs_value.ty == to_ty[0]:
            self.set_inst_value(inst, rs_value)
            return

        node = self.g.add_node(
            VirtualDagOps.ZERO_EXTEND, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_sext(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        if rs_value.ty == to_ty[0]:
            self.set_inst_value(inst, rs_value)
            return

        node = self.g.add_node(
            VirtualDagOps.SIGN_EXTEND, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_fptrunc(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.FP_ROUND, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_fpext(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.FP_EXTEND, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_fptoui(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.FP_TO_UINT, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_fptosi(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.FP_TO_SINT, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_uitofp(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.UINT_TO_FP, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_sitofp(self, inst: TruncInst):
        rs_value = self.get_value(inst.rs)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        node = self.g.add_node(
            VirtualDagOps.SINT_TO_FP, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_ptrtoint(self, inst: TruncInst):
        from_ty = compute_value_types(inst.rs.ty, self.data_layout)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        assert(from_ty[0] == 1)
        assert(to_ty[0] == 1)

        if from_ty[0].get_size_in_bits() <= to_ty[0].get_size_in_bits():
            self.visit_zext(inst)
        else:
            self.visit_trunc(inst)

    def visit_inttoptr(self, inst: TruncInst):
        from_ty = compute_value_types(inst.rs.ty, self.data_layout)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        assert(from_ty[0] == 1)
        assert(to_ty[0] == 1)

        if from_ty[0].get_size_in_bits() <= to_ty[0].get_size_in_bits():
            self.visit_zext(inst)
        else:
            self.visit_trunc(inst)

    def visit_bit_cast(self, inst: BitCastInst):
        rs_value = self.get_value(inst.rs)

        from_ty = compute_value_types(inst.rs.ty, self.data_layout)
        to_ty = compute_value_types(inst.ty, self.data_layout)

        assert(len(from_ty) == 1)
        assert(len(to_ty) == 1)

        if from_ty[0] == to_ty[0]:
            self.set_inst_value(inst, rs_value)
            return

        node = self.g.add_node(
            VirtualDagOps.BITCAST, to_ty, rs_value)

        self.set_inst_value(inst, DagValue(node, 0))

    def visit_address_space_cast(self, inst: BitCastInst):
        raise NotImplementedError()

    def visit_branch(self, inst: BranchInst):
        cond_value = self.get_value(inst.cond)

        value = DagValue(self.g.add_node(VirtualDagOps.BRCOND, [MachineValueType(ValueType.OTHER)], self.control_root, cond_value,
                                         self.get_basic_block(inst.then_target)), 0)

        self.root = value

        value = DagValue(self.g.add_node(
            VirtualDagOps.BR, [MachineValueType(ValueType.OTHER)], self.control_root, self.get_basic_block(inst.else_target)), 0)

        self.root = value

        mbb = self.mbb_map[inst.block]
        true_mbb = self.mbb_map[inst.then_target]
        false_mbb = self.mbb_map[inst.else_target]

        mbb.add_successor(true_mbb)
        mbb.add_successor(false_mbb)

    def visit_switch(self, inst: SwitchInst):
        value = self.get_value(inst.value)

        for cast_val, cast_target in inst.cases:
            cond = self.get_cond_code(CondCode.SETEQ)
            cond_value = DagValue(self.g.add_node(VirtualDagOps.SETCC, [
                MachineValueType(ValueType.I1)], value, self.get_value(cast_val), cond), 0)

            self.root = DagValue(self.g.add_node(VirtualDagOps.BRCOND, [MachineValueType(ValueType.OTHER)], self.control_root, cond_value,
                                                 self.get_basic_block(cast_target)), 0)

        self.root = DagValue(self.g.add_node(
            VirtualDagOps.BR, [MachineValueType(ValueType.OTHER)], self.control_root, self.get_basic_block(inst.default)), 0)

        mbb = self.mbb_map[inst.block]

        for case_target in inst.case_dests:
            mbb.add_successor(self.mbb_map[case_target])
        mbb.add_successor(self.mbb_map[inst.default])

    def visit_jump(self, inst: JumpInst):
        value = self.g.add_node(VirtualDagOps.BR, [MachineValueType(ValueType.OTHER)], self.control_root,
                                self.get_basic_block(inst.goto_target))

        self.root = DagValue(value, 0)

        mbb = self.mbb_map[inst.block]
        goto_mbb = self.mbb_map[inst.goto_target]

        mbb.add_successor(goto_mbb)

    def get_memcpy(self, src_value, dst_value, size, src_align, dst_align):
        self.target_lowering.lower_optimal_memory_op(
            size.value, src_value, dst_value, src_align, dst_align, self)

    def visit_memcpy(self, inst: CallInst):
        op_dst = self.get_value(inst.args[0])
        op_src = self.get_value(inst.args[1])
        op_size = self.get_value(inst.args[2])

        if not isinstance(op_size.node, ConstantDagNode):
            raise NotImplementedError()

        self.get_memcpy(op_src, op_dst, op_size.node.value, 4, 4)

    def visit_intrinsic_call(self, inst: CallInst):
        if inst.callee.name.startswith("llvm.memcpy"):
            self.visit_memcpy(inst)

    def is_intrinsic_func(self, func):
        if func.name.startswith("llvm.memcpy"):
            return True

        return False

    def visit_inline_asm(self, inst: CallInst):
        inline_asm = inst.callee

        arg_idx = 0
        res_idx = 0

        class ConstraintType(Enum):
            Register = auto()
            RegisterClass = auto()
            Memory = auto()
            Immediate = auto()
            Other = auto()
            Unknown = auto()

        def get_constraint_type(code):
            if code.startswith("{") and code.endswith("}"):
                if code[1:-1] == "memory":
                    return ConstraintType.Memory

                return ConstraintType.Register

            return ConstraintType.Unknown

        class AsmOperandInfo:
            def __init__(self, constraint_info):
                self.ty = constraint_info.ty
                self.codes = constraint_info.codes
                self.operand_value = None
                self.constraint_vt = MachineValueType(ValueType.OTHER)
                self.assigned_regs = []

            def compute_constraint(self):
                if len(self.codes) == 1:
                    self.constraint_code = self.codes[0]
                    self.constraint_type = get_constraint_type(
                        self.constraint_code)
                else:
                    raise ValueError("Invalid constraint code.")

        op_infos = []

        vts = compute_value_types(inst.ty, self.data_layout)

        for constraint in inline_asm.parse_constraints():
            op_info = AsmOperandInfo(constraint)
            op_infos.append(op_info)

            if constraint.ty == ConstraintPrefix.Output:
                op_info.constraint_vt = vts[res_idx]
                res_idx += 1
            elif constraint.ty == ConstraintPrefix.Input:
                op_info.operand_value = inst.args[arg_idx]
                arg_idx += 1

            if op_info.operand_value:
                op_info.constraint_vt = compute_value_types(
                    op_info.operand_value.ty, self.data_layout)[0]

            op_info.compute_constraint()

        ptr_ty = self.target_lowering.get_pointer_type(self.data_layout)

        asm_node_operands = []
        asm_node_operands.append(
            DagValue(self.g.add_external_symbol_node(ptr_ty, inline_asm.asm_string, True), 0))

        for op_info in op_infos:
            if op_info.ty in [ConstraintPrefix.Input, ConstraintPrefix.Output]:
                if op_info.constraint_type == ConstraintType.Register:
                    reg = self.target_lowering.get_reg_for_inline_asm_constraint(
                        self.reg_info, op_info.constraint_code, op_info.constraint_vt)
                    if not reg:
                        continue

                    op_info.assigned_regs.append(
                        (MachineRegister(reg), op_info.constraint_vt))

        chain = self.g.root
        flags = DagValue()

        for op_info in op_infos:
            if op_info.ty == ConstraintPrefix.Input:

                if op_info.constraint_type == ConstraintType.Register:
                    regs = []
                    for reg, vt in op_info.assigned_regs:
                        reg_val = DagValue(
                            self.g.add_register_node(vt, reg), 0)
                        regs.append(reg_val)
                        asm_node_operands.append(reg_val)

                    chain, flags = get_copy_to_parts(self.get_value(
                        op_info.operand_value), regs, op_info.constraint_vt, chain, self.g, flags)

        asm_node_operands.append(chain)

        chain = DagValue(self.g.add_node(VirtualDagOps.INLINEASM, [
                         MachineValueType(ValueType.OTHER)], *asm_node_operands, flags), 0)
        flags = chain.get_value(1)

        result_values = []
        for op_info in op_infos:
            if op_info.ty == ConstraintPrefix.Output:
                if op_info.constraint_type == ConstraintType.Register:
                    regs = []
                    for reg, vt in op_info.assigned_regs:
                        reg_val = DagValue(
                            self.g.add_register_node(vt, reg), 0)
                        regs.append(reg_val)
                        asm_node_operands.append(reg_val)

                    vts = [op_info.constraint_vt, MachineValueType(
                        ValueType.OTHER), MachineValueType(ValueType.GLUE)]
                    value = DagValue(self.g.add_node(
                        VirtualDagOps.COPY_FROM_REG, vts, chain, regs[0], flags), 0)
                    chain = value.get_value(1)
                    flags = value.get_value(2)

                    result_values.append(value)

        result_vts = compute_value_types(inst.ty, self.data_layout)

        if result_values:
            value = DagValue(self.g.add_node(
                VirtualDagOps.MERGE_VALUES, result_vts, *result_values), 0)

            self.set_inst_value(inst, value)

    def visit_call(self, inst: CallInst):
        callee = inst.callee
        if isinstance(callee, InlineAsm):
            self.visit_inline_asm(inst)
            return

        if self.is_intrinsic_func(callee):
            self.visit_memcpy(inst)
            return

        value = self.mfunc.target_info.get_calling_conv().lower_call(self, inst, self.g)

        if value is not None:
            self.set_inst_value(inst, value)

    def visit_return(self, inst: ReturnInst):
        value = self.mfunc.target_info.get_calling_conv().lower_return(self, inst, self.g)

        self.root = DagValue(value, 0)

    def create_regs(self, ty):
        vts = compute_value_types(ty, self.data_layout)

        regs = []
        for ty in vts:
            vreg = self.target_lowering.get_machine_vreg(ty)
            regs.append(vreg)

        return regs

    def handle_phi_node_in_succs(self, inst):
        def get_incomming_value_for_bb(phi: PHINode, bb):
            return phi.values[bb]

        block = inst.block
        for succ in inst.successors:
            for phi in succ.phis:
                incomming_value = get_incomming_value_for_bb(phi, inst.block)
                phi_value = self.get_value(incomming_value)

                if phi in self.func_info.reg_value_map:
                    vregs = self.func_info.reg_value_map[phi]
                else:
                    vregs = [self.mfunc.reg_info.create_virtual_register(
                        reg) for reg in self.create_regs(phi.ty)]

                    self.func_info.reg_value_map[phi] = vregs

                vts = compute_value_types(phi.ty, self.data_layout)

                reg_info = self.mfunc.reg_info

                vreg_idx = 0
                for val_idx, vt in enumerate(vts):
                    regs = []

                    reg_vt = self.target_lowering.get_register_type(vt)
                    reg_count = self.target_lowering.get_register_count(vt)

                    for reg_idx in range(reg_count):
                        vreg = vregs[vreg_idx]
                        reg = DagValue(
                            self.g.add_register_node(reg_vt, vreg), 0)
                        regs.append(reg)

                        vreg_idx += 1

                    self.g.root = get_copy_to_parts(phi_value.get_value(val_idx),
                                                    regs, reg_vt, self.g.root, self.g)

    def visit(self, inst):
        if inst.is_terminator:
            self.handle_phi_node_in_succs(inst)

        if isinstance(inst, AllocaInst):
            self.visit_alloca(inst)
        elif isinstance(inst, LoadInst):
            self.visit_load(inst)
        elif isinstance(inst, StoreInst):
            self.visit_store(inst)
        elif isinstance(inst, FenceInst):
            self.visit_fence(inst)
        elif isinstance(inst, BinaryInst):
            self.visit_binary(inst)
        elif isinstance(inst, GetElementPtrInst):
            self.visit_get_element_ptr(inst)
        elif isinstance(inst, InsertElementInst):
            self.visit_insert_element(inst)
        elif isinstance(inst, ExtractElementInst):
            self.visit_extract_element(inst)
        elif isinstance(inst, ExtractValueInst):
            self.visit_extract_value(inst)
        elif isinstance(inst, TruncInst):
            self.visit_trunc(inst)
        elif isinstance(inst, ZExtInst):
            self.visit_zext(inst)
        elif isinstance(inst, SExtInst):
            self.visit_sext(inst)
        elif isinstance(inst, FPTruncInst):
            self.visit_fptrunc(inst)
        elif isinstance(inst, FPExtInst):
            self.visit_fpext(inst)
        elif isinstance(inst, FPToUIInst):
            self.visit_fptoui(inst)
        elif isinstance(inst, UIToFPInst):
            self.visit_uitofp(inst)
        elif isinstance(inst, FPToSIInst):
            self.visit_fptosi(inst)
        elif isinstance(inst, SIToFPInst):
            self.visit_sitofp(inst)
        elif isinstance(inst, PtrToIntInst):
            self.visit_ptrtoint(inst)
        elif isinstance(inst, IntToPtrInst):
            self.visit_inttoptr(inst)
        elif isinstance(inst, BitCastInst):
            self.visit_bit_cast(inst)
        elif isinstance(inst, AddrSpaceCastInst):
            self.visit_address_space_cast(inst)
        elif isinstance(inst, CmpInst):
            self.visit_cmp(inst)
        elif isinstance(inst, FCmpInst):
            self.visit_fcmp(inst)
        elif isinstance(inst, JumpInst):
            self.visit_jump(inst)
        elif isinstance(inst, BranchInst):
            self.visit_branch(inst)
        elif isinstance(inst, SwitchInst):
            self.visit_switch(inst)
        elif isinstance(inst, CallInst):
            self.visit_call(inst)
        elif isinstance(inst, ReturnInst):
            self.visit_return(inst)
        elif isinstance(inst, PHINode):
            pass
        else:
            raise NotImplementedError(
                "{0} is not a supporting instruction.".format(inst.__class__.__name__))

        # Export virtual registers acrossing basic blocks.
        if inst in self.func_info.reg_value_map:
            value_types = compute_value_types(inst.ty, self.data_layout)

            vregs = self.func_info.reg_value_map[inst]

            reg_idx = 0
            for idx, vt in enumerate(value_types):
                regs = []

                reg_vt = self.target_lowering.get_register_type(vt)
                reg_count = self.target_lowering.get_register_count(vt)

                for _ in range(reg_count):
                    vreg = vregs[reg_idx]
                    regs.append(
                        DagValue(self.g.add_register_node(reg_vt, vreg), 0))

                    reg_idx += 1

                src = self.get_value(inst).get_value(idx)

                self.root = get_copy_to_parts(
                    src, regs, reg_vt, self.root, self.g)
