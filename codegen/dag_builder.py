#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from ir.types import *
from ir.values import *
from codegen.dag import *
from codegen.spec import *
from codegen.mir_emitter import compute_value_types


def align_offset(offset, align):
    return int(int((offset + align - 1) / align) * align)


class DagBuilder:
    def __init__(self, mfunc, mbb_map, data_layout: DataLayout):
        self.data_layout = data_layout
        self.func_info = mfunc.func_info
        self.inst_map = {}
        self.mbb_map = mbb_map
        self.g = Dag(mfunc)
        self.condcodes = {}
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()

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
        assert(isinstance(value, (ConstantInt, ConstantFP, ConstantVector)))

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
            vreg = self.func_info.reg_value_map[ir_value]

            value_types = compute_value_types(ir_value.ty, self.data_layout)
            if len(value_types) > 1:
                raise NotImplementedError()

            vt = value_types[0]

            node = self.g.add_register_node(vt, vreg)

            node = self.g.add_copy_from_reg_node(vt, DagValue(node, 0))
            self.inst_map[ir_value] = DagValue(node, 0)

            return self.inst_map[ir_value]

        if isinstance(ir_value, ConstantInt):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, ConstantFP):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, ConstantVector):
            return self.get_or_create_constant(ir_value)

        if isinstance(ir_value, GlobalVariable):
            return self.get_or_create_global_address(ir_value)

        if isinstance(ir_value, AllocaInst):
            return self.get_or_create_frame_idx(ir_value)

        raise NotImplementedError()

        value_types = compute_value_types(ir_value.ty, self.data_layout)

        if len(value_types) > 1:
            raise NotImplementedError()

        regs = []
        for ty in value_types:
            vreg = self.target_lowering.get_machine_vreg(ty)
            regs.append(vreg)

        if ir_value in self.func_info.reg_value_map:
            reg = self.func_info.reg_value_map[ir_value]
        else:
            reg = self.func_info.reg_value_map[ir_value] = MachineVirtualRegister(
                regs[0], len(self.func_info.reg_value_map))

        node = RegisterDagNode(value_types, reg)

        node = self.g.add_copy_from_reg_node(DagValue(node, 0))
        self.inst_map[ir_value] = DagValue(node, 0)

        return self.inst_map[ir_value]

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

        values = []
        for i, vt in enumerate(vts):
            offset = self.get_or_create_constant(
                ConstantInt(offsets[i], inst.rs.ty))
            if offset != 0:
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

        chains = []
        for i, vt in enumerate(vts):
            offset = self.get_or_create_constant(
                ConstantInt(offsets[i], inst.rd.ty))
            if offset.node.value.value != 0:
                addr = DagValue(self.g.add_node(
                    VirtualDagOps.ADD, [rd.ty], rd, offset), 0)
            else:
                addr = rd
            value = DagValue(self.g.add_store_node(
                self.root, addr, DagValue(rs.node, i)), 0)
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
            self.data_layout, inst.pointee_ty.addr_space)

        ty = inst.pointee_ty
        for idx in inst.idx:
            if isinstance(idx, ConstantInt):
                offset, field_size = self.data_layout.get_elem_offset_in_bits(
                    ty, idx.value)
                offset = self.get_value(ConstantInt(
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
                    raise ValueError("The type isn't composite type.")

            else:
                raise NotImplementedError

            ptr = DagValue(self.g.add_node(
                VirtualDagOps.ADD, [ptr_ty], ptr, offset), 0)

        self.set_inst_value(inst, ptr)

        return

    def visit_insert_element(self, inst: InsertElementInst):
        vec = self.get_value(inst.vec)
        elem = self.get_value(inst.elem)
        idx = self.get_value(inst.idx)

        value = DagValue(self.g.add_node(
            VirtualDagOps.INSERT_VECTOR_ELT, vec.node.value_types, vec, elem, idx), 0)

        self.set_inst_value(inst, value)

    def visit_extract_element(self, inst: InsertElementInst):
        vec = self.get_value(inst.vec)
        idx = self.get_value(inst.idx)

        vts = compute_value_types(inst.ty, self.data_layout)

        value = DagValue(self.g.add_node(
            VirtualDagOps.INSERT_VECTOR_ELT, vts, vec, elem, idx), 0)

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

    def visit_fcmp(self, inst: CmpInst):
        rs_value = self.get_value(inst.rs)
        rt_value = self.get_value(inst.rt)

        cond = self.get_cond_code(self.get_fcmp_condcode_from_op(inst.op))

        node = self.g.add_node(VirtualDagOps.SETCC, [
            MachineValueType(ValueType.I1)], rs_value, rt_value, cond)

        self.set_inst_value(inst, DagValue(node, 0))

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

    def visit_branch(self, inst: BranchInst):
        cond_value = self.get_value(inst.cond)

        minus1 = self.get_or_create_constant(ConstantInt(-1, inst.cond.ty))
        not_cond_value = DagValue(self.g.add_node(
            VirtualDagOps.XOR, [MachineValueType(ValueType.I1)], cond_value, minus1), 0)

        value = self.g.add_node(VirtualDagOps.BRCOND, [MachineValueType(ValueType.OTHER)], self.control_root, cond_value,
                                self.get_basic_block(inst.then_target))

        self.root = DagValue(value, 0)

        value = self.g.add_node(
            VirtualDagOps.BR, [MachineValueType(ValueType.OTHER)], self.control_root, self.get_basic_block(inst.else_target))

        self.root = DagValue(value, 0)

        mbb = self.mbb_map[inst.block]
        true_mbb = self.mbb_map[inst.then_target]
        false_mbb = self.mbb_map[inst.else_target]

        mbb.add_successor(true_mbb)
        mbb.add_successor(false_mbb)

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

    def visit_call(self, inst: CallInst):
        callee = inst.callee
        if self.is_intrinsic_func(callee):
            self.visit_memcpy(inst)
            return

        value = self.mfunc.target_info.get_calling_conv().lower_call(self, inst, self.g)

        if value is not None:
            self.set_inst_value(inst, value)

    def visit_return(self, inst: ReturnInst):
        # if self.func_info.can_lower_return:
        value = self.mfunc.target_info.get_calling_conv().lower_return(self, inst, self.g)
        # else:
        #     raise NotImplementedError()

        self.root = DagValue(value, 0)

    def visit(self, inst):
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
        elif isinstance(inst, BitCastInst):
            self.visit_bit_cast(inst)
        elif isinstance(inst, CmpInst):
            self.visit_cmp(inst)
        elif isinstance(inst, FCmpInst):
            self.visit_fcmp(inst)
        elif isinstance(inst, JumpInst):
            self.visit_jump(inst)
        elif isinstance(inst, BranchInst):
            self.visit_branch(inst)
        elif isinstance(inst, CallInst):
            self.visit_call(inst)
        elif isinstance(inst, ReturnInst):
            self.visit_return(inst)
        else:
            raise NotImplementedError(
                "{0} is not a supporting instruction.".format(inst.__class__.__name__))

        if inst in self.func_info.reg_value_map:
            value_types = compute_value_types(inst.ty, self.data_layout)
            if len(value_types) > 1:
                raise NotImplementedError()

            vreg = self.func_info.reg_value_map[inst]

            node = self.g.add_register_node(value_types[0], vreg)

            src = self.get_value(inst)

            node = self.g.add_copy_to_reg_node(DagValue(node, 0), src)

            self.root = DagValue(node, 0)
