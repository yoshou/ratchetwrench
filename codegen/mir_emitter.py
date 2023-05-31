#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.dag_builder import *
from codegen.spec import *
from codegen.mir import *
from ir.types import *


def get_struct_member_offset(ty: CompositeType, idx, data_layout: DataLayout):
    size, align = data_layout.get_elem_offset_in_bits(ty, idx)
    assert(size % 8 == 0)
    return int(size / 8)


def get_vector_ty(vt, num_elems):
    if vt == ValueType.I1:
        pass
    if vt == ValueType.F32:
        if num_elems == 1:
            return ValueType.V1F32
        elif num_elems == 2:
            return ValueType.V2F32
        elif num_elems == 4:
            return ValueType.V4F32
        elif num_elems == 8:
            return ValueType.V8F32
        elif num_elems == 16:
            return ValueType.V16F32
        elif num_elems == 32:
            return ValueType.V16F32
        elif num_elems == 64:
            return ValueType.V64F32
        elif num_elems == 128:
            return ValueType.V128F32
        elif num_elems == 256:
            return ValueType.V256F32
        elif num_elems == 1024:
            return ValueType.V1024F32

    raise ValueError("The type is not supported.")


def get_vt(ty):
    INT_VALUE_TYPES = {
        i1: ValueType.I1,
        i8: ValueType.I8,
        i16: ValueType.I16,
        i32: ValueType.I32,
        i64: ValueType.I64,
        f16: ValueType.F16,
        f32: ValueType.F32,
        f64: ValueType.F64,
        f128: ValueType.F128,
    }

    if isinstance(ty, PrimitiveType):
        return INT_VALUE_TYPES[ty]
    elif isinstance(ty, PointerType):
        return ValueType.I64

    raise ValueError("The type is not primitive")


def compute_value_types(ty, data_layout, offsets=None, start_offset=0):
    if offsets is None:
        offsets = []

    if isinstance(ty, StructType):
        vts = []
        for field_idx, field_ty in enumerate(ty.fields):
            field_offset = get_struct_member_offset(ty, field_idx, data_layout)
            vts.extend(compute_value_types(field_ty, data_layout,
                                           offsets, start_offset + field_offset))
        return vts

    if isinstance(ty, VoidType):
        return []

    INT_VALUE_TYPES = {
        i1: ValueType.I8,
        i8: ValueType.I8,
        i16: ValueType.I16,
        i32: ValueType.I32,
        i64: ValueType.I64,
        f16: ValueType.F16,
        f32: ValueType.F32,
        f64: ValueType.F64,
        f128: ValueType.F128,
    }

    if isinstance(ty, PrimitiveType):
        vts = [MachineValueType(INT_VALUE_TYPES[ty])]
    elif isinstance(ty, PointerType):
        vts = [get_int_value_type(
            data_layout.get_pointer_size_in_bits(0))]
    elif isinstance(ty, VectorType):
        elem_vt = get_vt(ty.elem_ty)
        vt = get_vector_ty(elem_vt, ty.size)
        vts = [MachineValueType(vt)]
    elif isinstance(ty, ArrayType):
        elem_size = data_layout.get_type_alloc_size(ty.elem_ty)
        vts = []
        for i in range(ty.size):
            elem_vts = compute_value_types(
                ty.elem_ty, data_layout, offsets, start_offset + i * elem_size)

            vts.extend(elem_vts)
    elif isinstance(ty, FunctionType):
        vts = [MachineValueType(ValueType.OTHER)]
    else:
        raise ValueError("Can't get suitable value types.")

    offsets.append(start_offset)
    return vts


class MachineInstrEmitter:
    def __init__(self, bb: MachineBasicBlock, vr_map):
        self.bb = bb
        self.vr_map = vr_map

    def create_virtual_register(self, regclass):
        if isinstance(regclass, MachineRegisterClassDef):
            vreg = self.bb.func.reg_info.create_virtual_register(regclass)
            return vreg

        raise ValueError(
            "Argument regclass must be type of MachineVirtualRegister.")

    def get_virtual_register(self, node, idx):
        if node in self.vr_map:
            vals = self.vr_map[node]
            return vals[idx]

        if node.opcode == TargetDagOps.IMPLICIT_DEF:
            reg_info = self.bb.func.target_info.get_register_info()

            def get_subclass_with_subreg(regclass, subreg):
                if isinstance(subreg, ComposedSubRegDescription):
                    regclass = get_subclass_with_subreg(
                        regclass, subreg.subreg_a)
                    subreg = subreg.subreg_b

                return regclass.subclass_and_subregs[subreg]

            regclass = reg_info.get_regclass_for_vt(node.value_types[0])
            vreg = self.create_virtual_register(regclass)

            inst = self.create_instruction(TargetDagOps.IMPLICIT_DEF)
            reg = inst.add_reg(vreg, RegState.Define)

            self.bb.append_inst(inst)

            self.vr_map[node] = [reg]

            return reg

        raise ValueError(
            "Argument node is not mapped.")

    def add_register_operand(self, minst: MachineInstruction, operand: DagValue):
        assert(isinstance(operand, DagValue))

        is_kill = len(
            operand.node.uses) <= 1 and operand.node.opcode != VirtualDagOps.COPY_FROM_REG

        vreg_op = self.get_virtual_register(operand.node, operand.index)
        minst.add_reg(
            vreg_op.reg, RegState.Kill if is_kill else RegState.Non)

    def add_operand(self, minst: MachineInstruction, operand: DagValue, ii=None, operand_idx=-1):
        assert(isinstance(operand, DagValue))

        if isinstance(operand.node, ConstantDagNode):
            minst.add_imm(operand.node.value.value)
        elif isinstance(operand.node, ConstantPoolDagNode):
            align = operand.node.align
            ty = operand.node.constant.ty
            cp = self.bb.func.constant_pool

            if align == 0:
                align = int(cp.data_layout.get_pref_type_alignment(ty) / 8)

            index = cp.get_or_create_index(operand.node.value, align)

            minst.add_constant_pool_index(index, operand.node.target_flags)
        elif isinstance(operand.node, RegisterDagNode):
            reg_state = RegState.Non
            if operand_idx >= 0 and ii and operand_idx >= ii.num_operands:
                reg_state = RegState.Implicit
            minst.add_reg(operand.node.reg, reg_state)
        elif isinstance(operand.node, GlobalAddressDagNode):
            minst.add_global_address(
                operand.node.value, operand.node.target_flags)
        elif isinstance(operand.node, FrameIndexDagNode):
            minst.add_frame_index(operand.node.index)
        elif isinstance(operand.node, BasicBlockDagNode):
            minst.add_mbb(operand.node.bb)
        elif isinstance(operand.node, ExternalSymbolDagNode):
            minst.add_external_symbol(
                operand.node.symbol, operand.node.target_flags)
        elif isinstance(operand.node, RegisterMaskDagNode):
            minst.add_reg_mask(operand.node.mask)
        elif isinstance(operand.node, (MachineDagNode, DagNode)):
            self.add_register_operand(minst, operand)
        else:
            raise ValueError("This type of node is not able to emit.")

    def create_instruction(self, opcode):
        minst = MachineInstruction(opcode)
        return minst

    def emit_copy_to_reg(self, inst: DagNode, dag):
        chain = inst.operands[0]
        dest = inst.operands[1]
        src = inst.operands[2]

        assert(isinstance(dest.node, RegisterDagNode))
        assert(isinstance(dest.node.reg, (MachineRegister, MachineVirtualRegister)))

        minst = self.create_instruction(TargetDagOps.COPY)

        dst_reg = dest.node.reg
        if src.node.opcode == VirtualDagOps.REGISTER:
            src_reg = src.node.reg
        # else:
        #     src_reg = self.get_virtual_register(src.node, src.index).reg

            if isinstance(src_reg, MachineRegister):
                if src_reg == dst_reg:
                    return

        defs = []

        vreg_op = minst.add_reg(dst_reg, RegState.Define)
        self.add_operand(minst, src)
        defs.append(vreg_op)

        self.set_vreg_map(inst, defs)

        self.bb.append_inst(minst)

    def emit_copy_from_reg(self, inst: DagNode, dag):
        chain = inst.operands[0]
        src = inst.operands[1]

        assert(isinstance(src.node, RegisterDagNode))
        assert(isinstance(src.node.reg, (MachineRegister, MachineVirtualRegister)))

        if isinstance(src.node.reg, (MachineVirtualRegister)):
            if len(inst.operands) > 2 and inst.operands[2].ty == MachineValueType(ValueType.GLUE):
                pass
            else:
                defs = []
                defs.append(MOReg(src.node.reg, RegState.Non))
                self.set_vreg_map(inst, defs)
                return

        minst = self.create_instruction(TargetDagOps.COPY)

        if len(inst.operands) > 2 and inst.operands[2].ty == MachineValueType(ValueType.GLUE):
            minst.comment = "Glued"

        defs = []
        reg_info = self.bb.func.target_info.get_register_info()

        dst_regclass = reg_info.get_regclass_from_reg(src.node.reg.spec)

        dst_reg = self.create_virtual_register(dst_regclass)
        vreg_op = minst.add_reg(dst_reg, RegState.Define)
        defs.append(vreg_op)

        self.add_operand(minst, src)

        self.set_vreg_map(inst, defs)

        self.bb.append_inst(minst)
        return

    def set_vreg_map(self, node: DagNode, vreg):
        self.vr_map[node] = vreg

    def emit_subreg_node(self, inst: DagNode, dag):
        if inst.opcode == TargetDagOps.EXTRACT_SUBREG:
            src = inst.operands[0]
            subreg_idx = inst.operands[1]

            minst = self.create_instruction(TargetDagOps.COPY)

            src_reg_op = self.get_virtual_register(src.node, src.index)

            src_reg = src_reg_op.reg

            if isinstance(src_reg, MachineVirtualRegister):
                reg = src_reg.regclass
            else:
                raise NotImplementedError()

            defs = []
            reg_info = self.bb.func.target_info.get_register_info()

            def get_subclass_with_subreg(regclass, subreg):
                if isinstance(subreg, ComposedSubRegDescription):
                    regclass = get_subclass_with_subreg(
                        regclass, subreg.subreg_a)
                    subreg = subreg.subreg_b

                return regclass.subclass_and_subregs[subreg]

            assert(isinstance(subreg_idx.node, ConstantDagNode))
            dst_regclass = reg_info.get_regclass_for_vt(inst.value_types[0])
            # dst_regclass = get_subclass_with_subreg(
            #     dst_regclass, subregs[subreg_idx.node.value.value])

            # Add dest operand
            vreg = self.create_virtual_register(dst_regclass)
            vreg_op = minst.add_reg(vreg, RegState.Define)
            defs.append(vreg_op)

            # Add src operand
            minst.add_operand(
                MOReg(src_reg, RegState.Non, subreg_idx.node.value.value))

            self.set_vreg_map(inst, defs)

            self.bb.append_inst(minst)
        elif inst.opcode in [TargetDagOps.INSERT_SUBREG, TargetDagOps.SUBREG_TO_REG]:
            src = inst.operands[0]
            elem = inst.operands[1]
            subreg_idx = inst.operands[2]

            minst = self.create_instruction(inst.opcode)

            defs = []
            reg_info = self.bb.func.target_info.get_register_info()

            def get_subclass_with_subreg(regclass, subreg):
                if isinstance(subreg, ComposedSubRegDescription):
                    regclass = get_subclass_with_subreg(
                        regclass, subreg.subreg_a)
                    subreg = subreg.subreg_b

                return regclass.subclass_and_subregs[subreg]

            assert(isinstance(subreg_idx.node, ConstantDagNode))
            src_regclass = reg_info.get_regclass_for_vt(src.ty)
            src_regclass = get_subclass_with_subreg(
                src_regclass, subregs[subreg_idx.node.value.value])

            src_reg = self.create_virtual_register(src_regclass)
            vreg_op = minst.add_reg(src_reg, RegState.Define)
            defs.append(vreg_op)

            self.add_operand(minst, src)
            self.add_operand(minst, elem)
            self.add_operand(minst, subreg_idx)

            # Tie operands
            minst.operands[0].tied_to = 1
            minst.operands[1].tied_to = 0

            self.set_vreg_map(inst, defs)

            self.bb.append_inst(minst)

    def emit_copy_to_regclass(self, inst: DagNode, dag):
        src = inst.operands[0]
        regclass = inst.operands[1]

        minst = self.create_instruction(TargetDagOps.COPY)

        defs = []
        reg_info = self.bb.func.target_info.get_register_info()

        assert(isinstance(regclass.node, ConstantDagNode))
        regclasses_id = regclass.node.value.value
        dst_regclass = regclasses[regclasses_id]

        dst_reg = self.create_virtual_register(dst_regclass)
        vreg_op = minst.add_reg(dst_reg, RegState.Define)
        defs.append(vreg_op)

        self.add_operand(minst, src)

        self.set_vreg_map(inst, defs)

        self.bb.append_inst(minst)
        return

    def emit_machine_node(self, inst: DagNode, dag):
        defs = []

        minst = self.create_instruction(inst.opcode)

        ii = inst.opcode

        num_defs = len(ii.outs)
        for ty, _ in zip(inst.value_types, ii.outs):
            if ty.value_type == ValueType.GLUE:
                continue
            if ty.value_type == ValueType.OTHER:
                continue

            operand = list(ii.outs.items())[len(defs)]

            if isinstance(operand[1], MachineRegisterClassDef):
                vreg = self.create_virtual_register(operand[1])
            else:
                raise NotImplementedError()

            vreg_op = minst.add_reg(vreg, RegState.Define)
            defs.append(vreg_op)

        num_operands = ii.num_operands - num_defs

        operands = list(inst.operands)
        while operands and operands[-1].ty.value_type == ValueType.GLUE:
            operands.pop()
        if operands and operands[-1].ty.value_type == ValueType.OTHER:
            operands.pop()

        for i in range(len(operands)):
            operand = operands[i]

            if isinstance(operand.node, DagNode):
                if operand.node.opcode == VirtualDagOps.TOKEN_FACTOR:
                    continue

            self.add_operand(minst, operand, ii, i)

        constraint_dic = {c.op1: c.op2 for c in ii.constraints}

        operand_names = [name for (name, info) in ii.operands]

        for i, (name, info) in enumerate(ii.operands):
            if name in constraint_dic:
                tied_to = operand_names.index(constraint_dic[name])
                assert(info == ii.operands[tied_to][1])

                minst.operands[i].tied_to = tied_to
                minst.operands[tied_to].tied_to = i

                # if minst.operands[i].is_reg:
                #     minst.operands[i].is_kill = False
                # if minst.operands[tied_to].is_reg:
                #     minst.operands[tied_to].is_kill = False

        for imp_use in ii.uses:
            if isinstance(imp_use, MachineRegisterDef):
                reg = minst.add_reg(
                    MachineRegister(imp_use), RegState.Implicit)
            else:
                raise NotImplementedError()

        for imp_def in ii.defs:
            if isinstance(imp_def, MachineRegisterDef):
                reg = minst.add_reg(
                    MachineRegister(imp_def), RegState.Define | RegState.Implicit)

                defs.append(reg)
            else:
                raise NotImplementedError()

        self.set_vreg_map(inst, defs)

        self.bb.append_inst(minst)

    def emit_inlineasm(self, inst: DagNode, dag):
        minst = self.create_instruction(TargetDagOps.INLINEASM)
        asm_str = inst.operands[0].node.symbol

        minst.add_external_symbol(asm_str)

        self.bb.append_inst(minst)

    def emit(self, inst: DagNode, dag):
        if isinstance(inst, MachineDagNode):
            self.emit_machine_node(inst, dag)
            return

        if inst.opcode == TargetDagOps.COPY:
            assert(isinstance(
                inst.operands[0].node, RegisterDagNode))
            assert(isinstance(
                inst.operands[0].node.reg, (MachineRegister, MachineVirtualRegister)))

            minst = self.create_instruction(TargetDagOps.COPY)

            dst_reg = inst.operands[0].node.reg

            if isinstance(inst.operands[1].node, RegisterDagNode):
                src_reg_op = self.get_virtual_register(
                    inst.operands[1].node, inst.operands[1].index)

                src_reg = src_reg_op.reg

                if isinstance(src_reg, MachineRegister):
                    if src_reg == dst_reg:
                        return

            minst.add_reg(dst_reg, RegState.Define)
            self.add_operand(minst, inst.operands[1])

            self.bb.append_inst(minst)
            return

        if inst.opcode == TargetDagOps.EXTRACT_SUBREG:
            self.emit_subreg_node(inst, dag)
            return

        if inst.opcode == TargetDagOps.INSERT_SUBREG:
            self.emit_subreg_node(inst, dag)
            return

        if inst.opcode == VirtualDagOps.COPY_FROM_REG:
            self.emit_copy_from_reg(inst, dag)
            return

        if inst.opcode == VirtualDagOps.COPY_TO_REG:
            self.emit_copy_to_reg(inst, dag)
            return

        if inst.opcode == TargetDagOps.COPY_TO_REGCLASS:
            self.emit_copy_to_regclass(inst, dag)
            return

        if inst.opcode == TargetDagOps.SUBREG_TO_REG:
            self.emit_subreg_node(inst, dag)
            return

        if inst.opcode == VirtualDagOps.INLINEASM:
            self.emit_inlineasm(inst, dag)
            return
