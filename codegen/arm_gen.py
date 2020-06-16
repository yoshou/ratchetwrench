#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.spec import *
from codegen.mir_emitter import *
from codegen.isel import *
from codegen.arm_def import *
from codegen.matcher import *


class ARMOperandFlag(IntFlag):
    NO_FLAG = auto()
    MO_LO16 = auto()
    MO_HI16 = auto()


def is_null_constant(value):
    return isinstance(value.node, ConstantDagNode) and value.node.is_zero


def is_null_fp_constant(value):
    return isinstance(value.node, ConstantFPDagNode) and value.node.is_zero


class ARMInstructionSelector(InstructionSelector):
    def __init__(self):
        super().__init__()

    def select_setcc(self, node: DagNode, dag: Dag, new_ops):
        condcode = new_ops[0]
        cond = new_ops[1]

        if isinstance(cond.node, DagNode):
            value_ty = node.value_types[0]

            zero = DagValue(dag.add_target_constant_node(value_ty, 0), 0)
            zero = DagValue(dag.add_machine_dag_node(
                ARMMachineOps.MOVi16, [value_ty], zero), 0)
            one = DagValue(dag.add_target_constant_node(value_ty, 1), 0)

            return dag.add_machine_dag_node(ARMMachineOps.MOVCCi, node.value_types, zero, one, condcode, cond)

        raise NotImplementedError()

    def select_callseq_start(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        opt = dag.add_target_constant_node(MachineValueType(ValueType.I32), 0)
        return dag.add_machine_dag_node(ARMMachineOps.ADJCALLSTACKDOWN, node.value_types, in_bytes, out_bytes, DagValue(opt, 0), chain)

    def select_callseq_end(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        glue = self.get_glue(new_ops)

        ops = [in_bytes, out_bytes, chain]
        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(ARMMachineOps.ADJCALLSTACKUP, node.value_types, *ops)

    def get_glue(self, operands):
        for operand in operands:
            if operand.ty == MachineValueType(ValueType.GLUE):
                return operand

        return None

    def select_call(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        target = new_ops[1]
        glue = self.get_glue(new_ops)

        ops = [target, chain]
        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(ARMMachineOps.BL, node.value_types, *ops)

    def select_copy_from_reg(self, node: DagNode, dag: Dag, new_ops):
        return node
        # return dag.add_machine_dag_node(VirtualDagOps.COPY_FROM_REG, node.value_types, *new_ops)

        chain = new_ops[0]
        src = new_ops[1]

        if isinstance(src.node, RegisterDagNode):
            return src.node

        print("select_copy_from_reg")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_copy_to_reg(self, node: DagNode, dag: Dag, new_ops):
        chain = node.operands[0]
        dest = node.operands[1]
        src = node.operands[2]

        glue = self.get_glue(node.operands)

        ops = [chain, dest, src]
        if glue:
            ops.append(glue)

        return dag.add_node(VirtualDagOps.COPY_TO_REG, node.value_types, *ops)

    def select_code(self, node: DagNode, dag: Dag):
        ops_table = [op for op in ARMMachineOps.insts()]

        value = DagValue(node, 0)

        def match_node(inst: MachineInstructionDef):
            for pattern in inst.patterns:
                _, res = pattern.match(None, [value], 0, dag)
                if res:
                    return construct(inst, node, dag, res)

            return None

        for op in ops_table:
            matched = match_node(op)
            if matched:
                return matched

        return None

    def select_frame_index(self, node: DagNode, dag: Dag, new_ops):
        base = DagValue(dag.add_frame_index_node(
            node.value_types[0], node.index, True), 0)

        offset = DagValue(dag.add_target_constant_node(
            node.value_types[0], 0), 0)
        ops = [base, offset]

        return dag.add_machine_dag_node(ARMMachineOps.ADDri, node.value_types, *ops)

    def select_vdup(self, node: DagNode, dag: Dag, new_ops):
        in_type = node.operands[0].ty
        out_type = node.value_types[0]

        hwmode = dag.mfunc.target_info.hwmode
        tys = DPR.get_types(hwmode)

        if in_type in tys and out_type.value_type == ValueType.V4F32:
            lane = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I32), 0), 0)
            return dag.add_machine_dag_node(ARMMachineOps.VDUPLN32q, node.value_types, node.operands[0], lane)

        raise ValueError()

    def select_insert_vector_elt(self, node: DagNode, dag: Dag, new_ops):
        vec = node.operands[0]
        elem = node.operands[1]
        idx = node.operands[2]

        hwmode = dag.mfunc.target_info.hwmode

        if isinstance(idx.node, ConstantDagNode):
            if node.value_types[0].value_type in [ValueType.V2F32, ValueType.V4F32]:
                tys = SPR.get_types(hwmode)
                if elem.ty in tys:
                    idx = DagValue(dag.add_target_constant_node(
                        idx.ty, idx.node.value), 0)
                    subreg_idx = idx.node.value.value
                    if subreg_idx == 0:
                        subreg = ssub_0
                    elif subreg_idx == 1:
                        subreg = ssub_1
                    elif subreg_idx == 2:
                        subreg = ssub_2
                    elif subreg_idx == 3:
                        subreg = ssub_3
                    else:
                        raise ValueError("Invalid index")

                    subreg_id = DagValue(dag.add_target_constant_node(
                        MachineValueType(ValueType.I32), subregs.index(subreg)), 0)
                    return dag.add_node(TargetDagOps.INSERT_SUBREG, [vec.ty], vec, elem, subreg_id)

        # TODO: Neet to implement indexing with variable. A solution is using memory.

        raise NotImplementedError()

    def select_scalar_to_vector(self, node: DagNode, dag: Dag, new_ops):
        elem = node.operands[0]

        if node.value_types[0].value_type == ValueType.V2F32:
            value = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.V2F32), 0), 0)
            subreg = ssub_0

            subreg_id = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I32), subregs.index(subreg)), 0)
            return dag.add_node(TargetDagOps.INSERT_SUBREG, node.value_types, value, elem, subreg_id)

        raise NotImplementedError()

    def select(self, node: DagNode, dag: Dag):
        new_ops = node.operands

        SELECT_TABLE = {
            VirtualDagOps.COPY_FROM_REG: self.select_copy_from_reg,
            VirtualDagOps.COPY_TO_REG: self.select_copy_to_reg,
            VirtualDagOps.CALLSEQ_START: self.select_callseq_start,
            VirtualDagOps.CALLSEQ_END: self.select_callseq_end,
            VirtualDagOps.FRAME_INDEX: self.select_frame_index,
            VirtualDagOps.INSERT_VECTOR_ELT: self.select_insert_vector_elt,
            VirtualDagOps.SCALAR_TO_VECTOR: self.select_scalar_to_vector,
            ARMDagOps.SETCC: self.select_setcc,
            ARMDagOps.CALL: self.select_call,
            ARMDagOps.VDUP: self.select_vdup,
        }

        if isinstance(node.opcode, TargetDagOps):
            return node

        if node.opcode == VirtualDagOps.ENTRY:
            return dag.entry.node
        elif node.opcode == VirtualDagOps.UNDEF:
            return node
        elif node.opcode == VirtualDagOps.TARGET_CONSTANT:
            return node
        elif node.opcode == VirtualDagOps.TARGET_CONSTANT_POOL:
            return node
        elif node.opcode == VirtualDagOps.CONDCODE:
            return node
        elif node.opcode == VirtualDagOps.BASIC_BLOCK:
            return node
        elif node.opcode == VirtualDagOps.REGISTER:
            return node
        elif node.opcode == VirtualDagOps.TARGET_REGISTER:
            return node
        elif node.opcode == VirtualDagOps.EXTERNAL_SYMBOL:
            return dag.add_external_symbol_node(node.value_types[0], node.symbol, True)
        elif node.opcode == VirtualDagOps.MERGE_VALUES:
            return dag.add_node(node.opcode, node.value_types, *new_ops)
        elif node.opcode == VirtualDagOps.TOKEN_FACTOR:
            return dag.add_node(node.opcode, node.value_types, *new_ops)
        elif node.opcode == VirtualDagOps.TARGET_CONSTANT_FP:
            return node
        elif node.opcode == VirtualDagOps.TARGET_GLOBAL_ADDRESS:
            return node
        elif node.opcode == VirtualDagOps.TARGET_FRAME_INDEX:
            return node
        elif node.opcode in SELECT_TABLE:
            select_func = SELECT_TABLE[node.opcode]
            return select_func(node, dag, new_ops)

        matched = self.select_code(node, dag)

        if matched:
            return matched

        raise NotImplementedError(
            "Can't select the instruction: {}".format(node.opcode))


class ArgListEntry:
    def __init__(self, node, ty):
        self.node = node
        self.ty = ty


class CallInfo:
    def __init__(self, dag, chain, ret_ty, target, arg_list):
        self.dag = dag
        self.chain = chain
        self.ret_ty = ret_ty
        self.target = target
        self.arg_list = arg_list


class ARMCallingConv(CallingConv):
    def __init__(self):
        pass

    @property
    def id(self):
        return CallingConvID.C

    def can_lower_return(self, func: Function):
        return_size, align = func.module.data_layout.get_type_size_in_bits(
            func.vty.return_ty)
        return return_size / 8 <= 8

    def lower_return(self, builder: DagBuilder, inst: ReturnInst, g: Dag):
        mfunc = g.mfunc
        calling_conv = mfunc.target_info.get_calling_conv()
        reg_info = mfunc.target_info.get_register_info()
        data_layout = builder.data_layout

        demote_reg = builder.mfunc.func_info.sret_reg
        has_demote_arg = demote_reg is not None

        stack_pop_bytes = builder.get_value(ConstantInt(0, i32))

        if len(inst.operands) > 0:
            return_offsets = []
            return_vts = compute_value_types(
                inst.block.func.return_ty, data_layout, return_offsets)

            returns = []
            offset_in_arg = 0

            # Analyze return value
            for val_idx, vt in enumerate(return_vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                for reg_idx in range(reg_count):
                    flags = CCArgFlags()

                    returns.append(CallingConvReturn(
                        vt, reg_vt, 0, offset_in_arg, flags))

                    offset_in_arg += reg_vt.get_size_in_byte()

            # Apply caling convention
            ccstate = CallingConvState(calling_conv, mfunc)
            ccstate.compute_returns_layout(returns)

            # Handle return values
            ret_parts = []
            ret_value = builder.get_value(inst.rs)
            idx = 0
            for val_idx, vt in enumerate(return_vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                if reg_count > 1:
                    raise NotImplementedError()

                ret_parts.append(ret_value.get_value(idx))

                idx += reg_count

            reg_vals = []
            for idx, ret_val in enumerate(ccstate.values):
                assert(isinstance(ret_val, CCArgReg))
                ret_vt = ret_val.loc_vt

                reg_val = DagValue(
                    g.add_target_register_node(ret_vt, ret_val.reg), 0)
                copy_val = ret_parts[idx]

                chain = DagValue(
                    builder.g.add_copy_to_reg_node(reg_val, copy_val), 0)
                builder.root = chain
                reg_vals.append(reg_val)

            ops = [builder.root, stack_pop_bytes, *reg_vals]
        else:
            ops = [builder.root, stack_pop_bytes]

        if has_demote_arg:
            return_ty = inst.block.func.ty
            vts = compute_value_types(
                return_ty, inst.block.func.module.data_layout)
            assert(len(vts) == 1)
            assert(len(demote_reg) == 1)
            ret_val = DagValue(
                builder.g.add_register_node(vts[0], demote_reg[0]), 0)

            if ret_val.ty == MachineValueType(ValueType.I32):
                ret_reg = R0
            else:
                raise NotImplementedError()

            reg_node = DagValue(
                g.add_target_register_node(ret_val.ty, ret_reg), 0)

            node = g.add_copy_to_reg_node(reg_node, ret_val)
            builder.root = DagValue(node, 0)

            ops = [builder.root, stack_pop_bytes, reg_node]

        node = g.add_node(ARMDagOps.RETURN, [
                          MachineValueType(ValueType.OTHER)], *ops)

        builder.root = DagValue(node, 0)

        return node

    def compute_type_size_aligned(self, ty, data_layout: DataLayout):
        return data_layout.get_type_size_in_bits(ty)

    def lower_call(self, builder: DagBuilder, inst: CallInst, g: Dag):
        mfunc = g.mfunc
        func = inst.callee
        calling_conv = mfunc.target_info.get_calling_conv()
        reg_info = mfunc.target_info.get_register_info()
        data_layout = builder.data_layout

        func_address = builder.get_or_create_global_address(inst.callee, True)

        arg_list = []
        for arg, param in zip(inst.args, func.args):
            arg_list.append(ArgListEntry(builder.get_value(arg), param.ty))

        call_info = CallInfo(g, g.root, inst.ty, func_address, arg_list)

        call_result = self.lower_call_info(call_info)

        g.root = call_info.chain

        return call_result

    def lower_call_info(self, call_info):
        dag = call_info.dag
        mfunc = dag.mfunc
        calling_conv = mfunc.target_info.get_calling_conv()
        reg_info = mfunc.target_info.get_register_info()
        data_layout = dag.data_layout

        func_address = call_info.target
        arg_list = call_info.arg_list
        ret_ty = call_info.ret_ty

        # Handle arguments
        args = []
        for i, arg in enumerate(arg_list):
            vts = compute_value_types(arg.ty, data_layout)
            offset_in_arg = 0

            for val_idx, vt in enumerate(vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                for reg_idx in range(reg_count):
                    flags = CCArgFlags()

                    args.append(CallingConvArg(
                        vt, reg_vt, i, offset_in_arg, flags))

                    offset_in_arg += reg_vt.get_size_in_byte()

        ccstate = CallingConvState(calling_conv, mfunc)
        ccstate.compute_arguments_layout(args)

        # Estimate stack size to call function
        stack_bytes = 0
        arg_align_max = 1
        for arg in arg_list:
            size, align = self.compute_type_size_aligned(arg.ty, data_layout)
            arg_size = int(size / 8)
            arg_align = int(align / 8)
            stack_bytes = align_to(stack_bytes, arg_align)
            stack_bytes += arg_size

            arg_align_max = max(arg_align_max, arg_align)

        stack_bytes = align_to(stack_bytes, arg_align_max)

        chain = call_info.chain

        # Begin of function call sequence
        in_bytes = dag.add_target_constant_node(
            MachineValueType(ValueType.I32), stack_bytes)
        out_bytes = dag.add_target_constant_node(
            MachineValueType(ValueType.I32), 0)

        callseq_start_node = dag.add_node(VirtualDagOps.CALLSEQ_START, [
            MachineValueType(ValueType.OTHER)], chain, DagValue(in_bytes, 0), DagValue(out_bytes, 0))

        chain = DagValue(callseq_start_node, 0)

        # Save arguments
        arg_parts = []
        for arg in arg_list:
            idx = 0
            arg_value = arg.node
            vts = compute_value_types(arg.ty, data_layout)
            for val_idx, vt in enumerate(vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                if reg_count > 1:
                    raise NotImplementedError()

                arg_parts.append(arg_value.get_value(idx))

                idx += reg_count

        reg_vals = []
        for idx, arg_val in enumerate(ccstate.values):
            assert(isinstance(arg_val, CCArgReg))
            reg_val = DagValue(dag.add_target_register_node(
                arg_val.vt, arg_val.reg), 0)
            copy_val = arg_parts[idx]

            if arg_val.loc_info == CCArgLocInfo.Full:
                pass
            elif arg_val.loc_info == CCArgLocInfo.BCvt:
                copy_val = DagValue(dag.add_node(
                    VirtualDagOps.BITCAST, [arg_val.loc_vt], copy_val), 0)
            elif arg_val.loc_info == CCArgLocInfo.Indirect:
                arg_mem_size = arg_val.vt.get_size_in_byte()
                arg_mem_align = int(data_layout.get_pref_type_alignment(
                    arg_val.vt.get_ir_type()) / 8)
                arg_mem_frame_idx = mfunc.frame.create_stack_object(
                    arg_mem_size, arg_mem_align)
                arg_mem_val = DagValue(
                    dag.add_frame_index_node(arg_mem_frame_idx), 0)

                chain = DagValue(dag.add_store_node(
                    chain, arg_mem_val, copy_val), 0)

                copy_val = arg_mem_val
            else:
                raise ValueError()

            operands = [chain, reg_val, copy_val]
            if idx != 0:
                operands.append(copy_to_reg_chain.get_value(1))

            copy_to_reg_chain = DagValue(dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(
                ValueType.OTHER), MachineValueType(ValueType.GLUE)], *operands), 0)

            chain = copy_to_reg_chain
            reg_vals.append(reg_val)

        # Function call
        call_node = dag.add_node(
            ARMDagOps.CALL, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)], chain, func_address, copy_to_reg_chain.get_value(1))

        chain = DagValue(call_node, 0)

        callseq_end_node = dag.add_node(VirtualDagOps.CALLSEQ_END, [
            MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)], chain, DagValue(in_bytes, 0), DagValue(out_bytes, 0), DagValue(call_node, 1))

        chain = DagValue(callseq_end_node, 0)

        # Handle returns
        return_offsets = []
        return_vts = compute_value_types(ret_ty, data_layout, return_offsets)

        returns = []
        offset_in_ret = 0

        for val_idx, vt in enumerate(return_vts):
            reg_vt = reg_info.get_register_type(vt)
            reg_count = reg_info.get_register_count(vt)

            for reg_idx in range(reg_count):
                flags = CCArgFlags()

                returns.append(CallingConvReturn(
                    vt, reg_vt, 0, offset_in_ret, flags))

                offset_in_ret += reg_vt.get_size_in_byte()

        ccstate = CallingConvState(calling_conv, mfunc)
        ccstate.compute_returns_layout(returns)

        glue_val = DagValue(callseq_end_node, 1)

        # Restore return values
        ret_vals = []
        for idx, ret_val in enumerate(ccstate.values):
            assert(isinstance(ret_val, CCArgReg))
            reg = MachineRegister(ret_val.reg)

            reg_node = DagValue(
                dag.add_register_node(ret_val.vt, reg), 0)
            ret_val_node = DagValue(dag.add_node(VirtualDagOps.COPY_FROM_REG, [
                                    ret_val.vt, MachineValueType(ValueType.GLUE)], chain, reg_node, glue_val), 0)

            glue_val = ret_val_node.get_value(1)
            ret_vals.append(ret_val_node)

        ret_parts = []
        idx = 0
        for val_idx, vt in enumerate(return_vts):
            reg_vt = reg_info.get_register_type(vt)
            reg_count = reg_info.get_register_count(vt)

            if reg_count > 1:
                raise NotImplementedError()

            ret_parts.append(ret_vals[idx])

            idx += reg_count

        call_info.chain = chain

        if len(ret_parts) == 0:
            return None

        return dag.add_merge_values(ret_parts)

    def allocate_return_arm_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.F32:
            regs = [S0, S1, S2, S3]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs = [R0, R1, R2, R3]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.V4F32]:
            loc_vt = MachineValueType(ValueType.V2F64)
            loc_info = CCArgLocInfo.BCvt

        if loc_vt.value_type == ValueType.V2F64:
            regs = [Q0, Q1, Q2, Q3]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_return(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        self.allocate_return_arm_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)

    def allocate_argument_arm_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.F32:
            regs = [S0, S1, S2, S3, S4, S5, S6, S7,
                    S8, S9, S10, S11, S12, S13, S14, S15]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs = [R0, R1, R2, R3]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            stack_offset = ccstate.alloc_stack(4, 4)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        if loc_vt.value_type == ValueType.F32:
            stack_offset = ccstate.alloc_stack(4, 4)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        if loc_vt.value_type in [ValueType.V4F32]:
            loc_vt = MachineValueType(ValueType.V2F64)
            loc_info = CCArgLocInfo.BCvt

        if loc_vt.value_type == ValueType.V2F64:
            regs = [Q0, Q1, Q2, Q3]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_argument(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        self.allocate_argument_arm_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)


class ARMTargetInstInfo(TargetInstInfo):
    def __init__(self):
        super().__init__()

    def copy_phys_reg(self, src_reg, dst_reg, kill_src, inst: MachineInstruction):
        assert(isinstance(src_reg, MachineRegister))
        assert(isinstance(dst_reg, MachineRegister))

        if src_reg.spec in GPR.regs and dst_reg.spec in GPR.regs:
            opcode = ARMMachineOps.MOVsi
        elif src_reg.spec in SPR.regs and dst_reg.spec in SPR.regs:
            opcode = ARMMachineOps.VMOVS
        elif src_reg.spec in DPR.regs and dst_reg.spec in DPR.regs:
            opcode = ARMMachineOps.VMOVD
        elif src_reg.spec in QPR.regs and dst_reg.spec in QPR.regs:
            opcode = ARMMachineOps.VORRq
        else:
            raise NotImplementedError(
                "Move instructions support GPR, SPR, DPR or QPR at the present time.")

        copy_inst = MachineInstruction(opcode)

        copy_inst.add_reg(dst_reg, RegState.Define)
        copy_inst.add_reg(src_reg, RegState.Kill if kill_src else RegState.Non)

        if opcode == ARMMachineOps.VORRq:
            copy_inst.add_reg(
                src_reg, RegState.Kill if kill_src else RegState.Non)

        if opcode == ARMMachineOps.MOVsi:
            copy_inst.add_imm(2)

        copy_inst.insert_after(inst)

    def copy_reg_to_stack(self, reg, stack_slot, regclass, inst: MachineInstruction):
        hwmode = inst.mbb.func.target_info.hwmode

        tys = regclass.get_types(hwmode)

        align = int(regclass.align / 8)
        size = tys[0].get_size_in_bits()
        size = int(int((size + 7) / 8))

        noreg = MachineRegister(NOREG)

        def has_reg_regclass(reg, regclass):
            if isinstance(reg, MachineVirtualRegister):
                reg_info = inst.mbb.func.target_info.get_register_info()
                return reg.regclass == regclass or reg_info.is_subclass(regclass, reg.regclass)
            else:
                return reg.spec in regclass.regs

        if size == 1:
            raise NotImplementedError()
        elif size == 4:
            if has_reg_regclass(reg, GPR):
                copy_inst = MachineInstruction(ARMMachineOps.STRi12)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, SPR):
                copy_inst = MachineInstruction(ARMMachineOps.VSTRS)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if has_reg_regclass(reg, DPR):
                copy_inst = MachineInstruction(ARMMachineOps.VSTRD)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
            if has_reg_regclass(reg, QPR):
                mfunc = inst.mbb.func
                regclass = GPR  # TODO
                scratch_reg = mfunc.reg_info.create_virtual_register(regclass)

                addr_inst = MachineInstruction(ARMMachineOps.ADDri)
                addr_inst.add_reg(scratch_reg, RegState.Define)
                addr_inst.add_frame_index(stack_slot)
                addr_inst.add_imm(0)

                addr_inst.insert_before(inst)

                copy_inst = MachineInstruction(ARMMachineOps.VST1q64)
                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(scratch_reg, RegState.Kill)
                copy_inst.add_imm(1)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst.insert_before(inst)
        return copy_inst

    def copy_reg_from_stack(self, reg, stack_slot, regclass, inst: MachineInstruction):
        hwmode = inst.mbb.func.target_info.hwmode

        tys = regclass.get_types(hwmode)

        align = int(regclass.align / 8)
        size = tys[0].get_size_in_bits()
        size = int(int((size + 7) / 8))

        noreg = MachineRegister(NOREG)

        def has_reg_regclass(reg, regclass):
            if isinstance(reg, MachineVirtualRegister):
                reg_info = inst.mbb.func.target_info.get_register_info()
                return reg.regclass == regclass or reg_info.is_subclass(regclass, reg.regclass)
            else:
                return reg.spec in regclass.regs

        if size == 1:
            raise NotImplementedError()
        elif size == 4:
            if has_reg_regclass(reg, GPR):
                copy_inst = MachineInstruction(ARMMachineOps.LDRi12)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, SPR):
                copy_inst = MachineInstruction(ARMMachineOps.VLDRS)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if has_reg_regclass(reg, DPR):
                copy_inst = MachineInstruction(ARMMachineOps.VLDRD)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
            if has_reg_regclass(reg, QPR):
                mfunc = inst.mbb.func
                regclass = GPR  # TODO
                scratch_reg = mfunc.reg_info.create_virtual_register(regclass)

                addr_inst = MachineInstruction(ARMMachineOps.ADDri)
                addr_inst.add_reg(scratch_reg, RegState.Define)
                addr_inst.add_frame_index(stack_slot)
                addr_inst.add_imm(0)

                addr_inst.insert_before(inst)

                copy_inst = MachineInstruction(ARMMachineOps.VLD1q64)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_reg(scratch_reg, RegState.Kill)
                copy_inst.add_imm(1)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst.insert_before(inst)
        return copy_inst

    def calculate_frame_offset(self, func: MachineFunction, idx):
        slot_size = 8
        frame = func.frame
        stack_obj = frame.get_stack_object(idx)
        frame_lowering = func.target_info.get_frame_lowering()
        if idx < 0:
            return stack_obj.offset + frame_lowering.frame_spill_size

        return stack_obj.offset

    def eliminate_frame_index(self, func: MachineFunction, inst: MachineInstruction, idx):
        # Analyze the frame index into a base register and a displacement.
        operand = inst.operands[idx]
        if isinstance(operand, MOFrameIndex):
            base_reg = MachineRegister(R11)
            stack_obj = func.frame.get_stack_object(operand.index)
            offset = self.calculate_frame_offset(func, operand.index)

            if inst.opcode in [ARMMachineOps.VLDRS, ARMMachineOps.VSTRS, ARMMachineOps.VLDRD, ARMMachineOps.VSTRD]:
                assert(offset % 4 == 0)
                offset = offset >> 2

                assert(abs(inst.operands[idx + 1].val + offset) < 256)

            inst.operands[idx] = MOReg(base_reg, RegState.Non)
            inst.operands[idx + 1] = MOImm(inst.operands[idx + 1].val + offset)

            if inst.operands[idx + 1].val < 0:
                if inst.opcode == ARMMachineOps.ADDri:
                    inst.opcode = ARMMachineOps.SUBri
                    inst.operands[idx + 1] = MOImm(-inst.operands[idx + 1].val)
                elif inst.opcode == ARMMachineOps.SUBri:
                    inst.opcode = ARMMachineOps.ADDri
                    inst.operands[idx + 1] = MOImm(-inst.operands[idx + 1].val)
                elif inst.opcode in [
                        ARMMachineOps.STRi12, ARMMachineOps.LDRi12, ARMMachineOps.VSTRS, ARMMachineOps.VLDRS,
                        ARMMachineOps.VSTRD, ARMMachineOps.VLDRD, ARMMachineOps.VST1q64, ARMMachineOps.VLD1q64]:
                    pass
                else:
                    raise ValueError()

    def optimize_compare_inst(self, func: MachineFunction, inst: MachineInstruction):
        # Eliminate destination register.
        reginfo = func.reg_info
        if reginfo.is_use_empty(inst.operands[0].reg):
            pass

    def expand_post_ra_pseudo(self, inst: MachineInstruction):
        if inst.opcode == ARMMachineOps.MOVi32imm:
            dst = inst.operands[0]
            src = inst.operands[1]

            lo_opc = ARMMachineOps.MOVi16
            hi_opc = ARMMachineOps.MOVTi16

            lo_inst = MachineInstruction(lo_opc)
            lo_inst.add_reg(dst.reg, RegState.Define)

            hi_inst = MachineInstruction(hi_opc)
            hi_inst.add_reg(dst.reg, RegState.Define)
            hi_inst.add_reg(dst.reg, RegState.Non)

            if isinstance(src, MOGlobalAddress):
                lo_inst.add_global_address(
                    src.value, target_flags=(src.target_flags | ARMOperandFlag.MO_LO16.value))
                hi_inst.add_global_address(
                    src.value, target_flags=(src.target_flags | ARMOperandFlag.MO_HI16.value))
            else:
                raise ValueError()

            hi_inst.insert_after(inst)
            lo_inst.insert_after(inst)
            inst.remove()

        if inst.opcode == ARMMachineOps.LEApcrel:
            dst = inst.operands[0]
            src = inst.operands[1]

            lo_opc = ARMMachineOps.MOVi16
            hi_opc = ARMMachineOps.MOVTi16

            lo_inst = MachineInstruction(lo_opc)
            lo_inst.add_reg(dst.reg, RegState.Define)

            hi_inst = MachineInstruction(hi_opc)
            hi_inst.add_reg(dst.reg, RegState.Define)
            hi_inst.add_reg(dst.reg, RegState.Non)

            if isinstance(src, MOConstantPoolIndex):
                lo_inst.add_constant_pool_index(
                    src.index, target_flags=(src.target_flags | ARMOperandFlag.MO_LO16.value))
                hi_inst.add_constant_pool_index(
                    src.index, target_flags=(src.target_flags | ARMOperandFlag.MO_HI16.value))
            else:
                raise ValueError()

            hi_inst.insert_after(inst)
            lo_inst.insert_after(inst)
            inst.remove()


def get_super_regs(reg):
    assert(isinstance(reg, MachineRegisterDef))
    regs = MachineRegisterDef.regs

    intersects = {}
    for a_reg in regs:
        intersects[a_reg] = set()

    for a_reg in regs:
        for subreg in a_reg.subregs:
            intersects[subreg].add(a_reg)

    stk = list(intersects[reg])
    supers = set()
    while len(stk) > 0:
        poped = stk.pop()

        if poped in supers:
            continue

        supers.add(poped)

        for super_reg in intersects[poped]:
            stk.append(super_reg)

    return supers


def count_if(values, pred):
    return len([v for v in values if pred(v)])


def find_if(values, pred):
    for i, v in enumerate(values):
        if pred(v):
            return i

    return -1


ARMCC_EQ = 0b0000  # Equal                      Equal
ARMCC_NE = 0b0001  # Not equal                  Not equal, or unordered
ARMCC_HS = 0b0010  # Carry set                  >, ==, or unordered
ARMCC_LO = 0b0011  # Carry clear                Less than
ARMCC_MI = 0b0100  # Minus, negative            Less than
ARMCC_PL = 0b0101  # Plus, positive or zero     >, ==, or unordered
ARMCC_VS = 0b0110  # Overflow                   Unordered
ARMCC_VC = 0b0111  # No overflow                Not unordered
ARMCC_HI = 0b1000  # Unsigned higher            Greater than, or unordered
ARMCC_LS = 0b1001  # Unsigned lower or same     Less than or equal
ARMCC_GE = 0b1010  # Greater than or equal      Greater than or equal
ARMCC_LT = 0b1011  # Less than                  Less than, or unordered
ARMCC_GT = 0b1100  # Greater than               Greater than
ARMCC_LE = 0b1101  # Less than or equal         <, ==, or unordered
ARMCC_AL = 0b1110  # Always (unconditional)     Always (unconditional)


class ARMConstantPoolKind(Enum):
    Value = auto()
    ExtSymbol = auto()
    BlockAddress = auto()
    BasicBlock = auto()


class ARMConstantPoolModifier(Enum):
    Non = auto()
    TLSGlobalDesc = auto()


class ARMConstantPoolConstant(MachineConstantPoolValue):
    def __init__(self, ty, label_id, value, kind, modifier, pc_offset, relative):
        super().__init__(ty)
        self.label_id = label_id
        self.value = value
        self.kind = kind
        self.modifier = modifier
        self.pc_offset = pc_offset
        self.relative = relative

    def __hash__(self):
        return hash((self.value, self.kind, self.modifier, self.pc_offset, self.relative))

    def __eq__(self, other):
        if not isinstance(other, ARMConstantPoolConstant):
            return False

        eq1 = self.value == other.value and self.kind == other.kind and self.modifier == other.modifier

        return eq1 and self.pc_offset == other.pc_offset and self.relative == other.relative

    def __ne__(self, other):
        return not self.__eq__(other)


class ARMTargetLowering(TargetLowering):
    def __init__(self):
        super().__init__()

        self.reg_type_for_vt = {MachineValueType(
            e): MachineValueType(e) for e in ValueType}

        self.reg_count_for_vt = {MachineValueType(e): 1 for e in ValueType}

    def get_register_type(self, vt):
        if vt in self.reg_type_for_vt:
            return self.reg_type_for_vt[vt]

        raise NotImplementedError()

    def get_register_count(self, vt):
        if vt in self.reg_count_for_vt:
            return self.reg_count_for_vt[vt]

        raise NotImplementedError()

    def lower_setcc(self, node: DagNode, dag: Dag):
        op1 = node.operands[0]
        op2 = node.operands[1]
        cond = node.operands[2]

        is_fcmp = op1.node.value_types[0].value_type in [
            ValueType.F32, ValueType.F64]

        def compute_condcode(cond):
            ty = MachineValueType(ValueType.I8)
            swap = False
            cond = cond.node.cond

            if is_fcmp:
                if cond in [CondCode.SETEQ, CondCode.SETOEQ]:
                    node = dag.add_target_constant_node(ty, ARMCC_EQ)
                elif cond in [CondCode.SETGT, CondCode.SETOGT]:
                    node = dag.add_target_constant_node(ty, ARMCC_GT)
                elif cond in [CondCode.SETLT, CondCode.SETOLT]:
                    node = dag.add_target_constant_node(ty, ARMCC_GE)
                    swap = True
                elif cond in [CondCode.SETGE, CondCode.SETOGE]:
                    node = dag.add_target_constant_node(ty, ARMCC_GE)
                else:
                    raise NotImplementedError()
            else:
                if cond == CondCode.SETEQ:
                    node = dag.add_target_constant_node(ty, ARMCC_EQ)
                elif cond == CondCode.SETNE:
                    node = dag.add_target_constant_node(ty, ARMCC_NE)
                elif cond == CondCode.SETLT:
                    node = dag.add_target_constant_node(ty, ARMCC_LT)
                elif cond == CondCode.SETGT:
                    node = dag.add_target_constant_node(ty, ARMCC_GT)
                elif cond == CondCode.SETLE:
                    node = dag.add_target_constant_node(ty, ARMCC_LE)
                elif cond == CondCode.SETGE:
                    node = dag.add_target_constant_node(ty, ARMCC_GE)
                elif cond == CondCode.SETULT:
                    node = dag.add_target_constant_node(ty, 0b0011)
                elif cond == CondCode.SETUGT:
                    node = dag.add_target_constant_node(ty, 0b1000)
                elif cond == CondCode.SETULE:
                    node = dag.add_target_constant_node(ty, 0b1001)
                elif cond == CondCode.SETUGE:
                    node = dag.add_target_constant_node(ty, 0b0010)
                else:
                    raise NotImplementedError()

            return node, swap

        condcode, swap = compute_condcode(cond)
        if swap:
            op1, op2 = op2, op1

        if is_fcmp:
            cmp_node = DagValue(dag.add_node(ARMDagOps.CMPFP,
                                             [MachineValueType(ValueType.GLUE)], op1, op2), 0)
            cmp_node = DagValue(dag.add_node(ARMDagOps.FMSTAT,
                                             [MachineValueType(ValueType.GLUE)], cmp_node), 0)
        else:
            cmp_node = DagValue(dag.add_node(ARMDagOps.CMP,
                                             [MachineValueType(ValueType.GLUE)], op1, op2), 0)

        # operand 1 is eflags.
        setcc_node = dag.add_node(ARMDagOps.SETCC, node.value_types,
                                  DagValue(condcode, 0), cmp_node)

        return setcc_node

    def lower_brcond(self, node: DagNode, dag: Dag):
        chain = node.operands[0]
        cond = node.operands[1]
        dest = node.operands[2]

        if cond.node.opcode == VirtualDagOps.SETCC:
            cond = DagValue(self.lower_setcc(cond.node, dag), 0)

        if cond.node.opcode == ARMDagOps.SETCC:
            condcode = cond.node.operands[0]
            glue = cond.node.operands[1]
            cond = DagValue(dag.add_target_register_node(
                MachineValueType(ValueType.I32), CPSR), 0)

            return dag.add_node(ARMDagOps.BRCOND, node.value_types, chain, dest, condcode, cond, glue)
        else:
            if cond.ty == MachineValueType(ValueType.I1):
                cond = DagValue(dag.add_node(VirtualDagOps.ZERO_EXTEND, [
                                MachineValueType(ValueType.I32)], cond), 0)

            one = DagValue(dag.add_constant_node(cond.ty, 1), 0)
            condcode = DagValue(dag.add_condition_code_node(CondCode.SETEQ), 0)
            cond = DagValue(dag.add_node(VirtualDagOps.SETCC, [
                            MachineValueType(ValueType.I1)], cond, one, condcode), 0)
            cond = DagValue(self.lower_setcc(cond.node, dag), 0)

            condcode = cond.node.operands[0]
            glue = cond.node.operands[1]
            cond = DagValue(dag.add_target_register_node(
                MachineValueType(ValueType.I32), CPSR), 0)

            return dag.add_node(ARMDagOps.BRCOND, node.value_types, chain, dest, condcode, cond, glue)

    def lower_global_address(self, node: DagNode, dag: Dag):
        target_address = dag.add_global_address_node(
            node.value_types[0], node.value, True)
        return dag.add_node(ARMDagOps.WRAPPER, node.value_types, DagValue(target_address, 0))

    def lower_global_tls_address(self, node: DagNode, dag: Dag):
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)
        global_value = node.value

        if global_value.thread_local == ThreadLocalMode.GeneralDynamicTLSModel:
            pc_label_id = dag.mfunc.func_info.create_pic_label_id()

            cp_value = ARMConstantPoolConstant(
                global_value.ty, pc_label_id, global_value, ARMConstantPoolKind.Value, ARMConstantPoolModifier.TLSGlobalDesc, 8, True)

            argument = DagValue(
                dag.add_constant_pool_node(ptr_ty, cp_value, True, 4), 0)

            argument = DagValue(dag.add_node(
                ARMDagOps.WRAPPER, [MachineValueType(ValueType.I32)], argument), 0)

            argument = DagValue(dag.add_load_node(
                ptr_ty, dag.entry, argument, False), 0)

            chain = argument.get_value(1)

            pic_label = DagValue(dag.add_target_constant_node(
                ptr_ty, ConstantInt(pc_label_id, i32)), 0)

            argument = DagValue(dag.add_node(ARMDagOps.PIC_ADD,
                                             [ptr_ty], argument, pic_label), 0)

            arg_list = []
            arg_list.append(ArgListEntry(argument, i32))

            tls_get_addr_func = DagValue(
                dag.add_external_symbol_node(ptr_ty, "__tls_get_addr"), 0)

            calling_conv = dag.mfunc.target_info.get_calling_conv()

            tls_addr = calling_conv.lower_call_info(
                CallInfo(dag, chain, i32, tls_get_addr_func, arg_list))

            return tls_addr.node

        raise ValueError("Not supporing TLS model.")

    def get_pointer_type(self, data_layout, addr_space=0):
        return get_int_value_type(data_layout.get_pointer_size_in_bits(addr_space))

    def get_frame_index_type(self, data_layout):
        return get_int_value_type(data_layout.get_pointer_size_in_bits(0))

    def lower_constant_fp(self, node: DagNode, dag: Dag):
        assert(isinstance(node, ConstantFPDagNode))
        data_layout = dag.mfunc.func_info.func.module.data_layout

        constant_pool = dag.add_constant_pool_node(
            self.get_pointer_type(data_layout), node.value, False)
        return dag.add_load_node(node.value_types[0], dag.entry, DagValue(constant_pool, 0), False)

    def lower_constant_pool(self, node: DagNode, dag: Dag):
        assert(isinstance(node, ConstantPoolDagNode))
        data_layout = dag.mfunc.func_info.func.module.data_layout

        target_constant_pool = dag.add_constant_pool_node(
            self.get_pointer_type(data_layout), node.value, True)
        return dag.add_node(ARMDagOps.WRAPPER, node.value_types, DagValue(target_constant_pool, 0))

    def lower_build_vector(self, node: DagNode, dag: Dag):
        assert(node.opcode == VirtualDagOps.BUILD_VECTOR)

        is_one_val = True
        for idx in range(1, len(node.operands)):
            if node.operands[0].node != node.operands[idx].node or node.operands[0].index != node.operands[idx].index:
                is_one_val = False
                break

        if is_one_val:
            if node.value_types[0] == MachineValueType(ValueType.V4F32):
                vec = DagValue(dag.add_node(VirtualDagOps.SCALAR_TO_VECTOR, [
                               MachineValueType(ValueType.V2F32)], node.operands[0]), 0)
                lane = DagValue(dag.add_target_constant_node(
                    MachineValueType(ValueType.I32), 0), 0)
                return dag.add_node(ARMDagOps.VDUP, node.value_types, vec, lane)

        operands = []
        for operand in node.operands:
            target_constant_fp = dag.add_target_constant_fp_node(
                operand.ty, operand.node.value)
            operands.append(DagValue(target_constant_fp, 0))

        assert(len(operands) > 0)

        return dag.add_node(VirtualDagOps.BUILD_VECTOR, node.value_types, *operands)

    def lower_sub(self, node: DagNode, dag: Dag):
        return dag.add_node(ARMDagOps.SUB, node.value_types, *node.operands)

    def lower_bitcast(self, node: DagNode, dag: Dag):
        hwmode = dag.mfunc.target_info.hwmode

        tys = QPR.get_types(hwmode)

        if node.operands[0].ty in tys:
            if node.value_types[0] in tys:
                return dag.add_node(VirtualDagOps.OR, node.value_types, node.operands[0], node.operands[0])

        return node

    def lower(self, node: DagNode, dag: Dag):
        if node.opcode == VirtualDagOps.ENTRY:
            return dag.entry.node
        if node.opcode == VirtualDagOps.BRCOND:
            return self.lower_brcond(node, dag)
        elif node.opcode == VirtualDagOps.SETCC:
            return self.lower_setcc(node, dag)
        elif node.opcode == VirtualDagOps.SUB:
            return self.lower_sub(node, dag)
        elif node.opcode == VirtualDagOps.GLOBAL_ADDRESS:
            return self.lower_global_address(node, dag)
        elif node.opcode == VirtualDagOps.CONSTANT_FP:
            return self.lower_constant_fp(node, dag)
        elif node.opcode == VirtualDagOps.TARGET_CONSTANT_FP:
            return self.lower_constant_fp(node, dag)
        elif node.opcode == VirtualDagOps.CONSTANT_POOL:
            return self.lower_constant_pool(node, dag)
        elif node.opcode == VirtualDagOps.BUILD_VECTOR:
            return self.lower_build_vector(node, dag)
        elif node.opcode == VirtualDagOps.BITCAST:
            return self.lower_bitcast(node, dag)
        elif node.opcode == VirtualDagOps.GLOBAL_TLS_ADDRESS:
            return self.lower_global_tls_address(node, dag)
        else:
            return node

    def lower_arguments(self, func: Function, builder: DagBuilder):
        arg_load_chains = []
        chain = builder.root

        mfunc = builder.mfunc
        calling_conv = mfunc.target_info.get_calling_conv()
        reg_info = mfunc.target_info.get_register_info()
        data_layout = func.module.data_layout

        target_lowering = mfunc.target_info.get_lowering()

        ptr_ty = target_lowering.get_frame_index_type(data_layout)

        args = []
        for i, arg in enumerate(func.args):
            vts = compute_value_types(arg.ty, data_layout)
            offset_in_arg = 0

            for val_idx, vt in enumerate(vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                for reg_idx in range(reg_count):
                    flags = CCArgFlags()

                    args.append(CallingConvArg(
                        vt, reg_vt, i, offset_in_arg, flags))

                    offset_in_arg += reg_vt.get_size_in_byte()

        ccstate = CallingConvState(calling_conv, mfunc)
        ccstate.compute_arguments_layout(args)

        arg_vals = []
        for arg_val in ccstate.values:
            arg_vt = arg_val.loc_vt
            loc_info = arg_val.loc_info
            if isinstance(arg_val, CCArgReg):
                if arg_vt.value_type == ValueType.I8:
                    regclass = GPRwoPC
                elif arg_vt.value_type == ValueType.I16:
                    regclass = GPRwoPC
                elif arg_vt.value_type == ValueType.I32:
                    regclass = GPRwoPC
                elif arg_vt.value_type == ValueType.I64:
                    raise ValueError()
                elif arg_vt.value_type == ValueType.F32:
                    regclass = SPR
                elif arg_vt.value_type == ValueType.F64:
                    regclass = DPR
                elif arg_vt.value_type == ValueType.V2F64:
                    regclass = QPR
                else:
                    raise ValueError()

                reg = mfunc.reg_info.create_virtual_register(regclass)

                mfunc.reg_info.add_live_in(MachineRegister(arg_val.reg), reg)

                reg_node = DagValue(
                    builder.g.add_register_node(arg_vt, reg), 0)

                arg_val_node = DagValue(
                    builder.g.add_copy_from_reg_node(arg_vt, reg_node), 0)

                if loc_info == CCArgLocInfo.Full:
                    pass
                elif loc_info == CCArgLocInfo.BCvt:
                    arg_val_node = DagValue(
                        builder.g.add_node(VirtualDagOps.BITCAST, [arg_val.vt], arg_val_node), 0)
                else:
                    raise ValueError()
            else:
                assert(isinstance(arg_val, CCArgMem))

                size = arg_vt.get_size_in_byte()
                offset = arg_val.offset

                frame_idx = builder.mfunc.frame.create_fixed_stack_object(
                    size, offset)

                frame_idx_node = DagValue(
                    builder.g.add_frame_index_node(ptr_ty, frame_idx), 0)

                arg_val_node = DagValue(builder.g.add_load_node(
                    arg_vt, builder.root, frame_idx_node, False), 0)

            if arg_val.loc_info == CCArgLocInfo.Indirect:
                arg_val_node = DagValue(builder.g.add_load_node(
                    arg_val.vt, builder.root, arg_val_node, False), 0)

            arg_vals.append(arg_val_node)

        idx = 0
        for i, arg in enumerate(func.args):
            vts = compute_value_types(arg.ty, data_layout)
            offset_in_arg = 0

            arg_parts = []

            for val_idx, vt in enumerate(vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                if reg_count > 1:
                    raise NotImplementedError()

                arg_parts.append(arg_vals[idx])

                idx += reg_count

            val = builder.g.add_merge_values(arg_parts)

            if val.node.opcode == VirtualDagOps.COPY_FROM_REG:
                reg = val.node.operands[1].node.reg
                if isinstance(reg, MachineVirtualRegister):
                    builder.func_info.reg_value_map[arg] = [reg]
            else:
                reg_info = builder.reg_info

                for ty, arg_part in zip(vts, arg_parts):
                    reg_vt = reg_info.get_register_type(ty)
                    reg_count = reg_info.get_register_count(ty)

                    regs = []
                    reg_vals = []
                    for idx in range(reg_count):
                        vreg = target_lowering.get_machine_vreg(
                            reg_vt)
                        reg = builder.mfunc.reg_info.create_virtual_register(
                            vreg)
                        regs.append(reg)
                        reg_vals.append(
                            DagValue(builder.g.add_register_node(reg_vt, reg), 0))

                    chain = get_copy_to_parts(
                        arg_part, reg_vals, reg_vt, chain, builder.g)

                builder.func_info.reg_value_map[arg] = regs

            builder.set_inst_value(arg, val)

        has_demote_arg = len(func.args) > 0 and func.args[0].has_attribute(
            AttributeKind.StructRet)

        if has_demote_arg:
            demote_arg = func.args[0]
            mfunc.func_info.sret_reg = mfunc.func_info.reg_value_map[demote_arg]
            pass
        else:
            mfunc.func_info.sret_reg = None

        # builder.root = DagValue(DagNode(VirtualDagOps.TOKEN_FACTOR, [
        #     MachineValueType(ValueType.OTHER)], arg_load_chains), 0)

    def is_frame_op(self, inst):
        if inst.opcode == ARMMachineOps.ADJCALLSTACKDOWN:
            return True
        if inst.opcode == ARMMachineOps.ADJCALLSTACKUP:
            return True

        return False

    def lower_prolog(self, func: MachineFunction, bb: MachineBasicBlock):
        inst_info = func.target_info.get_inst_info()
        frame_info = func.target_info.get_frame_lowering()
        reg_info = func.target_info.get_register_info()
        data_layout = func.func_info.func.module.data_layout

        front_inst = bb.insts[0]

        push_fp_lr_inst = MachineInstruction(ARMMachineOps.STMDB_UPD)
        push_fp_lr_inst.add_reg(MachineRegister(SP), RegState.Define)
        push_fp_lr_inst.add_reg(MachineRegister(SP), RegState.Non)
        push_fp_lr_inst.add_reg(MachineRegister(R11), RegState.Non)
        push_fp_lr_inst.add_reg(MachineRegister(LR), RegState.Non)

        push_fp_lr_inst.insert_before(front_inst)

        mov_esp_inst = MachineInstruction(ARMMachineOps.MOVr)
        mov_esp_inst.add_reg(MachineRegister(R11), RegState.Define)  # To
        mov_esp_inst.add_reg(MachineRegister(SP), RegState.Non)  # From

        mov_esp_inst.insert_before(front_inst)

        stack_size = func.frame.estimate_stack_size(
            ARMMachineOps.ADJCALLSTACKDOWN, ARMMachineOps.ADJCALLSTACKUP)

        max_align = max(func.frame.max_alignment, func.frame.stack_alignment)
        stack_size = int(
            int((stack_size + max_align - 1) / max_align) * max_align)

        assert(get_mod_imm(stack_size) != -1)

        sub_sp_inst = MachineInstruction(ARMMachineOps.SUBri)
        sub_sp_inst.add_reg(MachineRegister(SP), RegState.Define)
        sub_sp_inst.add_reg(MachineRegister(SP), RegState.Non)
        sub_sp_inst.add_imm(stack_size)

        sub_sp_inst.insert_before(front_inst)

        for cs_info in func.frame.calee_save_info:
            reg = cs_info.reg
            regclass = reg_info.get_regclass_from_reg(reg)
            frame_idx = cs_info.frame_idx

            inst_info.copy_reg_to_stack(MachineRegister(
                reg), frame_idx, regclass, front_inst)

    def lower_epilog(self, func: MachineFunction, bb: MachineBasicBlock):
        inst_info = func.target_info.get_inst_info()
        reg_info = func.target_info.get_register_info()
        data_layout = func.func_info.func.module.data_layout

        front_inst = bb.insts[-1]

        for cs_info in func.frame.calee_save_info:
            reg = cs_info.reg
            regclass = reg_info.get_regclass_from_reg(reg)
            frame_idx = cs_info.frame_idx

            inst_info.copy_reg_from_stack(MachineRegister(
                reg), frame_idx, regclass, front_inst)

        restore_esp_inst = MachineInstruction(ARMMachineOps.MOVr)
        restore_esp_inst.add_reg(MachineRegister(SP), RegState.Define)  # To
        restore_esp_inst.add_reg(MachineRegister(R11), RegState.Non)  # From

        restore_esp_inst.insert_before(front_inst)

        pop_fp_lr_inst = MachineInstruction(ARMMachineOps.LDMIA_UPD)
        pop_fp_lr_inst.add_reg(MachineRegister(SP), RegState.Define)
        pop_fp_lr_inst.add_reg(MachineRegister(SP), RegState.Non)
        pop_fp_lr_inst.add_reg(MachineRegister(R11), RegState.Non)
        pop_fp_lr_inst.add_reg(MachineRegister(LR), RegState.Non)

        pop_fp_lr_inst.insert_before(front_inst)

    def eliminate_call_frame_pseudo_inst(self, func, inst: MachineInstruction):
        inst.remove()

    def get_machine_vreg(self, ty: MachineValueType):
        if ty.value_type == ValueType.I8:
            return GPR
        elif ty.value_type == ValueType.I16:
            return GPR
        elif ty.value_type == ValueType.I32:
            return GPR
        elif ty.value_type == ValueType.I64:
            pass
        elif ty.value_type == ValueType.F32:
            return SPR
        elif ty.value_type == ValueType.F64:
            return DPR
        elif ty.value_type == ValueType.V4F32:
            return QPR

        raise NotImplementedError()

    def lower_optimal_memory_op(self, size, src_op, dst_op, src_align, dst_align, builder: DagBuilder):
        chain = builder.root
        is_volatile = False

        offset = 0
        chains = []
        while offset < size:
            copy_size = min(4, size - offset)
            copy_ty = MachineValueType(ValueType.I32)

            if offset != 0:
                src_ty = src_op.ty
                size_node = DagValue(
                    builder.g.add_target_constant_node(src_ty, offset), 0)
                src_ptr = DagValue(builder.g.add_node(
                    VirtualDagOps.ADD, [src_ty], src_op, size_node), 0)
            else:
                src_ptr = src_op

            if offset != 0:
                dst_ty = dst_op.ty
                size_node = DagValue(
                    builder.g.add_target_constant_node(dst_ty, offset), 0)
                dst_ptr = DagValue(builder.g.add_node(
                    VirtualDagOps.ADD, [dst_ty], dst_op, size_node), 0)
            else:
                dst_ptr = dst_op

            load_op = builder.g.add_load_node(
                copy_ty, chain, src_ptr, is_volatile)
            store_op = builder.g.add_store_node(
                chain, dst_ptr, DagValue(load_op, 0))

            chains.extend([DagValue(store_op, 0)])

            offset += copy_size

        builder.root = DagValue(builder.g.add_node(VirtualDagOps.TOKEN_FACTOR, [
            MachineValueType(ValueType.OTHER)], *chains), 0)

    def get_setcc_result_type_for_vt(self, vt):
        return MachineValueType(ValueType.I32)


class ARMTargetRegisterInfo(TargetRegisterInfo):
    def __init__(self, target_info):
        super().__init__()

        self.target_info = target_info

    def get_reserved_regs(self):
        reserved = []
        reserved.extend([R11, R12, SP, LR, PC])

        return reserved

    @property
    def allocatable_regs(self):
        regs = set()
        regs |= set(GPR.regs)
        regs |= set(SPR.regs)
        regs |= set(DPR.regs)
        regs |= set(QPR.regs)

        return regs

    def get_callee_saved_regs(self):
        callee_save_regs = [R4, R5, R6, R7, R8, R9,
                            R10, D8, D9, D10, D11, D12, D13, D14, D15]

        return callee_save_regs

    def get_callee_clobbered_regs(self):
        regs = [R0, R1, R2, R3,
                D0, D1, D2, D3, D4, D5, D6, D7]

        return regs

    def get_ordered_regs(self, regclass):
        reserved_regs = self.get_reserved_regs()

        free_regs = set(regclass.regs) - set(reserved_regs)

        return [reg for reg in regclass.regs if reg in free_regs]

    def get_regclass_for_vt(self, vt):
        hwmode = self.target_info.hwmode
        for regclass in arm_regclasses:
            tys = regclass.get_types(hwmode)
            if vt in tys:
                return regclass

        raise ValueError("Could not find the register class.")


class ARMFrameLowering(TargetFrameLowering):
    def __init__(self, alignment):
        super().__init__(alignment)

        self.frame_spill_size = 8

    @property
    def stack_grows_direction(self):
        return StackGrowsDirection.Down


class ARMLegalizer(Legalizer):
    def __init__(self):
        super().__init__()

    def get_legalized_op(self, operand, legalized):
        if operand.node not in legalized:
            return operand

        legalized_node = legalized[operand.node]

        if isinstance(legalized_node, (list, tuple)):
            return [DagValue(n, operand.index) for n in legalized_node]

        return DagValue(legalized_node, operand.index)

    def promote_integer_result_setcc(self, node, dag, legalized):
        target_lowering = dag.mfunc.target_info.get_lowering()
        setcc_ty = target_lowering.get_setcc_result_type_for_vt(
            node.value_types[0])

        return dag.add_node(node.opcode, [setcc_ty], *node.operands)

    def promote_integer_result_bin(self, node, dag, legalized):
        lhs = self.get_legalized_op(node.operands[0], legalized)
        rhs = self.get_legalized_op(node.operands[1], legalized)

        return dag.add_node(node.opcode, [lhs.ty], lhs, rhs)

    def promote_integer_result(self, node, dag, legalized):
        if node.opcode == VirtualDagOps.SETCC:
            return self.promote_integer_result_setcc(node, dag, legalized)
        elif node.opcode in [VirtualDagOps.AND, VirtualDagOps.OR]:
            return self.promote_integer_result_bin(node, dag, legalized)
        elif node.opcode in [VirtualDagOps.LOAD]:
            return dag.add_node(node.opcode, [MachineValueType(ValueType.I32)], *node.operands)
        else:
            raise ValueError("No method to promote.")

    def legalize_node_result(self, node: DagNode, dag: Dag, legalized):
        for vt in node.value_types:
            if vt.value_type == ValueType.I1:
                return self.promote_integer_result(node, dag, legalized)

        return node

    def split_vector_result_build_vec(self, node, dag, legalized):
        return tuple([operand.node for operand in node.operands])

    def split_vector_result_bin(self, node, dag, legalized):
        ops_lhs = legalized[node.operands[0].node]
        ops_rhs = legalized[node.operands[1].node]

        assert(len(ops_lhs) == len(ops_rhs))

        values = []
        for lhs, rhs in zip(ops_lhs, ops_rhs):
            lhs_val = DagValue(lhs, node.operands[0].index)
            rhs_val = DagValue(rhs, node.operands[1].index)

            values.append(dag.add_node(
                node.opcode, [rhs_val.ty], lhs_val, rhs_val))

        return tuple(values)

    def split_value_type(self, dag: Dag, vt):
        if vt.value_type == ValueType.V4F32:
            return [vt.get_vector_elem_type()] * vt.get_num_vector_elems()

        raise NotImplementedError()

    def split_vector_result_load(self, node, dag, legalized):
        ops_chain = DagValue(
            legalized[node.operands[0].node], node.operands[0].index)
        ops_ptr = DagValue(
            legalized[node.operands[1].node], node.operands[1].index)

        vts = self.split_value_type(dag, node.value_types[0])

        ofs = 0
        values = []
        for vt in vts:
            elem_size = vt.get_size_in_byte()

            if ofs != 0:
                offset = DagValue(dag.add_constant_node(ops_ptr.ty, ofs), 0)
                addr = DagValue(dag.add_node(VirtualDagOps.ADD, [
                    ops_ptr.ty], ops_ptr, offset), 0)
            else:
                addr = ops_ptr

            value = dag.add_load_node(vt, ops_chain, addr, False)
            values.append(value)

            ofs += elem_size

        return values

    def split_vector_result_ins_vec_elt(self, node, dag, legalized):
        vec_op = node.operands[0]
        ins_elem_op = node.operands[1]
        index_op = node.operands[2]

        assert(index_op.node.opcode in [
               VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT])
        index = index_op.node.value.value

        values = []
        legalized_vec_op = legalized[vec_op.node]

        for i, elem in enumerate(legalized_vec_op):
            if i == index:
                values.append(ins_elem_op.node)
            else:
                values.append(elem)

        return values

    def split_vector_result(self, node, dag, legalized):
        if node.opcode == VirtualDagOps.BUILD_VECTOR:
            return self.split_vector_result_build_vec(node, dag, legalized)
        if node.opcode in [VirtualDagOps.FADD, VirtualDagOps.FSUB, VirtualDagOps.FMUL, VirtualDagOps.FDIV]:
            return self.split_vector_result_bin(node, dag, legalized)
        if node.opcode == VirtualDagOps.LOAD:
            return self.split_vector_result_load(node, dag, legalized)
        if node.opcode == VirtualDagOps.INSERT_VECTOR_ELT:
            return self.split_vector_result_ins_vec_elt(node, dag, legalized)

        return node

    def split_vector_operand_store(self, node, dag: Dag, legalized):

        ops_chain = DagValue(
            legalized[node.operands[0].node], node.operands[0].index)
        ops_src_vec = [DagValue(n, node.operands[1].index)
                       for n in legalized[node.operands[1].node]]
        ops_ptr = DagValue(
            legalized[node.operands[2].node], node.operands[2].index)

        vts = self.split_value_type(dag, node.operands[1].node.value_types[0])

        ofs = 0
        chains = []
        for idx, vt in enumerate(vts):
            elem_size = vt.get_size_in_byte()

            if ofs != 0:
                offset = DagValue(dag.add_constant_node(ops_ptr.ty, ofs), 0)
                addr = DagValue(dag.add_node(VirtualDagOps.ADD, [
                    ops_ptr.ty], ops_ptr, offset), 0)
            else:
                addr = ops_ptr

            ops_src = ops_src_vec[idx]

            chain = DagValue(dag.add_store_node(
                ops_chain, addr, ops_src, False), 0)
            chains.append(chain)

            ofs += elem_size

        return dag.add_node(VirtualDagOps.TOKEN_FACTOR, [MachineValueType(ValueType.OTHER)], *chains)

    def split_vector_operand_ext_vec_elt(self, node, dag: Dag, legalized):
        vec_op = node.operands[0]
        index_op = node.operands[1]

        assert(index_op.node.opcode == VirtualDagOps.TARGET_CONSTANT)
        index = index_op.node.value.value

        legalized_vec_op = legalized[vec_op.node]

        return legalized_vec_op[index]

    def promote_integer_operand_brcond(self, node, dag: Dag, legalized):
        chain_op = node.operands[0]
        cond_op = node.operands[1]
        dst_op = node.operands[2]

        cond_op = DagValue(legalized[cond_op.node], cond_op.index)

        return dag.add_node(VirtualDagOps.BRCOND, node.value_types, chain_op, cond_op, dst_op)

    def legalize_node_operand(self, node, i, dag: Dag, legalized):
        operand = node.operands[i]
        vt = operand.ty

        if vt.value_type == ValueType.I1:
            if node.opcode == VirtualDagOps.BRCOND:
                return self.promote_integer_operand_brcond(
                    node, dag, legalized)

        return None


class ARMTargetInfo(TargetInfo):
    def __init__(self, triple):
        super().__init__(triple)

        self._inst_info = ARMTargetInstInfo()
        self._lowering = ARMTargetLowering()
        self._reg_info = ARMTargetRegisterInfo(self)
        self._calling_conv = ARMCallingConv()
        self._isel = ARMInstructionSelector()
        self._legalizer = ARMLegalizer()
        self._frame_lowering = ARMFrameLowering(16)

    def get_inst_info(self) -> TargetInstInfo:
        return self._inst_info

    def get_lowering(self) -> TargetLowering:
        return self._lowering

    def get_register_info(self) -> TargetRegisterInfo:
        return self._reg_info

    def get_calling_conv(self) -> CallingConv:
        return self._calling_conv

    def get_instruction_selector(self) -> InstructionSelector:
        return self._isel

    def get_legalizer(self) -> Legalizer:
        return self._legalizer

    def get_frame_lowering(self) -> TargetFrameLowering:
        return self._frame_lowering

    @property
    def hwmode(self) -> MachineHWMode:
        if self.triple.arch == ArchType.ARM:
            return ARM

        raise ValueError("Invalid arch type")


class ARMTargetMachine:
    def __init__(self, triple):
        self.triple = triple

    def get_target_info(self, func: Function):
        return ARMTargetInfo(self.triple)

    def add_mc_emit_passes(self, pass_manager, mccontext, output, is_asm):
        from codegen.arm_asm_printer import ARMAsmInfo, MCAsmStream, ARMCodeEmitter, ARMAsmBackend, ARMAsmPrinter
        from codegen.coff import WinCOFFObjectWriter, WinCOFFObjectStream
        from codegen.elf import ELFObjectStream, ELFObjectWriter, ARMELFObjectWriter
        from codegen.arm_constant_island import ARMConstantIsland

        pass_manager.passes.append(ARMConstantIsland())

        objformat = self.triple.objformat

        mccontext.asm_info = ARMAsmInfo()
        if is_asm:
            raise NotImplementedError()
        else:
            emitter = ARMCodeEmitter(mccontext)
            backend = ARMAsmBackend()

            if objformat == ObjectFormatType.ELF:
                target_writer = ARMELFObjectWriter()
                writer = ELFObjectWriter(output, target_writer)
                stream = ELFObjectStream(mccontext, backend, writer, emitter)

        pass_manager.passes.append(ARMAsmPrinter(stream))
