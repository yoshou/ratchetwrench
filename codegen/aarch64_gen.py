#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.spec import *
from codegen.mir_emitter import *
from codegen.isel import *
from codegen.aarch64_def import *
from codegen.matcher import *


class AArch64OperandFlag(IntFlag):
    NO_FLAG = auto()
    MO_PAGE = auto()
    MO_PAGEOFF = auto()
    MO_NC = auto()
    MO_TLS = auto()


def is_null_constant(value):
    return isinstance(value.node, ConstantDagNode) and value.node.is_zero


def is_null_fp_constant(value):
    return isinstance(value.node, ConstantFPDagNode) and value.node.is_zero


class AArch64InstructionSelector(InstructionSelector):
    def __init__(self):
        super().__init__()

    def select_callseq_start(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        opt = dag.add_target_constant_node(MachineValueType(ValueType.I32), 0)
        return dag.add_machine_dag_node(AArch64MachineOps.ADJCALLSTACKDOWN, node.value_types, in_bytes, out_bytes, DagValue(opt, 0), chain)

    def select_callseq_end(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        glue = self.get_glue(new_ops)

        ops = [in_bytes, out_bytes, chain]
        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(AArch64MachineOps.ADJCALLSTACKUP, node.value_types, *ops)

    def get_glue(self, operands):
        for operand in operands:
            if operand.ty == MachineValueType(ValueType.GLUE):
                return operand

        return None

    def select_call(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        target = new_ops[1]
        glue = self.get_glue(new_ops)

        ops = [target]

        i = 2
        while i < len(node.operands):
            operand = node.operands[i]
            i += 1

            if operand == glue:
                continue

            ops.append(operand)

        ops.append(chain)

        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(AArch64MachineOps.BL, node.value_types, *ops)

    def select_copy_from_reg(self, node: DagNode, dag: Dag, new_ops):
        return node

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
        ops_table = [op for op in AArch64MachineOps.insts()]

        value = DagValue(node, 0)

        def match_node(inst: MachineInstructionDef):
            if inst.enabled:
                if not inst.enabled(dag.mfunc.target_info):
                    return None

            for pattern in inst.patterns:
                _, res = pattern.match(None, [value], 0, dag)
                if res:
                    return construct(inst, node, dag, res)

            return None

        for op in ops_table:
            matched = match_node(op)
            if matched:
                return matched

        for pattern in aarch64_patterns:
            _, res = pattern.match(node, dag)
            if res:
                return pattern.construct(node, dag, res).node

        return None

    def select_frame_index(self, node: DagNode, dag: Dag, new_ops):
        base = DagValue(dag.add_frame_index_node(
            node.value_types[0], node.index, True), 0)

        offset = DagValue(dag.add_target_constant_node(
            node.value_types[0], 0), 0)
        ops = [base, offset]

        node = dag.add_machine_dag_node(
            AArch64MachineOps.ADDXri, node.value_types, *ops)
        node.temp = "frame_index"
        return node

    def select_dup(self, node: DagNode, dag: Dag, new_ops):
        in_type = node.operands[0].ty
        out_type = node.value_types[0]

        hwmode = dag.mfunc.target_info.hwmode
        tys = FPR32.get_types(hwmode)

        elem = node.operands[0]

        if in_type in tys and out_type.value_type == ValueType.V4F32:
            value = DagValue(dag.add_node(TargetDagOps.IMPLICIT_DEF,
                                          [MachineValueType(ValueType.V4I32)]), 0)
            subreg = ssub

            subreg_id = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I32), subregs.index(subreg)), 0)
            vec = DagValue(dag.add_node(TargetDagOps.INSERT_SUBREG,
                                        node.value_types, value, elem, subreg_id), 0)

            lane = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I64), 0), 0)
            return dag.add_machine_dag_node(AArch64MachineOps.DUPv4i32lane, node.value_types, vec, lane)

        raise ValueError()

    def select_insert_vector_elt(self, node: DagNode, dag: Dag, new_ops):
        vec = node.operands[0]
        elem = node.operands[1]
        idx = node.operands[2]

        hwmode = dag.mfunc.target_info.hwmode

        if isinstance(idx.node, ConstantDagNode):
            if node.value_types[0].value_type in [ValueType.V2F32, ValueType.V4F32]:
                tys = FPR32.get_types(hwmode)
                if elem.ty in tys:
                    value = DagValue(dag.add_node(TargetDagOps.IMPLICIT_DEF,
                                                  node.value_types), 0)
                    subreg = ssub

                    subreg_id = DagValue(dag.add_target_constant_node(
                        MachineValueType(ValueType.I32), subregs.index(subreg)), 0)
                    value = DagValue(dag.add_node(
                        TargetDagOps.INSERT_SUBREG, node.value_types, value, elem, subreg_id), 0)

                    idx = DagValue(dag.add_target_constant_node(
                        MachineValueType(ValueType.I32), idx.node.value), 0)
                    idx2 = DagValue(dag.add_target_constant_node(
                        MachineValueType(ValueType.I64), 0), 0)

                    return dag.add_machine_dag_node(AArch64MachineOps.INSvi32lane, [MachineValueType(ValueType.V4F32)], vec, idx, value, idx2)

        # TODO: Neet to implement indexing with variable. A solution is using memory.

        raise NotImplementedError()

    def select_scalar_to_vector(self, node: DagNode, dag: Dag, new_ops):
        elem = node.operands[0]

        if node.value_types[0].value_type == ValueType.V2F32:
            value = DagValue(dag.add_node(TargetDagOps.IMPLICIT_DEF,
                                          [MachineValueType(ValueType.V2F32)]), 0)
            subreg = ssub

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
            AArch64DagOps.CALL: self.select_call,
            AArch64DagOps.DUP: self.select_dup,
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
        elif node.opcode == VirtualDagOps.TARGET_FRAME_INDEX:
            return node
        elif node.opcode == VirtualDagOps.CONDCODE:
            return node
        elif node.opcode == VirtualDagOps.BASIC_BLOCK:
            return node
        elif node.opcode == VirtualDagOps.REGISTER:
            return node
        elif node.opcode == VirtualDagOps.TARGET_REGISTER:
            return node
        elif node.opcode == VirtualDagOps.REGISTER_MASK:
            return node
        elif node.opcode == VirtualDagOps.TARGET_CONSTANT_FP:
            return node
        elif node.opcode == VirtualDagOps.TARGET_GLOBAL_ADDRESS:
            return node
        elif node.opcode == VirtualDagOps.TARGET_GLOBAL_TLS_ADDRESS:
            return node
        elif node.opcode == VirtualDagOps.EXTERNAL_SYMBOL:
            return dag.add_external_symbol_node(node.value_types[0], node.symbol, True)
        elif node.opcode == VirtualDagOps.MERGE_VALUES:
            return dag.add_node(node.opcode, node.value_types, *new_ops)
        elif node.opcode == VirtualDagOps.TOKEN_FACTOR:
            return dag.add_node(node.opcode, node.value_types, *new_ops)
        elif node.opcode in SELECT_TABLE:
            select_func = SELECT_TABLE[node.opcode]
            return select_func(node, dag, new_ops)

        matched = self.select_code(node, dag)

        if matched:
            return matched

        def bitcast_fp_to_i32(f):
            import struct
            f = struct.unpack('f', struct.pack('f', f))[0]
            return struct.unpack('<I', struct.pack('<f', f))[0]

        if node.opcode == VirtualDagOps.CONSTANT_FP:
            if node.value_types[0].value_type == ValueType.F32:
                imm = node.value.value
                imm = bitcast_fp_to_i32(imm)
                value = DagValue(dag.add_constant_node(
                    MachineValueType(ValueType.I32), ConstantInt(imm, i32), True), 0)
                value = DagValue(dag.add_machine_dag_node(AArch64MachineOps.MOVi32imm, [
                                 MachineValueType(ValueType.I32)], value), 0)

                regclass_id = regclasses.index(FPR32)
                regclass_id_val = DagValue(dag.add_target_constant_node(
                    MachineValueType(ValueType.I32), regclass_id), 0)
                return dag.add_node(TargetDagOps.COPY_TO_REGCLASS, node.value_types, value, regclass_id_val)

        raise NotImplementedError(
            "Can't select the instruction: {}".format(node.opcode))


class ArgListEntry:
    def __init__(self, node, ty):
        self.node = node
        self.ty = ty
        self.is_sret = False


class CallInfo:
    def __init__(self, dag, chain, ret_ty, target, arg_list):
        self.dag = dag
        self.chain = chain
        self.ret_ty = ret_ty
        self.target = target
        self.arg_list = arg_list


class AArch64CallingConv(CallingConv):
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
            for _, vt in enumerate(return_vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                for _ in range(reg_count):
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
                ret_reg = W0
            elif ret_val.ty == MachineValueType(ValueType.I64):
                ret_reg = X0
            else:
                raise NotImplementedError()

            reg_node = DagValue(
                g.add_target_register_node(ret_val.ty, ret_reg), 0)

            node = g.add_copy_to_reg_node(reg_node, ret_val)
            builder.root = DagValue(node, 0)

            ops = [builder.root, stack_pop_bytes, reg_node]

        node = g.add_node(AArch64DagOps.RETURN, [
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
            arg_entry = ArgListEntry(builder.get_value(arg), param.ty)
            if param.has_attribute(AttributeKind.StructRet):
                arg_entry.is_sret = True

            arg_list.append(arg_entry)

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

            for _, vt in enumerate(vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                for _ in range(reg_count):
                    flags = CCArgFlags()

                    flags.is_sret = arg.is_sret

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
        glue = copy_to_reg_chain.get_value(1)

        mask = DagValue(dag.add_register_mask_node(
            reg_info.get_call_reserved_mask(calling_conv)), 0)

        ops = []
        ops.append(chain)
        ops.append(func_address)
        ops.append(mask)
        ops.extend(reg_vals)
        ops.append(glue)

        call_node = dag.add_node(
            AArch64DagOps.CALL, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)], *ops)

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

    def allocate_return_aarch64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)
            loc_info = CCArgLocInfo.ZExt

        if loc_vt.value_type == ValueType.I32:
            regs1 = [W0, W1, W2, W3, W4, W5, W6, W7]
            regs2 = [X0, X1, X2, X3, X4, X5, X6, X7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs1 = [X0, X1, X2, X3, X4, X5, X6, X7]
            regs2 = [W0, W1, W2, W3, W4, W5, W6, W7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.F32:
            regs1 = [S0, S1, S2, S3, S4, S5, S6, S7]
            regs2 = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.F64:
            regs1 = [D0, D1, D2, D3, D4, D5, D6, D7]
            regs2 = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.V4F32]:
            loc_vt = MachineValueType(ValueType.F128)
            loc_info = CCArgLocInfo.BCvt

        if loc_vt.value_type == ValueType.F128:
            regs = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_return(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        self.allocate_return_aarch64_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)

    def allocate_argument_aarch64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)
            loc_info = CCArgLocInfo.ZExt

        if flags.is_sret:
            if loc_vt.value_type == ValueType.I64:
                regs1 = [X8]
                regs2 = [W8]
                reg = ccstate.alloc_reg_from_list(regs1, regs2)
                if reg is not None:
                    ccstate.assign_reg_value(
                        idx, vt, loc_vt, loc_info, reg, flags)
                    return False

        if loc_vt.value_type == ValueType.I32:
            regs1 = [W0, W1, W2, W3, W4, W5, W6, W7]
            regs2 = [X0, X1, X2, X3, X4, X5, X6, X7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs1 = [X0, X1, X2, X3, X4, X5, X6, X7]
            regs2 = [W0, W1, W2, W3, W4, W5, W6, W7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.F32:
            regs1 = [S0, S1, S2, S3, S4, S5, S6, S7]
            regs2 = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.F64:
            regs1 = [D0, D1, D2, D3, D4, D5, D6, D7]
            regs2 = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.V4F32]:
            loc_vt = MachineValueType(ValueType.F128)
            loc_info = CCArgLocInfo.BCvt

        if loc_vt.value_type == ValueType.F128:
            regs = [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            stack_offset = ccstate.alloc_stack(8, 8)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        if loc_vt.value_type == ValueType.F32:
            stack_offset = ccstate.alloc_stack(8, 8)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        if loc_vt.value_type == ValueType.F128:
            stack_offset = ccstate.alloc_stack(16, 16)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_argument(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        self.allocate_argument_aarch64_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)


class ShiftExtendType(Enum):
    InvalidShiftExtend = -1
    LSL = 0
    LSR = auto()
    ASR = auto()
    ROR = auto()
    MSL = auto()

    UXTB = auto()
    UXTH = auto()
    UXTW = auto()
    UXTX = auto()

    SXTB = auto()
    SXTH = auto()
    SXTW = auto()
    SXTX = auto()


def get_shifter_imm(shift_ty, imm):
    if shift_ty == ShiftExtendType.LSL:
        enc = 0
    elif shift_ty == ShiftExtendType.LSR:
        enc = 1
    elif shift_ty == ShiftExtendType.ASR:
        enc = 2
    elif shift_ty == ShiftExtendType.ROR:
        enc = 3
    elif shift_ty == ShiftExtendType.MSL:
        enc = 4
    else:
        raise ValueError("Invalid shift type")

    return (enc << 6) | (imm & 0x3f)


class AArch64TargetInstInfo(TargetInstInfo):
    def __init__(self):
        super().__init__()

    def copy_phys_reg(self, src_reg, dst_reg, kill_src, inst: MachineInstruction):
        assert(isinstance(src_reg, MachineRegister))
        assert(isinstance(dst_reg, MachineRegister))

        if src_reg.spec in GPR32sp.regs and dst_reg.spec in GPR32.regs:
            opcode = AArch64MachineOps.ADDWri
        elif src_reg.spec in GPR64sp.regs and dst_reg.spec in GPR64.regs:
            opcode = AArch64MachineOps.ADDXri
        elif src_reg.spec in FPR32.regs and dst_reg.spec in FPR32.regs:
            opcode = AArch64MachineOps.FMOVSr
        elif src_reg.spec in FPR64.regs and dst_reg.spec in FPR64.regs:
            opcode = AArch64MachineOps.FMOVDr
        elif src_reg.spec in FPR128.regs and dst_reg.spec in FPR128.regs:
            opcode = AArch64MachineOps.ORRv16i8
        elif src_reg.spec in GPR32sp.regs and dst_reg.spec in FPR32.regs:
            opcode = AArch64MachineOps.FMOVWSr
        elif src_reg.spec in FPR32.regs and dst_reg.spec in GPR32sp.regs:
            opcode = AArch64MachineOps.FMOVSWr
        elif src_reg.spec in GPR64sp.regs and dst_reg.spec in FPR64.regs:
            opcode = AArch64MachineOps.FMOVWDr
        elif src_reg.spec in FPR64.regs and dst_reg.spec in GPR64sp.regs:
            opcode = AArch64MachineOps.FMOVDWr
        else:
            raise NotImplementedError(
                "Move instructions support GPR, SPR, DPR or QPR at the present time.")

        copy_inst = MachineInstruction(opcode)

        copy_inst.add_reg(dst_reg, RegState.Define)
        copy_inst.add_reg(src_reg, RegState.Kill if kill_src else RegState.Non)

        if opcode in [AArch64MachineOps.ORRv16i8]:
            copy_inst.add_reg(
                src_reg, RegState.Kill if kill_src else RegState.Non)

        if opcode in [AArch64MachineOps.ADDWri, AArch64MachineOps.ADDXri]:
            # copy_inst.add_reg(
            #     src_reg, RegState.Kill if kill_src else RegState.Non)
            copy_inst.add_imm(get_shifter_imm(ShiftExtendType.LSL, 0))

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
            if has_reg_regclass(reg, GPR32sp):
                copy_inst = MachineInstruction(AArch64MachineOps.STRWui)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR32):
                copy_inst = MachineInstruction(AArch64MachineOps.STRSui)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if has_reg_regclass(reg, GPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.STRXui)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.STRDui)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
            if has_reg_regclass(reg, FPR128):
                copy_inst = MachineInstruction(AArch64MachineOps.STRQui)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst.insert_before(inst)
        return copy_inst

    def copy_reg_pair_to_stack(self, reg, reg2, stack_slot, regclass, inst: MachineInstruction):
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
            if has_reg_regclass(reg, GPR32sp):
                copy_inst = MachineInstruction(AArch64MachineOps.STPWi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR32):
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif size == 8:
            if has_reg_regclass(reg, GPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.STPXi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.STPDi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
            if has_reg_regclass(reg, FPR128):
                raise NotImplementedError()
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
            if has_reg_regclass(reg, GPR32sp):
                copy_inst = MachineInstruction(AArch64MachineOps.LDRWui)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR32):
                copy_inst = MachineInstruction(AArch64MachineOps.LDRSui)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if has_reg_regclass(reg, GPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.LDRXui)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.LDRDui)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
            if has_reg_regclass(reg, FPR128):
                copy_inst = MachineInstruction(AArch64MachineOps.LDRQui)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst.insert_before(inst)
        return copy_inst

    def copy_reg_pair_from_stack(self, reg, reg2, stack_slot, regclass, inst: MachineInstruction):
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
            if has_reg_regclass(reg, GPR32sp):
                copy_inst = MachineInstruction(AArch64MachineOps.LDPWi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR32):
                copy_inst = MachineInstruction(AArch64MachineOps.LDPSi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if has_reg_regclass(reg, GPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.LDPXi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif has_reg_regclass(reg, FPR64):
                copy_inst = MachineInstruction(AArch64MachineOps.LDPDi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
            if has_reg_regclass(reg, FPR128):
                copy_inst = MachineInstruction(AArch64MachineOps.LDPQi)

                copy_inst.add_reg(reg, RegState.Non)
                copy_inst.add_reg(reg2, RegState.Non)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
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

        has_fp = False
        if has_fp:
            if idx < 0:
                return FP, stack_obj.offset + frame_lowering.frame_spill_size

            return FP, stack_obj.offset

        if idx < 0:
            return SP, stack_obj.offset + frame_lowering.frame_spill_size + func.frame.stack_size

        return SP, stack_obj.offset + func.frame.stack_size

    def eliminate_frame_index(self, func: MachineFunction, inst: MachineInstruction, idx):
        # Analyze the frame index into a base register and a displacement.
        def get_scale(opc):
            if opc in [AArch64MachineOps.LDRWui, AArch64MachineOps.STRWui]:
                return 2
            elif opc in [AArch64MachineOps.LDRXui, AArch64MachineOps.STRXui]:
                return 3
            elif opc in [AArch64MachineOps.LDRHui, AArch64MachineOps.STRHui]:
                return 1
            elif opc in [AArch64MachineOps.LDRSui, AArch64MachineOps.STRSui]:
                return 2
            elif opc in [AArch64MachineOps.LDRDui, AArch64MachineOps.STRDui]:
                return 3
            elif opc in [AArch64MachineOps.LDRQui, AArch64MachineOps.STRQui]:
                return 4

        operand = inst.operands[idx]
        if isinstance(operand, MOFrameIndex):
            stack_obj = func.frame.get_stack_object(operand.index)
            frame_reg, offset = self.calculate_frame_offset(
                func, operand.index)
            base_reg = MachineRegister(frame_reg)

            if inst.opcode in [AArch64MachineOps.LDRWui, AArch64MachineOps.LDRXui, AArch64MachineOps.LDRSui, AArch64MachineOps.LDRDui, AArch64MachineOps.LDRQui,
                               AArch64MachineOps.STRWui, AArch64MachineOps.STRXui, AArch64MachineOps.STRSui, AArch64MachineOps.STRDui, AArch64MachineOps.STRQui]:
                scale = get_scale(inst.opcode)

                assert(offset & (scale - 1) == 0)
                new_offset = (inst.operands[idx + 1].val + (offset >> scale))
                assert(0 <= new_offset and new_offset <= 4095)

                offset = offset >> scale

            if inst.opcode in [AArch64MachineOps.STURWi, AArch64MachineOps.STURXi, AArch64MachineOps.STURSi, AArch64MachineOps.STURDi, AArch64MachineOps.STURQi,
                               AArch64MachineOps.LDURWi, AArch64MachineOps.LDURXi, AArch64MachineOps.LDURSi, AArch64MachineOps.LDURDi, AArch64MachineOps.LDURQi]:
                new_offset = inst.operands[idx + 1].val + offset
                assert(new_offset >= -256 and new_offset <= 255)

            inst.operands[idx] = MOReg(base_reg, RegState.Non)
            inst.operands[idx + 1] = MOImm(inst.operands[idx + 1].val + offset)

            if inst.operands[idx + 1].val < 0:
                if inst.opcode == AArch64MachineOps.ADDXri:
                    inst.opcode = AArch64MachineOps.SUBXri
                    inst.operands[idx + 1] = MOImm(-inst.operands[idx + 1].val)
                elif inst.opcode == AArch64MachineOps.SUBXri:
                    inst.opcode = AArch64MachineOps.ADDXri
                    inst.operands[idx + 1] = MOImm(-inst.operands[idx + 1].val)
                elif inst.opcode in [AArch64MachineOps.STURWi, AArch64MachineOps.STURXi, AArch64MachineOps.STURSi, AArch64MachineOps.STURDi, AArch64MachineOps.STURQi,
                                     AArch64MachineOps.LDURWi, AArch64MachineOps.LDURXi, AArch64MachineOps.LDURSi, AArch64MachineOps.LDURDi, AArch64MachineOps.LDURQi]:
                    pass
                else:
                    raise ValueError()

    def optimize_compare_inst(self, func: MachineFunction, inst: MachineInstruction):
        # Eliminate destination register.
        reginfo = func.reg_info
        if reginfo.is_use_empty(inst.operands[0].reg):
            pass

    def expand_post_ra_pseudo(self, inst: MachineInstruction):
        if inst.opcode == AArch64MachineOps.RET_ReallyLR:
            new_inst = MachineInstruction(AArch64MachineOps.RET)
            new_inst.add_reg(MachineRegister(LR), RegState.Non)

            new_inst.insert_after(inst)
            inst.remove()

        if inst.opcode == AArch64MachineOps.MOVi64imm:
            dst = inst.operands[0]
            src = inst.operands[1]

            if src.val == 0:
                new_inst = MachineInstruction(AArch64MachineOps.ORRXrr)
                new_inst.add_reg(dst.reg, RegState.Define)
                new_inst.add_reg(MachineRegister(XZR), RegState.Non)
                new_inst.add_reg(MachineRegister(XZR), RegState.Non)

                new_inst.insert_after(inst)
            else:
                insert_point = inst

                for i in range(0, 64, 16):
                    if i == 0:
                        new_inst = MachineInstruction(AArch64MachineOps.MOVKWi)
                        new_inst.add_reg(dst.reg, RegState.Define)
                        new_inst.add_reg(MachineRegister(WZR), RegState.Non)
                    else:
                        new_inst = MachineInstruction(AArch64MachineOps.MOVKWi)
                        new_inst.add_reg(dst.reg, RegState.Define)
                        new_inst.add_reg(dst.reg, RegState.Non)

                    if isinstance(src, MOGlobalAddress):
                        raise NotImplementedError()
                    elif isinstance(src, MOImm):
                        val = (src.val >> i) & 0xFFFF
                        if val == 0:
                            continue

                        new_inst.add_imm(val)
                        new_inst.add_imm(i)
                    else:
                        raise ValueError()

                    new_inst.insert_after(insert_point)
                    insert_point = new_inst
            inst.remove()

        if inst.opcode == AArch64MachineOps.MOVi32imm:
            dst = inst.operands[0]
            src = inst.operands[1]

            if src.val == 0:
                new_inst = MachineInstruction(AArch64MachineOps.ORRWrr)
                new_inst.add_reg(dst.reg, RegState.Define)
                new_inst.add_reg(MachineRegister(WZR), RegState.Non)
                new_inst.add_reg(MachineRegister(WZR), RegState.Non)

                new_inst.insert_after(inst)
            else:
                lo_opc = AArch64MachineOps.MOVKWi
                hi_opc = AArch64MachineOps.MOVKWi

                lo_inst = MachineInstruction(lo_opc)
                lo_inst.add_reg(dst.reg, RegState.Define)

                hi_inst = MachineInstruction(hi_opc)
                hi_inst.add_reg(dst.reg, RegState.Define)

                if isinstance(src, MOGlobalAddress):
                    raise NotImplementedError()
                elif isinstance(src, MOImm):
                    lo_val = (src.val >> 0) & 0xFFFF

                    reg = MachineRegister(WZR)

                    if lo_val:
                        lo_inst.add_reg(reg, RegState.Non)
                        lo_inst.add_imm(lo_val)
                        lo_inst.add_imm(0)

                        lo_inst.insert_after(inst)

                        reg = dst.reg

                    hi_inst.add_reg(reg, RegState.Non)
                    hi_inst.add_imm((src.val >> 16) & 0xFFFF)
                    hi_inst.add_imm(16)

                    hi_inst.insert_after(inst)
                else:
                    raise ValueError()

            inst.remove()

        if inst.opcode == AArch64MachineOps.CMPSWrr:
            src1 = inst.operands[0]
            src2 = inst.operands[1]

            new_inst = MachineInstruction(AArch64MachineOps.SUBSWrr)
            new_inst.add_reg(MachineRegister(WZR), RegState.Define)
            new_inst.add_reg(
                src1.reg, RegState.Kill if src1.is_kill else RegState.Non)
            new_inst.add_reg(
                src2.reg, RegState.Kill if src2.is_kill else RegState.Non)

            new_inst.insert_after(inst)
            inst.remove()

        if inst.opcode in [AArch64MachineOps.MOVaddr, AArch64MachineOps.MOVaddrCP]:
            dst = inst.operands[0]
            src1 = inst.operands[1]
            src2 = inst.operands[2]

            hi_opc = AArch64MachineOps.MOVKWi

            new_inst1 = MachineInstruction(AArch64MachineOps.ADRP)
            new_inst1.add_reg(dst.reg, RegState.Define)

            if isinstance(src1, MOGlobalAddress):
                new_inst1.add_global_address(
                    src1.value, target_flags=src1.target_flags)
            elif isinstance(src1, MOConstantPoolIndex):
                new_inst1.add_constant_pool_index(
                    src1.index, target_flags=src1.target_flags)
            else:
                raise ValueError()

            new_inst2 = MachineInstruction(AArch64MachineOps.ADDXri)
            new_inst2.add_reg(dst.reg, RegState.Define)
            new_inst2.add_reg(dst.reg, RegState.Non)

            if isinstance(src2, MOGlobalAddress):
                new_inst2.add_global_address(
                    src2.value, target_flags=src2.target_flags)
            elif isinstance(src2, MOConstantPoolIndex):
                new_inst2.add_constant_pool_index(
                    src2.index, target_flags=src2.target_flags)
            else:
                raise ValueError()

            new_inst1.insert_after(inst)
            new_inst2.insert_after(new_inst1)
            inst.remove()

        if inst.opcode == AArch64MachineOps.MOVbaseTLS:
            sysreg = AArch64SysReg.TPIDR_EL0

            dst = inst.operands[0]

            new_inst = MachineInstruction(AArch64MachineOps.MRS)
            new_inst.add_reg(dst.reg, RegState.Define)
            new_inst.add_imm(list(AArch64SysReg).index(sysreg))

            new_inst.insert_after(inst)
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


AArch64CC_EQ = 0b0000  # Equal                      Equal
AArch64CC_NE = 0b0001  # Not equal                  Not equal, or unordered
AArch64CC_HS = 0b0010  # Carry set                  >, ==, or unordered
AArch64CC_LO = 0b0011  # Carry clear                Less than
AArch64CC_MI = 0b0100  # Minus, negative            Less than
AArch64CC_PL = 0b0101  # Plus, positive or zero     >, ==, or unordered
AArch64CC_VS = 0b0110  # Overflow                   Unordered
AArch64CC_VC = 0b0111  # No overflow                Not unordered
AArch64CC_HI = 0b1000  # Unsigned higher            Greater than, or unordered
AArch64CC_LS = 0b1001  # Unsigned lower or same     Less than or equal
AArch64CC_GE = 0b1010  # Greater than or equal      Greater than or equal
AArch64CC_LT = 0b1011  # Less than                  Less than, or unordered
AArch64CC_GT = 0b1100  # Greater than               Greater than
AArch64CC_LE = 0b1101  # Less than or equal         <, ==, or unordered
AArch64CC_AL = 0b1110  # Always (unconditional)     Always (unconditional)
AArch64CC_NV = 0b1111  # Always (unconditional)     Always (unconditional)


def get_inverted_condcode(code):
    return code ^ 0x1


class RegPairType(Enum):
    GPR = auto()
    FPR64 = auto()
    FPR128 = auto()
    PPR = auto()
    ZPR = auto()


class RegPairInfo:
    def __init__(self, ty: RegPairType, reg1, reg2=None):
        self.ty = ty
        self.reg1 = reg1
        self.reg2 = reg2

    def is_paird(self):
        return self.reg2 is not None


class AArch64TargetLowering(TargetLowering):
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

    def get_aarch64_cmp(self, lhs: DagValue, rhs: DagValue, cond: CondCode, dag: Dag):

        is_fcmp = lhs.ty.value_type in [
            ValueType.F32, ValueType.F64]

        def compute_condcode(cond):
            ty = MachineValueType(ValueType.I8)
            swap = False

            if is_fcmp:
                if cond in [CondCode.SETEQ, CondCode.SETOEQ]:
                    node = dag.add_target_constant_node(ty, AArch64CC_EQ)
                elif cond in [CondCode.SETGT, CondCode.SETOGT]:
                    node = dag.add_target_constant_node(ty, AArch64CC_GT)
                elif cond in [CondCode.SETLT]:
                    node = dag.add_target_constant_node(ty, AArch64CC_LT)
                elif cond in [CondCode.SETOLT]:
                    node = dag.add_target_constant_node(ty, AArch64CC_MI)
                elif cond in [CondCode.SETGE, CondCode.SETOGE]:
                    node = dag.add_target_constant_node(ty, AArch64CC_GE)
                elif cond in [CondCode.SETLE]:
                    node = dag.add_target_constant_node(ty, AArch64CC_LE)
                elif cond in [CondCode.SETOLE]:
                    node = dag.add_target_constant_node(ty, AArch64CC_LS)
                else:
                    raise NotImplementedError()
            else:
                assert(lhs.ty.value_type in [
                       ValueType.I8, ValueType.I16, ValueType.I32, ValueType.I64])
                if cond == CondCode.SETEQ:
                    node = dag.add_target_constant_node(ty, AArch64CC_EQ)
                elif cond == CondCode.SETNE:
                    node = dag.add_target_constant_node(ty, AArch64CC_NE)
                elif cond == CondCode.SETLT:
                    node = dag.add_target_constant_node(ty, AArch64CC_LT)
                elif cond == CondCode.SETGT:
                    node = dag.add_target_constant_node(ty, AArch64CC_GT)
                elif cond == CondCode.SETLE:
                    node = dag.add_target_constant_node(ty, AArch64CC_LE)
                elif cond == CondCode.SETGE:
                    node = dag.add_target_constant_node(ty, AArch64CC_GE)
                elif cond == CondCode.SETULT:
                    node = dag.add_target_constant_node(ty, AArch64CC_HI)
                    swap = True
                elif cond == CondCode.SETUGT:
                    node = dag.add_target_constant_node(ty, AArch64CC_HI)
                elif cond == CondCode.SETULE:
                    node = dag.add_target_constant_node(ty, AArch64CC_LS)
                elif cond == CondCode.SETUGE:
                    node = dag.add_target_constant_node(ty, AArch64CC_LS)
                    swap = True
                else:
                    raise NotImplementedError()

            return node, swap

        condcode, swap = compute_condcode(cond)
        if swap:
            lhs, rhs = rhs, lhs

        if is_fcmp:
            cmp_node = DagValue(dag.add_node(AArch64DagOps.CMPFP,
                                             [MachineValueType(ValueType.GLUE)], lhs, rhs), 0)
        else:
            cmp_node = DagValue(dag.add_node(AArch64DagOps.CMP,
                                             [MachineValueType(ValueType.GLUE)], lhs, rhs), 0)

        return cmp_node, condcode

    def lower_setcc(self, node: DagNode, dag: Dag):
        op1 = node.operands[0]
        op2 = node.operands[1]
        cond = node.operands[2]

        cmp_node, condcode = self.get_aarch64_cmp(
            op1, op2, cond.node.cond, dag)

        vt = node.value_types[0]

        true_val = DagValue(dag.add_constant_node(
            vt, ConstantInt(1, vt.get_ir_type())), 0)

        false_val = DagValue(dag.add_constant_node(
            vt, ConstantInt(0, vt.get_ir_type())), 0)

        condcode = DagValue(dag.add_target_constant_node(
            condcode.value_types[0], get_inverted_condcode(condcode.value.value)), 0)

        return dag.add_node(AArch64DagOps.CSEL, [vt], false_val, true_val, condcode, cmp_node)

    def lower_brcond(self, node: DagNode, dag: Dag):
        chain = node.operands[0]
        cond = node.operands[1]
        dest = node.operands[2]

        if cond.node.opcode == VirtualDagOps.SETCC:
            lhs = cond.node.operands[0]
            rhs = cond.node.operands[1]
            condcode = cond.node.operands[2]

            cmp_node, condcode = self.get_aarch64_cmp(
                lhs, rhs, condcode.node.cond, dag)
            glue = cmp_node
            cond = DagValue(dag.add_target_register_node(
                MachineValueType(ValueType.I32), NZCV), 0)

            return dag.add_node(AArch64DagOps.BRCOND, node.value_types, chain, dest, DagValue(condcode, 0), cond, glue)
        else:
            lhs = cond
            rhs = DagValue(dag.add_constant_node(cond.ty, 0), 0)
            condcode = DagValue(dag.add_condition_code_node(CondCode.SETNE), 0)

            cmp_node, condcode = self.get_aarch64_cmp(
                lhs, rhs, condcode.node.cond, dag)
            glue = cmp_node
            cond = DagValue(dag.add_target_register_node(
                MachineValueType(ValueType.I32), NZCV), 0)

            return dag.add_node(AArch64DagOps.BRCOND, node.value_types, chain, dest, DagValue(condcode, 0), cond, glue)

    def lower_global_address(self, node: DagNode, dag: Dag):
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)

        hi_adr = DagValue(dag.add_global_address_node(
            node.value_types[0], node.value, True, AArch64OperandFlag.MO_PAGE), 0)

        lo_adr = DagValue(dag.add_global_address_node(
            node.value_types[0], node.value, True, AArch64OperandFlag.MO_PAGEOFF | AArch64OperandFlag.MO_NC), 0)

        adrp = DagValue(dag.add_node(AArch64DagOps.ADRP,
                                     node.value_types, hi_adr), 0)

        return dag.add_node(AArch64DagOps.ADDlow, [ptr_ty], adrp, lo_adr)

    def lower_global_tls_address(self, node: DagNode, dag: Dag):
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)
        global_value = node.value

        assert(dag.mfunc.target_info.triple.os == OS.Linux)

        thread_base = DagValue(dag.add_node(
            AArch64DagOps.THREAD_POINTER, [ptr_ty]), 0)

        if global_value.thread_local == ThreadLocalMode.GeneralDynamicTLSModel:
            sym_addr = DagValue(dag.add_global_address_node(
                ptr_ty, global_value, True, AArch64OperandFlag.MO_TLS), 0)

            chain = dag.entry

            vts = [MachineValueType(ValueType.OTHER),
                   MachineValueType(ValueType.GLUE)]

            chain = DagValue(dag.add_node(
                AArch64DagOps.TLSDESC_CALLSEQ, vts, chain, sym_addr), 0)
            glue = chain.get_value(1)

            reg_node = DagValue(dag.add_register_node(
                ptr_ty, MachineRegister(X0)), 0)

            tpoff = DagValue(dag.add_node(VirtualDagOps.COPY_FROM_REG, [
                ptr_ty], chain, reg_node, glue), 0)

            return dag.add_node(VirtualDagOps.ADD, [ptr_ty], thread_base, tpoff)

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
        ptr_ty = self.get_pointer_type(data_layout)

        hi_adr = DagValue(dag.add_constant_pool_node(
            ptr_ty, node.value, True, AArch64OperandFlag.MO_PAGE), 0)

        lo_adr = DagValue(dag.add_constant_pool_node(
            ptr_ty, node.value, True, AArch64OperandFlag.MO_PAGEOFF | AArch64OperandFlag.MO_NC), 0)

        adrp = DagValue(dag.add_node(AArch64DagOps.ADRP,
                                     node.value_types, hi_adr), 0)

        return dag.add_node(AArch64DagOps.ADDlow, [ptr_ty], adrp, lo_adr)

    def lower_build_vector(self, node: DagNode, dag: Dag):
        assert(node.opcode == VirtualDagOps.BUILD_VECTOR)

        is_one_val = True
        for idx in range(1, len(node.operands)):
            if node.operands[0].node != node.operands[idx].node or node.operands[0].index != node.operands[idx].index:
                is_one_val = False
                break

        if is_one_val:
            if node.value_types[0] == MachineValueType(ValueType.V4F32):
                return dag.add_node(AArch64DagOps.DUP, node.value_types, node.operands[0])

        operands = []
        for operand in node.operands:
            target_constant_fp = dag.add_target_constant_fp_node(
                operand.ty, operand.node.value)
            operands.append(DagValue(target_constant_fp, 0))

        assert(len(operands) > 0)

        return dag.add_node(VirtualDagOps.BUILD_VECTOR, node.value_types, *operands)

    def lower_bitcast(self, node: DagNode, dag: Dag):
        hwmode = dag.mfunc.target_info.hwmode

        tys = FPR128.get_types(hwmode)

        # if node.operands[0].ty in tys:
        #     if node.value_types[0] in tys:
        #         return dag.add_node(VirtualDagOps.OR, node.value_types, node.operands[0], node.operands[0])

        return node

    def lower(self, node: DagNode, dag: Dag):
        if node.opcode == VirtualDagOps.ENTRY:
            return dag.entry.node
        if node.opcode == VirtualDagOps.BRCOND:
            return self.lower_brcond(node, dag)
        elif node.opcode == VirtualDagOps.SETCC:
            return self.lower_setcc(node, dag)
        elif node.opcode == VirtualDagOps.GLOBAL_ADDRESS:
            return self.lower_global_address(node, dag)
        # elif node.opcode == VirtualDagOps.CONSTANT_FP:
        #     return self.lower_constant_fp(node, dag)
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

                    if arg.has_attribute(AttributeKind.StructRet):
                        flags.is_sret = True

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
                    regclass = GPR32
                elif arg_vt.value_type == ValueType.I16:
                    regclass = GPR32
                elif arg_vt.value_type == ValueType.I32:
                    regclass = GPR32
                elif arg_vt.value_type == ValueType.I64:
                    regclass = GPR64
                elif arg_vt.value_type == ValueType.F32:
                    regclass = FPR32
                elif arg_vt.value_type == ValueType.F64:
                    regclass = FPR64
                elif arg_vt.value_type == ValueType.V2F64:
                    regclass = FPR128
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
                    for _ in range(reg_count):
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
            mfunc.func_info.sret_reg = builder.func_info.reg_value_map[demote_arg]
            pass
        else:
            mfunc.func_info.sret_reg = None

        # builder.root = DagValue(DagNode(VirtualDagOps.TOKEN_FACTOR, [
        #     MachineValueType(ValueType.OTHER)], arg_load_chains), 0)

    def is_frame_op(self, inst):
        if inst.opcode == AArch64MachineOps.ADJCALLSTACKDOWN:
            return True
        if inst.opcode == AArch64MachineOps.ADJCALLSTACKUP:
            return True

        return False

    def compute_scr_pairs(self, func):
        csi = func.frame.calee_save_info

        def has_reg_regclass(reg, regclass):
            if isinstance(reg, MachineVirtualRegister):
                reg_info = inst.mbb.func.target_info.get_register_info()
                return reg.regclass == regclass or reg_info.is_subclass(regclass, reg.regclass)
            else:
                return reg.spec in regclass.regs

        pairs = []
        i = 0
        while i < len(csi):
            cs_info = csi[i]
            i += 1

            if has_reg_regclass(MachineRegister(cs_info.reg), GPR64):
                pair_ty = RegPairType.GPR
            elif has_reg_regclass(MachineRegister(cs_info.reg), FPR64):
                pair_ty = RegPairType.FPR64
            else:
                raise ValueError()

            pair_info = RegPairInfo(pair_ty, cs_info.reg)
            pair_info.frame_idx = cs_info.frame_idx

            if i < len(csi):
                cs_info = csi[i]

                if has_reg_regclass(MachineRegister(cs_info.reg), GPR64):
                    pair_ty2 = RegPairType.GPR
                elif has_reg_regclass(MachineRegister(cs_info.reg), FPR64):
                    pair_ty2 = RegPairType.FPR64
                else:
                    raise ValueError()

                if pair_ty == pair_ty2 and (cs_info.reg.encoding - pair_info.reg1.encoding) == 1:
                    pair_info.reg2 = cs_info.reg

                    # A higher frame index has small address because stack direction is downward
                    pair_info.frame_idx = max(
                        cs_info.frame_idx, pair_info.frame_idx)
                    i += 1

            pairs.append(pair_info)
        return pairs

    def lower_prolog(self, func: MachineFunction, bb: MachineBasicBlock):
        inst_info = func.target_info.get_inst_info()
        frame_info = func.target_info.get_frame_lowering()
        reg_info = func.target_info.get_register_info()
        data_layout = func.func_info.func.module.data_layout

        front_inst = bb.insts[0]

        push_fp_lr_inst = MachineInstruction(AArch64MachineOps.STPXprei)
        push_fp_lr_inst.add_reg(MachineRegister(FP), RegState.Non)
        push_fp_lr_inst.add_reg(MachineRegister(LR), RegState.Non)
        push_fp_lr_inst.add_reg(MachineRegister(SP), RegState.Non)
        push_fp_lr_inst.add_imm(-32)

        push_fp_lr_inst.insert_before(front_inst)

        inst_info.copy_phys_reg(
            MachineRegister(SP), MachineRegister(FP), False, push_fp_lr_inst)

        stack_size = func.frame.estimate_stack_size(
            AArch64MachineOps.ADJCALLSTACKDOWN, AArch64MachineOps.ADJCALLSTACKUP)

        max_align = max(func.frame.max_alignment, func.frame.stack_alignment)
        stack_size = func.frame.stack_size = int(
            int((stack_size + max_align - 1) / max_align) * max_align)

        sub_sp_inst = MachineInstruction(AArch64MachineOps.SUBXri)
        sub_sp_inst.add_reg(MachineRegister(SP), RegState.Define)
        sub_sp_inst.add_reg(MachineRegister(SP), RegState.Non)
        sub_sp_inst.add_imm(stack_size)

        sub_sp_inst.insert_before(front_inst)

        pairs = self.compute_scr_pairs(func)

        for csr_pair in pairs:
            reg = csr_pair.reg1
            frame_idx = csr_pair.frame_idx
            regclass = reg_info.get_regclass_from_reg(reg)

            if csr_pair.reg2:
                reg2 = csr_pair.reg2

                inst_info.copy_reg_pair_to_stack(MachineRegister(
                    reg), MachineRegister(reg2), frame_idx, regclass, front_inst)

                continue

            inst_info.copy_reg_to_stack(MachineRegister(
                reg), frame_idx, regclass, front_inst)

    def lower_epilog(self, func: MachineFunction, bb: MachineBasicBlock):
        inst_info = func.target_info.get_inst_info()
        reg_info = func.target_info.get_register_info()
        data_layout = func.func_info.func.module.data_layout

        stack_size = func.frame.estimate_stack_size(
            AArch64MachineOps.ADJCALLSTACKDOWN, AArch64MachineOps.ADJCALLSTACKUP)

        max_align = max(func.frame.max_alignment, func.frame.stack_alignment)
        stack_size = int(
            int((stack_size + max_align - 1) / max_align) * max_align)

        front_inst = bb.insts[-1]

        pairs = self.compute_scr_pairs(func)

        for csr_pair in pairs:
            reg = csr_pair.reg1
            frame_idx = csr_pair.frame_idx
            regclass = reg_info.get_regclass_from_reg(reg)

            if csr_pair.reg2:
                reg2 = csr_pair.reg2

                inst_info.copy_reg_pair_from_stack(MachineRegister(
                    reg), MachineRegister(reg2), frame_idx, regclass, front_inst)

                continue

            inst_info.copy_reg_from_stack(MachineRegister(
                reg), frame_idx, regclass, front_inst)

        restore_sp_inst = MachineInstruction(AArch64MachineOps.ADDXri)
        restore_sp_inst.add_reg(MachineRegister(SP), RegState.Define)
        restore_sp_inst.add_reg(MachineRegister(SP), RegState.Non)
        restore_sp_inst.add_imm(stack_size)

        restore_sp_inst.insert_before(front_inst)

        pop_fp_lr_inst = MachineInstruction(AArch64MachineOps.LDPXposti)
        pop_fp_lr_inst.add_reg(MachineRegister(FP), RegState.Non)
        pop_fp_lr_inst.add_reg(MachineRegister(LR), RegState.Non)
        pop_fp_lr_inst.add_reg(MachineRegister(SP), RegState.Non)
        pop_fp_lr_inst.add_imm(32)

        pop_fp_lr_inst.insert_before(front_inst)

    def eliminate_call_frame_pseudo_inst(self, func, inst: MachineInstruction):
        inst.remove()

    def get_machine_vreg(self, ty: MachineValueType):
        if ty.value_type == ValueType.I8:
            return GPR32
        elif ty.value_type == ValueType.I16:
            return GPR32
        elif ty.value_type == ValueType.I32:
            return GPR32
        elif ty.value_type == ValueType.I64:
            return GPR64
        elif ty.value_type == ValueType.F32:
            return FPR32
        elif ty.value_type == ValueType.F64:
            return FPR64
        elif ty.value_type == ValueType.V4F32:
            return FPR128

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
                    builder.g.add_constant_node(src_ty, offset), 0)
                src_ptr = DagValue(builder.g.add_node(
                    VirtualDagOps.ADD, [src_ty], src_op, size_node), 0)
            else:
                src_ptr = src_op

            if offset != 0:
                dst_ty = dst_op.ty
                size_node = DagValue(
                    builder.g.add_constant_node(dst_ty, offset), 0)
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


class AArch64TargetRegisterInfo(TargetRegisterInfo):
    def __init__(self, target_info):
        super().__init__()

        self.target_info = target_info

    def is_reserved(self, reg):
        return reg in self.get_reserved_regs()

    def get_call_reserved_mask(self, cc):
        mask = []

        mask.extend(iter_super_regs(W0))
        mask.extend(iter_super_regs(W1))
        mask.extend(iter_super_regs(W2))
        mask.extend(iter_super_regs(W3))
        mask.extend(iter_super_regs(W4))
        mask.extend(iter_super_regs(W5))
        mask.extend(iter_super_regs(W6))
        mask.extend(iter_super_regs(W7))
        mask.extend(iter_super_regs(W8))

        mask.extend(iter_super_regs(B0))
        mask.extend(iter_super_regs(B1))
        mask.extend(iter_super_regs(B2))
        mask.extend(iter_super_regs(B3))
        mask.extend(iter_super_regs(B4))
        mask.extend(iter_super_regs(B5))
        mask.extend(iter_super_regs(B6))
        mask.extend(iter_super_regs(B7))

        return mask

    def get_reserved_regs(self):
        reserved = []
        reserved.extend(iter_super_regs(W29))
        reserved.extend(iter_super_regs(WSP))
        reserved.extend(iter_super_regs(WZR))

        reserved.extend(iter_super_regs(W0))
        reserved.extend(iter_super_regs(W1))
        reserved.extend(iter_super_regs(W2))
        reserved.extend(iter_super_regs(W3))
        reserved.extend(iter_super_regs(W4))
        reserved.extend(iter_super_regs(W5))
        reserved.extend(iter_super_regs(W6))
        reserved.extend(iter_super_regs(W7))
        reserved.extend(iter_super_regs(W8))

        reserved.extend(iter_super_regs(B0))
        reserved.extend(iter_super_regs(B1))
        reserved.extend(iter_super_regs(B2))
        reserved.extend(iter_super_regs(B3))
        reserved.extend(iter_super_regs(B4))
        reserved.extend(iter_super_regs(B5))
        reserved.extend(iter_super_regs(B6))
        reserved.extend(iter_super_regs(B7))

        return reserved

    @property
    def allocatable_regs(self):
        regs = set()
        regs |= set(GPR32.regs)
        regs |= set(GPR64.regs)
        regs |= set(FPR16.regs)
        regs |= set(FPR32.regs)
        regs |= set(FPR64.regs)
        regs |= set(FPR128.regs)

        return regs

    def get_callee_saved_regs(self):
        callee_save_regs = []
        callee_save_regs.extend([
            X19, X20, X21, X22,
            X23, X24, X25, X26, X27, X28, FP])
        callee_save_regs.extend([D8, D9, D10, D11, D12, D13, D14, D15])

        return callee_save_regs

    def get_callee_clobbered_regs(self):
        regs = [X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, D16, D17,
                D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31]

        return regs

    def get_ordered_regs(self, regclass):
        reserved_regs = self.get_reserved_regs()

        free_regs = set(regclass.regs) - set(reserved_regs)

        return [reg for reg in regclass.regs if reg in free_regs]

    def get_regclass_for_vt(self, vt):
        hwmode = self.target_info.hwmode
        for regclass in aarch64_regclasses:
            tys = regclass.get_types(hwmode)
            if vt in tys:
                return regclass

        raise ValueError("Could not find the register class.")


class AArch64FrameLowering(TargetFrameLowering):
    def __init__(self, alignment):
        super().__init__(alignment)

        self.frame_spill_size = 32

    @property
    def stack_grows_direction(self):
        return StackGrowsDirection.Down


class AArch64Legalizer(Legalizer):
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


class AArch64TargetInfo(TargetInfo):
    def __init__(self, triple):
        super().__init__(triple)

        self._inst_info = AArch64TargetInstInfo()
        self._lowering = AArch64TargetLowering()
        self._reg_info = AArch64TargetRegisterInfo(self)
        self._calling_conv = AArch64CallingConv()
        self._isel = AArch64InstructionSelector()
        self._legalizer = AArch64Legalizer()
        self._frame_lowering = AArch64FrameLowering(16)

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
        if self.triple.arch == ArchType.ARM64:
            return AArch64

        raise ValueError("Invalid arch type")


class AArch64TargetMachine:
    def __init__(self, triple):
        self.triple = triple

    def get_target_info(self, func: Function):
        return AArch64TargetInfo(self.triple)

    def add_mc_emit_passes(self, pass_manager, mccontext, output, is_asm):
        from codegen.aarch64_asm_printer import AArch64AsmInfo, MCAsmStream, AArch64CodeEmitter, AArch64AsmBackend, AArch64AsmPrinter
        from codegen.coff import WinCOFFObjectWriter, WinCOFFObjectStream
        from codegen.elf import ELFObjectStream, ELFObjectWriter, AArch64ELFObjectWriter

        objformat = self.triple.objformat

        mccontext.asm_info = AArch64AsmInfo()
        if is_asm:
            raise NotImplementedError()
        else:
            emitter = AArch64CodeEmitter(mccontext)
            backend = AArch64AsmBackend()

            if objformat == ObjectFormatType.ELF:
                target_writer = AArch64ELFObjectWriter()
                writer = ELFObjectWriter(output, target_writer)
                stream = ELFObjectStream(mccontext, backend, writer, emitter)

        pass_manager.passes.append(AArch64AsmPrinter(stream))
