#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.spec import *
from codegen.mir_emitter import *
from codegen.isel import *
from codegen.riscv_def import *
from codegen.matcher import *


class RISCVOperandFlag(IntFlag):
    NONE = auto()
    LO = auto()
    HI = auto()
    PCREL_LO = auto()
    PCREL_HI = auto()
    TLS_GD_HI = auto()
    CALL = auto()


def is_null_constant(value):
    return isinstance(value.node, ConstantDagNode) and value.node.is_zero


def is_null_fp_constant(value):
    return isinstance(value.node, ConstantFPDagNode) and value.node.is_zero


class RISCVInstructionSelector(InstructionSelector):
    def __init__(self):
        super().__init__()

    def select_callseq_start(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        opt = dag.add_target_constant_node(MachineValueType(ValueType.I32), 0)
        return dag.add_machine_dag_node(RISCVMachineOps.ADJCALLSTACKDOWN, node.value_types, in_bytes, out_bytes, DagValue(opt, 0), chain)

    def select_callseq_end(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        glue = self.get_glue(new_ops)

        ops = [in_bytes, out_bytes, chain]
        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(RISCVMachineOps.ADJCALLSTACKUP, node.value_types, *ops)

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

        return dag.add_machine_dag_node(RISCVMachineOps.BL, node.value_types, *ops)

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
        ops_table = [op for op in RISCVMachineOps.insts()]

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

        for pattern in riscv_patterns:
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

        return dag.add_machine_dag_node(RISCVMachineOps.ADDI, node.value_types, *ops)

    def select(self, node: DagNode, dag: Dag):
        new_ops = node.operands

        SELECT_TABLE = {
            VirtualDagOps.COPY_FROM_REG: self.select_copy_from_reg,
            VirtualDagOps.COPY_TO_REG: self.select_copy_to_reg,
            VirtualDagOps.CALLSEQ_START: self.select_callseq_start,
            VirtualDagOps.CALLSEQ_END: self.select_callseq_end,
            VirtualDagOps.FRAME_INDEX: self.select_frame_index,
        }

        if isinstance(node, MachineDagNode):
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
        elif node.opcode == VirtualDagOps.TARGET_GLOBAL_TLS_ADDRESS:
            return node
        elif node.opcode == VirtualDagOps.TARGET_EXTERNAL_SYMBOL:
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


class RISCVCallingConv(CallingConv):
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
        target_lowering = mfunc.target_info.get_lowering()
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
                reg_vt = target_lowering.get_register_type(vt)
                reg_count = target_lowering.get_register_count(vt)

                for reg_idx in range(reg_count):
                    flags = CCArgFlags()

                    returns.append(CallingConvReturn(
                        vt, reg_vt, 0, offset_in_arg, flags))

                    offset_in_arg += reg_vt.get_size_in_byte()

            # Apply caling convention
            ccstate = CallingConvState(calling_conv, mfunc)
            ccstate.compute_returns_layout(returns)

            # Handle return values
            reg_vals = []
            for idx, ret_val in enumerate(ccstate.values):
                assert(isinstance(ret_val, CCArgReg))
                ret_vt = ret_val.loc_vt

                reg_val = DagValue(
                    g.add_target_register_node(ret_vt, ret_val.reg), 0)

                reg_vals.append(reg_val)

            ret_parts = []
            ret_value = builder.get_value(inst.rs)
            idx = 0
            for val_idx, vt in enumerate(return_vts):
                reg_vt = target_lowering.get_register_type(vt)
                reg_count = target_lowering.get_register_count(vt)

                value = ret_value.get_value(val_idx)

                builder.g.root = get_copy_to_parts(
                    value, reg_vals[idx:idx+reg_count], reg_vt, builder.g.root, builder.g)

                idx += reg_count

            ops = [builder.root, stack_pop_bytes, *reg_vals]
        else:
            ops = [builder.root, stack_pop_bytes]

        if has_demote_arg:
            return_ty = inst.block.func.ty
            vts = compute_value_types(
                return_ty, inst.block.func.module.data_layout)
            assert(len(vts) == 1)
            ret_val = DagValue(
                builder.g.add_register_node(vts[0], demote_reg[0]), 0)

            if ret_val.ty == MachineValueType(ValueType.I32):
                ret_reg = X10
            elif ret_val.ty == MachineValueType(ValueType.I64):
                ret_reg = X10
            else:
                raise NotImplementedError()

            reg_node = DagValue(
                g.add_target_register_node(ret_val.ty, ret_reg), 0)

            node = g.add_copy_to_reg_node(reg_node, ret_val)
            builder.root = DagValue(node, 0)

            ops = [builder.root, stack_pop_bytes, reg_node]

        node = g.add_node(RISCVDagOps.RETURN, [
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
        target_lowering = mfunc.target_info.get_lowering()
        data_layout = dag.data_layout

        func_address = call_info.target
        arg_list = call_info.arg_list
        ret_ty = call_info.ret_ty

        if isinstance(func_address.node, GlobalAddressDagNode):
            func_address = DagValue(dag.add_global_address_node(
                func_address.ty, func_address.node.value, True, target_flags=RISCVOperandFlag.CALL.value), 0)

        # Handle arguments
        args = []
        for i, arg in enumerate(arg_list):
            vts = compute_value_types(arg.ty, data_layout)
            offset_in_arg = 0

            for val_idx, vt in enumerate(vts):
                reg_vt = target_lowering.get_register_type(vt)
                reg_count = target_lowering.get_register_count(vt)

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
        reg_vals = []
        idx = 0
        for arg in arg_list:
            arg_value = arg.node
            vts = compute_value_types(arg.ty, data_layout)
            for val_idx, vt in enumerate(vts):
                reg_vt = target_lowering.get_register_type(vt)
                reg_count = target_lowering.get_register_count(vt)

                arg_to_copy = arg_value.get_value(val_idx)

                parts = get_parts_to_copy(arg_to_copy, reg_count, reg_vt, dag)

                for reg_idx in range(reg_count):
                    arg_val = ccstate.values[idx]
                    copy_val = parts[reg_idx]

                    assert(isinstance(arg_val, CCArgReg))
                    reg_val = DagValue(dag.add_target_register_node(
                        arg_val.vt, arg_val.reg), 0)

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

                idx += reg_count

        # Function call
        call_node = dag.add_node(
            RISCVDagOps.CALL, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)], chain, func_address, copy_to_reg_chain.get_value(1))

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
            reg_vt = target_lowering.get_register_type(vt)
            reg_count = target_lowering.get_register_count(vt)

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
            reg_vt = target_lowering.get_register_type(vt)
            reg_count = target_lowering.get_register_count(vt)

            if reg_count > 1:
                raise NotImplementedError()

            ret_parts.append(ret_vals[idx])

            idx += reg_count

        call_info.chain = chain

        if len(ret_parts) == 0:
            return None

        return dag.add_merge_values(ret_parts)

    def allocate_return_riscv_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.F32:
            regs = [F10_F, F11_F, F12_F, F13_F, F14_F, F15_F, F16_F, F17_F]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs = [X10, X11, X12, X13, X14, X15, X16, X17]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_return_riscv64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.F32:
            regs = [F10_F, F11_F, F12_F, F13_F, F14_F, F15_F, F16_F, F17_F]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            loc_vt = MachineValueType(ValueType.I64)
            loc_info = CCArgLocInfo.SExt

            regs = [X10, X11]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs = [X10, X11]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_return(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        target_info = ccstate.mfunc.target_info
        if target_info.triple.arch == ArchType.RISCV64:
            self.allocate_return_riscv64_cdecl(
                idx, vt, loc_vt, loc_info, flags, ccstate)
            return
        self.allocate_return_riscv_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)

    def allocate_argument_riscv_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.F32:
            regs = [F10_F, F11_F, F12_F, F13_F, F14_F, F15_F, F16_F, F17_F]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs = [X10, X11, X12, X13, X14, X15, X16, X17]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
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

        raise NotImplementedError("The type is unsupporting.")

    def allocate_argument_riscv64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.F32:
            regs = [F10_F, F11_F, F12_F, F13_F, F14_F, F15_F, F16_F, F17_F]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            loc_vt = MachineValueType(ValueType.I64)
            loc_info = CCArgLocInfo.SExt

            regs = [X10, X11, X12, X13, X14, X15, X16, X17]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs = [X10, X11, X12, X13, X14, X15, X16, X17]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg:
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

        raise NotImplementedError("The type is unsupporting.")

    def allocate_argument(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        target_info = ccstate.mfunc.target_info
        if target_info.triple.arch == ArchType.RISCV64:
            self.allocate_argument_riscv64_cdecl(
                idx, vt, loc_vt, loc_info, flags, ccstate)
            return

        self.allocate_argument_riscv_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)


class RISCVTargetInstInfo(TargetInstInfo):
    def __init__(self):
        super().__init__()

    def copy_phys_reg(self, src_reg, dst_reg, kill_src, inst: MachineInstruction):
        assert(isinstance(src_reg, MachineRegister))
        assert(isinstance(dst_reg, MachineRegister))

        if src_reg.spec in GPR.regs and dst_reg.spec in GPR.regs:
            opcode = RISCVMachineOps.ADDI
        elif src_reg.spec in FPR32.regs and dst_reg.spec in FPR32.regs:
            opcode = RISCVMachineOps.FSGNJ_S
        elif src_reg.spec in FPR64.regs and dst_reg.spec in FPR64.regs:
            opcode = RISCVMachineOps.FSGNJ_D
        else:
            raise NotImplementedError(
                "Move instructions support GPR, SPR, DPR or QPR at the present time.")

        copy_inst = MachineInstruction(opcode)

        copy_inst.add_reg(dst_reg, RegState.Define)
        copy_inst.add_reg(src_reg, RegState.Kill if kill_src else RegState.Non)

        if opcode == RISCVMachineOps.ADDI:
            copy_inst.add_imm(0)

        if opcode in [RISCVMachineOps.FSGNJ_S, RISCVMachineOps.FSGNJ_D]:
            copy_inst.add_reg(
                src_reg, RegState.Kill if kill_src else RegState.Non)

        copy_inst.insert_after(inst)

    def copy_reg_to_stack(self, reg, stack_slot, regclass, inst: MachineInstruction):
        hwmode = inst.mbb.func.target_info.hwmode

        tys = regclass.get_types(hwmode)

        align = int(regclass.align / 8)
        size = tys[0].get_size_in_bits()
        size = int(int((size + 7) / 8))

        noreg = MachineRegister(NOREG)

        if size == 1:
            raise NotImplementedError()
        elif size == 4:
            if reg.spec in GPR.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.SW)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif reg.spec in FPR32.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.FSW)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if reg.spec in GPR.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.SD)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif reg.spec in FPR64.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.FSD)

                copy_inst.add_reg(reg, RegState.Non)

                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
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

        if size == 1:
            raise NotImplementedError()
        elif size == 4:
            if reg.spec in GPR.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.LW)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif reg.spec in FPR32.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.FLW)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 8:
            if reg.spec in GPR.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.LD)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            elif reg.spec in FPR64.regs:
                copy_inst = MachineInstruction(RISCVMachineOps.FLD)

                copy_inst.add_reg(reg, RegState.Define)
                copy_inst.add_frame_index(stack_slot)
                copy_inst.add_imm(0)
            else:
                raise NotImplementedError()
        elif size == 16:
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
            return X8, stack_obj.offset + frame_lowering.frame_spill_size

        for cs_info in func.frame.calee_save_info:
            if idx == cs_info.frame_idx:
                return X2, stack_obj.offset + func.frame.stack_size
        return X8, stack_obj.offset

    def eliminate_frame_index(self, func: MachineFunction, inst: MachineInstruction, idx):
        target_reg_info = func.target_info.get_register_info()

        # Analyze the frame index into a base register and a displacement.
        operand = inst.operands[idx]
        if isinstance(operand, MOFrameIndex):
            assert(inst.opcode in [RISCVMachineOps.SW, RISCVMachineOps.LW, RISCVMachineOps.SD, RISCVMachineOps.LD,
                                   RISCVMachineOps.ADDI, RISCVMachineOps.FLW, RISCVMachineOps.FSW])

            stack_obj = func.frame.get_stack_object(operand.index)
            frame_reg, offset = self.calculate_frame_offset(
                func, operand.index)

            base_reg = MachineRegister(frame_reg)

            inst.operands[idx] = MOReg(base_reg, RegState.Non)
            inst.operands[idx + 1] = MOImm(inst.operands[idx + 1].val + offset)

    def optimize_compare_inst(self, func: MachineFunction, inst: MachineInstruction):
        pass

    def expand_post_ra_pseudo(self, inst: MachineInstruction):
        if inst.opcode == RISCVMachineOps.PseudoBR:

            imm = inst.operands[0]

            jump_inst = MachineInstruction(RISCVMachineOps.JAL)
            jump_inst.add_reg(MachineRegister(X0), RegState.Non)
            jump_inst.add_mbb(imm.mbb)

            jump_inst.insert_after(inst)
            inst.remove()

        if inst.opcode == RISCVMachineOps.PseudoRET:
            jump_inst = MachineInstruction(RISCVMachineOps.JALR)
            jump_inst.add_reg(MachineRegister(X0), RegState.Non)
            jump_inst.add_reg(MachineRegister(X1), RegState.Non)
            jump_inst.add_imm(0)

            jump_inst.insert_after(inst)
            inst.remove()

        if inst.opcode == RISCVMachineOps.PseudoLLA:
            dst = inst.operands[0]
            disp = inst.operands[1]

            new_mbb = inst.mbb.split_basic_block(inst)

            hi_inst = MachineInstruction(RISCVMachineOps.AUIPC)
            hi_inst.add_reg(dst.reg, RegState.Define)

            if isinstance(disp, MOConstantPoolIndex):
                hi_inst.add_constant_pool_index(
                    disp.index, RISCVOperandFlag.PCREL_HI)
            else:
                raise NotImplementedError()

            lo_inst = MachineInstruction(RISCVMachineOps.ADDI)
            lo_inst.add_reg(dst.reg, RegState.Define)
            lo_inst.add_reg(dst.reg, RegState.Non)
            lo_inst.add_mbb(new_mbb, RISCVOperandFlag.PCREL_LO)

            hi_inst.insert_after(inst)
            lo_inst.insert_after(hi_inst)

            inst.remove()

        if inst.opcode == RISCVMachineOps.PseudoLA_TLS_GD:
            dst = inst.operands[0]
            disp = inst.operands[1]

            new_mbb = inst.mbb.split_basic_block(inst)

            hi_inst = MachineInstruction(RISCVMachineOps.AUIPC)
            hi_inst.add_reg(dst.reg, RegState.Define)

            if isinstance(disp, MOGlobalAddress):
                hi_inst.add_global_address(
                    disp.value, RISCVOperandFlag.TLS_GD_HI)
            else:
                raise NotImplementedError()

            lo_inst = MachineInstruction(RISCVMachineOps.ADDI)
            lo_inst.add_reg(dst.reg, RegState.Define)
            lo_inst.add_reg(dst.reg, RegState.Non)
            lo_inst.add_mbb(new_mbb, RISCVOperandFlag.PCREL_LO)

            hi_inst.insert_after(inst)
            lo_inst.insert_after(hi_inst)

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

class RISCVTargetLowering(TargetLowering):
    def __init__(self, target_machine):
        super().__init__()

        self.target_machine = target_machine

        self.reg_type_for_vt = {MachineValueType(
            e): MachineValueType(e) for e in ValueType}

        self.reg_type_for_vt[MachineValueType(
            ValueType.V4F32)] = MachineValueType(ValueType.F32)

        if self.target_machine.hwmode == RV64:
            self.reg_type_for_vt[MachineValueType(
                ValueType.I32)] = MachineValueType(ValueType.I64)

        self.reg_count_for_vt = {MachineValueType(e): 1 for e in ValueType}

        self.reg_count_for_vt[MachineValueType(
            ValueType.V4F32)] = 4

    def lower_global_address(self, node: DagNode, dag: Dag):
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)

        global_addr_hi = DagValue(dag.add_global_address_node(
            node.value_types[0], node.value, True, target_flags=RISCVOperandFlag.HI.value), 0)

        global_addr_lo = DagValue(dag.add_global_address_node(
            node.value_types[0], node.value, True, target_flags=RISCVOperandFlag.LO.value), 0)

        global_addr = DagValue(dag.add_machine_dag_node(RISCVMachineOps.LUI,
                                                        [ptr_ty], global_addr_hi), 0)

        global_addr = DagValue(dag.add_machine_dag_node(RISCVMachineOps.ADDI,
                                                        [ptr_ty], global_addr, global_addr_lo), 0)

        return global_addr.node

    def lower_global_tls_address(self, node: DagNode, dag: Dag):
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)
        global_value = node.value

        if global_value.thread_local == ThreadLocalMode.GeneralDynamicTLSModel:
            ga = DagValue(dag.add_global_address_node(
                ptr_ty, global_value, True), 0)

            ga_addr = DagValue(dag.add_machine_dag_node(RISCVMachineOps.PseudoLA_TLS_GD,
                                                        [ptr_ty], ga), 0)

            arg_list = []
            arg_list.append(ArgListEntry(
                ga_addr, get_integer_type(ptr_ty.get_size_in_bits())))

            tls_get_addr_func = DagValue(
                dag.add_external_symbol_node(ptr_ty, "__tls_get_addr", True, RISCVOperandFlag.CALL), 0)

            calling_conv = dag.mfunc.target_info.get_calling_conv()

            tls_addr = calling_conv.lower_call_info(
                CallInfo(dag, dag.entry, get_integer_type(ptr_ty.get_size_in_bits()), tls_get_addr_func, arg_list))

            return tls_addr.node

        raise ValueError("Not supporing TLS model.")

    def get_pointer_type(self, data_layout, addr_space=0):
        return get_int_value_type(data_layout.get_pointer_size_in_bits(addr_space))

    def get_frame_index_type(self, data_layout):
        return get_int_value_type(data_layout.get_pointer_size_in_bits(0))

    def lower_constant_fp(self, node: DagNode, dag: Dag):
        assert(isinstance(node, ConstantFPDagNode))
        data_layout = dag.mfunc.func_info.func.module.data_layout

        position_independent = True  # TODO
        if position_independent:
            cp = DagValue(dag.add_constant_pool_node(
                self.get_pointer_type(data_layout), node.value, True), 0)

            cp_addr = DagValue(dag.add_machine_dag_node(RISCVMachineOps.PseudoLLA,
                                                        [self.get_pointer_type(data_layout)], cp), 0)
        else:
            cp_hi = DagValue(dag.add_constant_pool_node(
                self.get_pointer_type(data_layout), node.value, True, target_flags=RISCVOperandFlag.HI.value), 0)

            cp_lo = DagValue(dag.add_constant_pool_node(
                self.get_pointer_type(data_layout), node.value, True, target_flags=RISCVOperandFlag.LO.value), 0)

            cp_addr = DagValue(dag.add_machine_dag_node(RISCVMachineOps.LUI,
                                                        [MachineValueType(ValueType.I32)], cp_hi), 0)

            cp_addr = DagValue(dag.add_machine_dag_node(RISCVMachineOps.ADDI,
                                                        [MachineValueType(ValueType.I32)], cp_addr, cp_lo), 0)

        return dag.add_load_node(node.value_types[0], dag.entry, cp_addr, False)

    def lower_constant_pool(self, node: DagNode, dag: Dag):
        assert(isinstance(node, ConstantPoolDagNode))
        data_layout = dag.mfunc.func_info.func.module.data_layout

        target_constant_pool = dag.add_constant_pool_node(
            self.get_pointer_type(data_layout), node.value, True)
        return dag.add_node(RISCVDagOps.WRAPPER, node.value_types, DagValue(target_constant_pool, 0))

    def lower_build_vector(self, node: DagNode, dag: Dag):
        assert(node.opcode == VirtualDagOps.BUILD_VECTOR)
        operands = []
        for operand in node.operands:
            if operand.node == VirtualDagOps.CONSTANT_FP:
                target_constant_fp = dag.add_target_constant_fp_node(
                    operand.node.value_types[0], operand.node.value)
                operands.append(DagValue(target_constant_fp, 0))
            else:
                operands.append(operand)

        assert(len(operands) > 0)

        is_one_val = True
        for idx in range(1, len(node.operands)):
            if node.operands[0].node != node.operands[idx].node or node.operands[0].index != node.operands[idx].index:
                is_one_val = False
                break

        if is_one_val:
            return dag.add_node(RISCVDagOps.VDUP, node.value_types, operands[0])

        return dag.add_node(VirtualDagOps.BUILD_VECTOR, node.value_types, *operands)

    def lower_bitcast(self, node: DagNode, dag: Dag):
        if node.operands[0].ty in QPR.tys:
            if node.value_types[0] in QPR.tys:
                return dag.add_node(VirtualDagOps.OR, node.value_types, node.operands[0], node.operands[0])

        return node

    def lower(self, node: DagNode, dag: Dag):
        if node.opcode == VirtualDagOps.ENTRY:
            return dag.entry.node
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

    def get_register_type(self, vt):
        if vt in self.reg_type_for_vt:
            return self.reg_type_for_vt[vt]

        raise NotImplementedError()

    def get_register_count(self, vt):
        if vt in self.reg_count_for_vt:
            return self.reg_count_for_vt[vt]

        raise NotImplementedError()

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
                reg_vt = self.get_register_type(vt)
                reg_count = self.get_register_count(vt)

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
                    regclass = GPR
                elif arg_vt.value_type == ValueType.I16:
                    regclass = GPR
                elif arg_vt.value_type == ValueType.I32:
                    regclass = GPR
                elif arg_vt.value_type == ValueType.I64:
                    regclass = GPR
                elif arg_vt.value_type == ValueType.F32:
                    regclass = FPR32
                elif arg_vt.value_type == ValueType.F64:
                    regclass = FPR64
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
                elif loc_info == CCArgLocInfo.SExt:
                    arg_val_node = DagValue(
                        builder.g.add_node(VirtualDagOps.SIGN_EXTEND, [arg_val.vt], arg_val_node), 0)
                elif loc_info == CCArgLocInfo.ZExt:
                    arg_val_node = DagValue(
                        builder.g.add_node(VirtualDagOps.ZERO_EXTEND, [arg_val.vt], arg_val_node), 0)
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
                reg_vt = self.get_register_type(vt)
                reg_count = self.get_register_count(vt)

                arg_parts.append(get_copy_from_parts(
                    arg_vals[idx:idx+reg_count], reg_vt, vt, builder.g))

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
        if inst.opcode == RISCVMachineOps.ADJCALLSTACKDOWN:
            return True
        if inst.opcode == RISCVMachineOps.ADJCALLSTACKUP:
            return True

        return False

    def lower_prolog(self, func: MachineFunction, bb: MachineBasicBlock):
        inst_info = func.target_info.get_inst_info()
        frame_info = func.target_info.get_frame_lowering()
        reg_info = func.target_info.get_register_info()
        data_layout = func.func_info.func.module.data_layout

        front_inst = bb.insts[0]

        stack_size = func.frame.estimate_stack_size(
            RISCVMachineOps.ADJCALLSTACKDOWN, RISCVMachineOps.ADJCALLSTACKUP)

        max_align = max(func.frame.max_alignment, func.frame.stack_alignment)
        stack_size = func.frame.stack_size = int(
            int((stack_size + max_align - 1) / max_align) * max_align)

        alloc_frame = MachineInstruction(RISCVMachineOps.ADDI)
        alloc_frame.add_reg(MachineRegister(X2), RegState.Define)
        alloc_frame.add_reg(MachineRegister(X2), RegState.Non)
        alloc_frame.add_imm(-stack_size)

        alloc_frame.insert_before(front_inst)

        for cs_info in func.frame.calee_save_info:
            reg = cs_info.reg
            regclass = reg_info.get_regclass_from_reg(reg)
            frame_idx = cs_info.frame_idx

            inst_info.copy_reg_to_stack(MachineRegister(
                reg), frame_idx, regclass, front_inst)

        calc_fp = MachineInstruction(RISCVMachineOps.ADDI)
        calc_fp.add_reg(MachineRegister(X8), RegState.Define)
        calc_fp.add_reg(MachineRegister(X2), RegState.Non)
        calc_fp.add_imm(stack_size)

        calc_fp.insert_before(front_inst)

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

        stack_size = func.frame.estimate_stack_size(
            RISCVMachineOps.ADJCALLSTACKDOWN, RISCVMachineOps.ADJCALLSTACKUP)

        max_align = max(func.frame.max_alignment, func.frame.stack_alignment)
        stack_size = int(
            int((stack_size + max_align - 1) / max_align) * max_align)

        restore_frame = MachineInstruction(RISCVMachineOps.ADDI)
        restore_frame.add_reg(MachineRegister(X2), RegState.Define)
        restore_frame.add_reg(MachineRegister(X2), RegState.Non)
        restore_frame.add_imm(stack_size)

        restore_frame.insert_before(front_inst)

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
            return FPR32
        elif ty.value_type == ValueType.F64:
            return FPR64
        elif ty.value_type == ValueType.V4F32:
            raise NotImplementedError()

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


class RISCVTargetRegisterInfo(TargetRegisterInfo):
    def __init__(self):
        super().__init__()

    def get_reserved_regs(self):
        reserved = []
        reserved.extend([X0, X1, X2, X3, X4, X8])

        return reserved

    @property
    def allocatable_regs(self):
        regs = set()
        regs |= set(GPR.regs)
        regs |= set(FPR32.regs)
        regs |= set(FPR64.regs)

        return regs

    def get_callee_saved_regs(self):
        callee_save_regs = []
        callee_save_regs.extend(
            [X8, X9, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27])
        callee_save_regs.extend(
            [F8_F, F9_F, F18_F, F19_F, F20_F, F21_F, F22_F, F23_F, F24_F, F25_F, F26_F, F27_F])

        return callee_save_regs

    def get_callee_clobbered_regs(self):
        regs = [X10, X11, X12, X13, X14, X15, X16, X17,
                F10_F, F11_F, F12_F, F13_F, F14_F, F15_F, F16_F, F17_F]

        return regs

    def get_ordered_regs(self, regclass):
        reserved_regs = self.get_reserved_regs()

        free_regs = set(regclass.regs) - set(reserved_regs)

        return [reg for reg in regclass.regs if reg in free_regs]

    def is_legal_for_regclass(self, regclass, value_type):
        for ty in regclass.tys:
            if ty == value_type:
                return True

        return False

    def is_subclass(self, regclass, subclass):
        return False

    def get_minimum_regclass_from_reg(self, reg, vt):
        from codegen.spec import regclasses

        rc = None
        for regclass in regclasses:
            if self.is_legal_for_regclass(regclass, vt) and reg in regclass.regs:
                if not rc or self.is_subclass(rc, regclass):
                    rc = regclass

        if not rc:
            raise ValueError("Could not find the register class.")

        return rc

    def get_regclass_from_reg(self, reg):
        from codegen.spec import regclasses

        for regclass in regclasses:
            if reg in regclass.regs:
                return regclass

        raise ValueError("Could not find the register class.")

    def get_regclass_for_vt(self, vt):
        for regclass in riscv_regclasses:
            if vt in regclass.tys:
                return regclass

        raise ValueError("Could not find the register class.")

    def get_subreg(self, reg, subreg_idx):

        def find_subreg(reg2, subreg_idx):
            for subreg in reg2.subregs:
                if subreg.idx == subreg_idx:
                    return subreg.reg
            raise KeyError("subreg_idx")

        subreg_idx = subregs[subreg_idx]
        if isinstance(subreg_idx, ComposedSubRegDescription):
            reg = find_subreg(reg, subreg_idx.subreg_a)
            subreg_idx = subreg_idx.subreg_b
        return find_subreg(reg, subreg_idx)

    @property
    def frame_register(self):
        return X8


class RISCVFrameLowering(TargetFrameLowering):
    def __init__(self, alignment):
        super().__init__(alignment)

        self.frame_spill_size = 4

    @property
    def stack_grows_direction(self):
        return StackGrowsDirection.Down

    def determinate_callee_saves(self, func, regs):
        regs = super().determinate_callee_saves(func, regs)

        regs.append(X1)
        regs.append(X8)

        return regs


class RISCVLegalizer(Legalizer):
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
        if dag.mfunc.target_info.triple.arch == ArchType.RISCV64:
            setcc_ty = MachineValueType(ValueType.I64)
        else:
            setcc_ty = MachineValueType(ValueType.I32)

        return dag.add_node(node.opcode, [setcc_ty], *node.operands)

    def promote_integer_result_bin(self, node, dag, legalized):
        lhs = self.get_legalized_op(node.operands[0], legalized)
        rhs = self.get_legalized_op(node.operands[1], legalized)
        opcode = node.opcode

        if node.opcode == VirtualDagOps.SRA:
            opcode = RISCVDagOps.SRAW
        elif node.opcode == VirtualDagOps.SRL:
            opcode = RISCVDagOps.SRLW
        elif node.opcode == VirtualDagOps.SHL:
            opcode = RISCVDagOps.SLLW

        return dag.add_node(opcode, [lhs.ty], lhs, rhs)

    def promote_integer_result(self, node, dag, legalized):
        if node.opcode == VirtualDagOps.SETCC:
            return self.promote_integer_result_setcc(node, dag, legalized)
        elif node.opcode in [
                VirtualDagOps.ADD, VirtualDagOps.SUB, VirtualDagOps.MUL, VirtualDagOps.SDIV, VirtualDagOps.UDIV,
                VirtualDagOps.FADD, VirtualDagOps.FSUB, VirtualDagOps.FMUL, VirtualDagOps.FDIV,
                VirtualDagOps.SRL, VirtualDagOps.SRA, VirtualDagOps.SHL,
                VirtualDagOps.AND, VirtualDagOps.OR, VirtualDagOps.XOR]:
            return self.promote_integer_result_bin(node, dag, legalized)
        elif node.opcode == VirtualDagOps.LOAD:
            chain = node.operands[0]
            ptr = node.operands[1]
            return dag.add_load_node(MachineValueType(ValueType.I64), chain, ptr, False, mem_operand=node.mem_operand)
        elif node.opcode == VirtualDagOps.ZERO_EXTEND:
            return dag.add_node(node.opcode, [MachineValueType(ValueType.I64)], *node.operands)
        elif node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            return dag.add_constant_node(MachineValueType(ValueType.I64), node.value)

        return None

    def legalize_node_result(self, node: DagNode, dag: Dag, legalized):
        for vt in node.value_types:
            if vt.value_type == ValueType.I1:
                return self.promote_integer_result(node, dag, legalized)

            if vt.value_type == ValueType.I32:
                if dag.mfunc.target_info.triple.arch == ArchType.RISCV32:
                    return None

                return self.promote_integer_result(node, dag, legalized)

            if vt.value_type in [ValueType.V4F32]:
                return self.split_vector_result(node, dag, legalized)

        return None

    def split_vector_result_build_vec(self, node, dag, legalized):
        return tuple([operand.node for operand in node.operands])

    def split_vector_result_bin(self, node, dag, legalized):
        ops_lhs = self.get_legalized_op(node.operands[0], legalized)
        ops_rhs = self.get_legalized_op(node.operands[1], legalized)

        assert(len(ops_lhs) == len(ops_rhs))

        values = []
        for lhs_val, rhs_val in zip(ops_lhs, ops_rhs):

            values.append(dag.add_node(
                node.opcode, [rhs_val.ty], lhs_val, rhs_val))

        return tuple(values)

    def split_value_type(self, dag: Dag, vt):
        if vt.value_type == ValueType.V4F32:
            return [vt.get_vector_elem_type()] * vt.get_num_vector_elems()

        raise NotImplementedError()

    def split_vector_result_load(self, node, dag, legalized):
        ops_chain = node.operands[0]
        ops_ptr = node.operands[1]

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

        return None

    def split_vector_operand_store(self, node, i, dag: Dag, legalized):
        ops_chain = self.get_legalized_op(node.operands[0], legalized)
        ops_src_vec = self.get_legalized_op(node.operands[1], legalized)
        ops_ptr = node.operands[2]

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
                ops_chain, addr, ops_src, False, mem_operand=node.mem_operand), 0)
            chains.append(chain)

            ofs += elem_size

        return dag.add_node(VirtualDagOps.TOKEN_FACTOR, [MachineValueType(ValueType.OTHER)], *chains)

    def split_vector_operand_ext_vec_elt(self, node, i, dag: Dag, legalized):
        vec_op = self.get_legalized_op(node.operands[0], legalized)
        index_op = node.operands[1]

        assert(index_op.node.opcode == VirtualDagOps.TARGET_CONSTANT)
        index = index_op.node.value.value

        return vec_op[index]

    def promote_integer_operand_setcc(self, node, i, dag: Dag, legalized):
        lhs = self.get_legalized_op(node.operands[0], legalized)
        rhs = self.get_legalized_op(node.operands[1], legalized)
        condcode = node.operands[2]

        return dag.add_node(node.opcode, node.value_types, lhs, rhs, condcode)

    def promote_integer_operand_brcond(self, node, i, dag: Dag, legalized):
        chain_op = self.get_legalized_op(node.operands[0], legalized)
        cond_op = self.get_legalized_op(node.operands[1], legalized)
        dst_op = self.get_legalized_op(node.operands[2], legalized)

        return dag.add_node(VirtualDagOps.BRCOND, node.value_types, chain_op, cond_op, dst_op)

    def promote_integer_operand_copy_to_reg(self, node, i, dag: Dag, legalized):
        chain_op = node.operands[0]
        dst_op = node.operands[1]
        src_op = self.get_legalized_op(node.operands[2], legalized)

        return dag.add_node(VirtualDagOps.COPY_TO_REG, node.value_types, chain_op, dst_op, src_op)

    def legalize_node_operand(self, node, i, dag: Dag, legalized):
        operand = node.operands[i]
        vt = operand.ty

        if vt.value_type == ValueType.I1:
            if node.opcode == VirtualDagOps.BRCOND:
                return self.promote_integer_operand_brcond(
                    node, i, dag, legalized)

        if vt.value_type == ValueType.I32:
            if dag.mfunc.target_info.triple.arch == ArchType.RISCV32:
                return None

            if node.opcode == VirtualDagOps.COPY_TO_REG:
                return self.promote_integer_operand_copy_to_reg(
                    node, i, dag, legalized)

            if node.opcode == VirtualDagOps.ADD:
                return None

            if node.opcode == VirtualDagOps.SETCC:
                return self.promote_integer_operand_setcc(
                    node, i, dag, legalized)

            if node.opcode == RISCVDagOps.RETURN:
                ops = []
                for op in node.operands:
                    ops.append(self.get_legalized_op(op, legalized))
                return dag.add_node(node.opcode, node.value_types, *ops)

            if node.opcode == VirtualDagOps.STORE:
                op_chain = node.operands[0]
                op_val = self.get_legalized_op(node.operands[1], legalized)
                op_ptr = node.operands[2]
                return dag.add_store_node(op_chain, op_ptr, op_val, False, mem_operand=node.mem_operand)

            if node.opcode == VirtualDagOps.BRCOND:
                return self.promote_integer_operand_brcond(
                    node, dag, i, legalized)

            if node.opcode == VirtualDagOps.CALLSEQ_START:
                return None
            if node.opcode == VirtualDagOps.CALLSEQ_END:
                return None

            return None

        if vt.value_type == ValueType.V4F32:
            if node.opcode == VirtualDagOps.STORE:
                return self.split_vector_operand_store(
                    node, i, dag, legalized)
            if node.opcode == VirtualDagOps.EXTRACT_VECTOR_ELT:
                return self.split_vector_operand_ext_vec_elt(
                    node, i, dag, legalized)

            if node.opcode in [
                    VirtualDagOps.FADD, VirtualDagOps.FSUB, VirtualDagOps.FMUL, VirtualDagOps.FDIV]:
                return None

            if node.opcode == VirtualDagOps.INSERT_VECTOR_ELT:
                return None

            raise NotImplementedError()

        return None


class RISCVTargetInfo(TargetInfo):
    def __init__(self, triple):
        super().__init__(triple)

        self._inst_info = RISCVTargetInstInfo()
        self._lowering = RISCVTargetLowering(self)
        self._reg_info = RISCVTargetRegisterInfo()
        self._calling_conv = RISCVCallingConv()
        self._isel = RISCVInstructionSelector()
        self._legalizer = RISCVLegalizer()
        self._frame_lowering = RISCVFrameLowering(16)

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
        if self.triple.arch == ArchType.RISCV32:
            return RV32
        elif self.triple.arch == ArchType.RISCV64:
            return RV64

        raise ValueError("Invalid arch type")


class RISCVTargetMachine:
    def __init__(self, triple):
        self.triple = triple

    def get_target_info(self, func: Function):
        return RISCVTargetInfo(self.triple)

    def add_mc_emit_passes(self, pass_manager, mccontext, output, is_asm):
        from codegen.riscv_asm_printer import RISCVAsmInfo, MCAsmStream, RISCVCodeEmitter, RISCVAsmBackend, RISCVAsmPrinter
        from codegen.elf import ELFObjectStream, ELFObjectWriter, RISCVELFObjectWriter

        objformat = self.triple.objformat

        mccontext.asm_info = RISCVAsmInfo()
        if is_asm:
            raise NotImplementedError()
        else:
            emitter = RISCVCodeEmitter(mccontext)
            backend = RISCVAsmBackend()

            if objformat == ObjectFormatType.ELF:
                target_writer = RISCVELFObjectWriter()

                if self.triple.arch == ArchType.RISCV64:
                    target_writer.is_64bit = True

                writer = ELFObjectWriter(output, target_writer)
                stream = ELFObjectStream(mccontext, backend, writer, emitter)

        pass_manager.passes.append(RISCVAsmPrinter(stream))
