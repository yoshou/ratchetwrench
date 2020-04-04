#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.spec import *
from codegen.mir_emitter import *
from codegen.isel import *
from codegen.x64_def import *
from codegen.matcher import *


class X64OperandFlag(IntFlag):
    NO_FLAG = auto()

    # GOT_ABSOLUTE_ADDRESS - On a symbol operand = auto() this represents a
    # relocation of:
    #    SYMBOL_LABEL + [. - PICBASELABEL]
    GOT_ABSOLUTE_ADDRESS = auto()

    # PIC_BASE_OFFSET - On a symbol operand this indicates that the
    # immediate should get the value of the symbol minus the PIC base label:
    #    SYMBOL_LABEL - PICBASELABEL
    PIC_BASE_OFFSET = auto()

    # GOT - On a symbol operand this indicates that the immediate is the
    # offset to the GOT entry for the symbol name from the base of the GOT.
    #
    # See the X86-64 ELF ABI supplement for more details.
    #    SYMBOL_LABEL @GOT
    GOT = auto()

    # GOTOFF - On a symbol operand this indicates that the immediate is
    # the offset to the location of the symbol name from the base of the GOT.
    #
    # See the X86-64 ELF ABI supplement for more details.
    #    SYMBOL_LABEL @GOTOFF
    GOTOFF = auto()

    # GOTPCREL - On a symbol operand this indicates that the immediate is
    # offset to the GOT entry for the symbol name from the current code
    # location.
    #
    # See the X86-64 ELF ABI supplement for more details.
    #    SYMBOL_LABEL @GOTPCREL
    GOTPCREL = auto()

    # PLT - On a symbol operand this indicates that the immediate is
    # offset to the PLT entry of symbol name from the current code location.
    #
    # See the X86-64 ELF ABI supplement for more details.
    #    SYMBOL_LABEL @PLT
    PLT = auto()

    # TLSGD - On a symbol operand this indicates that the immediate is
    # the offset of the GOT entry with the TLS index structure that contains
    # the module number and variable offset for the symbol. Used in the
    # general dynamic TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @TLSGD
    TLSGD = auto()

    # TLSLD - On a symbol operand this indicates that the immediate is
    # the offset of the GOT entry with the TLS index for the module that
    # contains the symbol. When this index is passed to a call to
    # __tls_get_addr = auto() the function will return the base address of the TLS
    # block for the symbol. Used in the x86-64 local dynamic TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @TLSLD
    TLSLD = auto()

    # TLSLDM - On a symbol operand this indicates that the immediate is
    # the offset of the GOT entry with the TLS index for the module that
    # contains the symbol. When this index is passed to a call to
    # ___tls_get_addr = auto() the function will return the base address of the TLS
    # block for the symbol. Used in the IA32 local dynamic TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @TLSLDM
    TLSLDM = auto()

    # GOTTPOFF - On a symbol operand this indicates that the immediate is
    # the offset of the GOT entry with the thread-pointer offset for the
    # symbol. Used in the x86-64 initial exec TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @GOTTPOFF
    GOTTPOFF = auto()

    # INDNTPOFF - On a symbol operand this indicates that the immediate is
    # the absolute address of the GOT entry with the negative thread-pointer
    # offset for the symbol. Used in the non-PIC IA32 initial exec TLS access
    # model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @INDNTPOFF
    INDNTPOFF = auto()

    # TPOFF - On a symbol operand this indicates that the immediate is
    # the thread-pointer offset for the symbol. Used in the x86-64 local
    # exec TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @TPOFF
    TPOFF = auto()

    # DTPOFF - On a symbol operand this indicates that the immediate is
    # the offset of the GOT entry with the TLS offset of the symbol. Used
    # in the local dynamic TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @DTPOFF
    DTPOFF = auto()

    # NTPOFF - On a symbol operand this indicates that the immediate is
    # the negative thread-pointer offset for the symbol. Used in the IA32
    # local exec TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @NTPOFF
    NTPOFF = auto()

    # GOTNTPOFF - On a symbol operand this indicates that the immediate is
    # the offset of the GOT entry with the negative thread-pointer offset for
    # the symbol. Used in the PIC IA32 initial exec TLS access model.
    #
    # See 'ELF Handling for Thread-Local Storage' for more details.
    #    SYMBOL_LABEL @GOTNTPOFF
    GOTNTPOFF = auto()

    # DLLIMPORT - On a symbol operand "FOO" = auto() this indicates that the
    # reference is actually to the "__imp_FOO" symbol.  This is used for
    # dllimport linkage on windows.
    DLLIMPORT = auto()

    # DARWIN_NONLAZY - On a symbol operand "FOO" = auto() this indicates that the
    # reference is actually to the "FOO$non_lazy_ptr" symbol = auto() which is a
    # non-PIC-base-relative reference to a non-hidden dyld lazy pointer stub.
    DARWIN_NONLAZY = auto()

    # DARWIN_NONLAZY_PIC_BASE - On a symbol operand "FOO" = auto() this indicates
    # that the reference is actually to "FOO$non_lazy_ptr - PICBASE" = auto() which is
    # a PIC-base-relative reference to a non-hidden dyld lazy pointer stub.
    DARWIN_NONLAZY_PIC_BASE = auto()

    # TLVP - On a symbol operand this indicates that the immediate is
    # some TLS offset.
    #
    # This is the TLS offset for the Darwin TLS mechanism.
    TLVP = auto()

    # TLVP_PIC_BASE - On a symbol operand this indicates that the immediate
    # is some TLS offset from the picbase.
    #
    # This is the 32-bit TLS offset for Darwin TLS in PIC mode.
    TLVP_PIC_BASE = auto()

    # SECREL - On a symbol operand this indicates that the immediate is
    # the offset from beginning of section.
    #
    # This is the TLS offset for the COFF/Windows TLS mechanism.
    SECREL = auto()

    # ABS8 - On a symbol operand this indicates that the symbol is known
    # to be an absolute symbol in range [0 = auto()128) = auto() so we can use the @ABS8
    # symbol modifier.
    ABS8 = auto()

    # COFFSTUB - On a symbol operand "FOO" = auto() this indicates that the
    # reference is actually to the ".refptr.FOO" symbol.  This is used for
    # stub symbols on windows.
    COFFSTUB = auto()


def is_null_constant(value):
    return isinstance(value.node, ConstantDagNode) and value.node.is_zero


def is_null_fp_constant(value):
    return isinstance(value.node, ConstantFPDagNode) and value.node.is_zero


def is_x86_zero(value):
    return is_null_constant(value) or is_null_fp_constant(value)


class X64InstructionSelector(InstructionSelector):
    def __init__(self):
        super().__init__()

    def lower_wrapper_rip(self, node, dag):
        noreg = MachineRegister(NOREG)
        MVT = MachineValueType

        ty = node.value_types[0]

        base = DagValue(dag.add_register_node(
            MVT(ValueType.I64), MachineRegister(RIP)), 0)
        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        disp = node.operands[0]
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        lea_ops = (base, scale, index, disp, segment)

        if ty == MachineValueType(ValueType.I64):
            lea_operand = X64MachineOps.LEA64r
        elif ty == MachineValueType(ValueType.I32):
            lea_operand = X64MachineOps.LEA32r
        else:
            raise ValueError()

        return dag.add_machine_dag_node(lea_operand, node.value_types, *lea_ops)

    def get_memory_operands(self, dag: Dag, operand: DagValue):
        assert(isinstance(operand, DagValue))

        noreg = MachineRegister(NOREG)
        MVT = MachineValueType

        if operand.node.opcode == VirtualDagOps.ADD:
            sub_op1 = operand.node.operands[0]
            sub_op2 = operand.node.operands[1]

            if sub_op2.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
                if sub_op1.node.opcode == X64DagOps.WRAPPER_RIP:
                    base = DagValue(dag.add_target_register_node(
                        MVT(ValueType.I64), RIP), 0)
                    assert(sub_op2.node.value == 0)
                    disp = sub_op1.node.operands[0]
                else:
                    base = sub_op1
                    disp = sub_op2
            elif sub_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
                if sub_op2.node.opcode == X64DagOps.WRAPPER_RIP:
                    base = DagValue(dag.add_target_register_node(
                        MVT(ValueType.I64), RIP), 0)
                    assert(sub_op1.node.value == 0)
                    disp = sub_op2.node.operands[0]
                else:
                    base = sub_op2
                    disp = sub_op1
            else:
                raise ValueError()

            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

            return (base, scale, index, disp, segment)
        elif operand.node.opcode == VirtualDagOps.SUB:
            sub_op1 = operand.node.operands[0]
            sub_op2 = operand.node.operands[1]
            if sub_op2.node.opcode == VirtualDagOps.CONSTANT:
                base = sub_op1
                disp = DagValue(dag.add_target_constant_node(
                    sub_op2.node.value_types[0], -sub_op2.node.value), 0)
            elif sub_op1.node.opcode == VirtualDagOps.CONSTANT:
                base = sub_op2
                disp = DagValue(dag.add_target_constant_node(
                    sub_op1.node.value_ty[0], -sub_op1.node.value), 0)
            elif sub_op1.node.opcode == VirtualDagOps.CONSTANT:
                base = operand
                disp = DagValue(dag.add_target_constant_node(
                    MVT(ValueType.I32), 0), 0)

            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

            return (base, scale, index, disp, segment)
        elif operand.node.opcode == X64DagOps.WRAPPER_RIP:
            base = DagValue(dag.add_register_node(
                MVT(ValueType.I64), MachineRegister(RIP)), 0)
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            disp = operand.node.operands[0]
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

            return (base, scale, index, disp, segment)
        elif operand.node.opcode == X64DagOps.WRAPPER:
            base = DagValue(dag.add_register_node(
                MVT(ValueType.I64), MachineRegister(RIP)), 0)
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            disp = operand.node.operands[0]
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            return (base, scale, index, disp, segment)
        elif operand.node.opcode == VirtualDagOps.FRAME_INDEX:
            base = operand
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            disp = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I32), 0), 0)
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

            return (base, scale, index, disp, segment)
        else:
            base = operand
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            disp = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I32), 0), 0)
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

            return (base, scale, index, disp, segment)

        raise ValueError()

    def select_load(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        src = new_ops[1]

        base, scale, index, disp, segment = self.get_memory_operands(dag, src)

        ops = [base, scale, index, disp, segment, chain]

        if node.value_types[0] == MachineValueType(ValueType.I64):
            operand = X64MachineOps.MOV64rm
        elif node.value_types[0] == MachineValueType(ValueType.I32):
            operand = X64MachineOps.MOV32rm
        elif node.value_types[0] == MachineValueType(ValueType.F64):
            operand = X64MachineOps.MOVSDrm
        elif node.value_types[0] == MachineValueType(ValueType.F32):
            operand = X64MachineOps.MOVSSrm
        elif node.value_types[0] == MachineValueType(ValueType.V4F32):
            operand = X64MachineOps.MOVAPSrm
        else:
            raise ValueError()

        return dag.add_machine_dag_node(operand, node.value_types, *ops)

    def select_store(self, node: DagNode, dag: Dag, new_ops):
        assert(node.operands[0].ty.value_type == ValueType.OTHER)

        chain = new_ops[0]
        src = new_ops[1]
        dst = new_ops[2]

        base, scale, index, disp, segment = self.get_memory_operands(dag, dst)

        if isinstance(src.node, ConstantDagNode):
            ops = [base, scale, index, disp, segment, src, chain]
            return dag.add_machine_dag_node(X64MachineOps.MOV32mi, node.value_types, *ops)
        elif src.node.opcode in [VirtualDagOps.TARGET_FRAME_INDEX, VirtualDagOps.FRAME_INDEX]:
            lea_ops = self.get_memory_operands(dag, src)

            if src.ty == MachineValueType(ValueType.I64):
                lea_operand = X64MachineOps.LEA64r
            elif src.ty == MachineValueType(ValueType.I32):
                lea_operand = X64MachineOps.LEA32r
            else:
                raise ValueError()
            src = DagValue(dag.add_machine_dag_node(
                lea_operand, [src.ty], *lea_ops), 0)
            ops = [base, scale, index, disp, segment, src, chain]

            if src.ty == MachineValueType(ValueType.I64):
                operand = X64MachineOps.MOV64mr
            elif src.ty == MachineValueType(ValueType.I32):
                operand = X64MachineOps.MOV32mr
            else:
                raise ValueError()
            return dag.add_machine_dag_node(operand, node.value_types, *ops)
        elif isinstance(src.node, DagNode):
            ops = [base, scale, index, disp, segment, src, chain]

            if src.ty == MachineValueType(ValueType.I64):
                operand = X64MachineOps.MOV64mr
            elif src.ty == MachineValueType(ValueType.I32):
                operand = X64MachineOps.MOV32mr
            elif src.ty == MachineValueType(ValueType.F64):
                operand = X64MachineOps.MOVSDmr
            elif src.ty == MachineValueType(ValueType.F32):
                operand = X64MachineOps.MOVSSmr
            elif src.ty == MachineValueType(ValueType.V4F32):
                operand = X64MachineOps.MOVAPSmr
            else:
                raise ValueError()
            return dag.add_machine_dag_node(operand, node.value_types, *ops)

    def select_add(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            noreg = MachineRegister(NOREG)

            MVT = MachineValueType

            base = op1
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            disp = op2
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)

            ops = [base, scale, index, disp, segment]

            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.LEA32r
            elif node.value_types[0] == MachineValueType(ValueType.I64):
                opcode = X64MachineOps.LEA64r
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, *ops)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.ADD32ri
            elif node.value_types[0] == MachineValueType(ValueType.I64):
                opcode = X64MachineOps.ADD64ri
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.ADD32rr
            elif node.value_types[0] == MachineValueType(ValueType.I64):
                opcode = X64MachineOps.ADD64rr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_add")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_fadd(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.F32):
                imm_val = DagValue(dag.add_target_constant_node(
                    ValueType.F32, op2.node.value), 0)
            elif node.value_types[0] == MachineValueType(ValueType.F64):
                imm_val = DagValue(dag.add_target_constant_node(
                    ValueType.F64, op2.node.value), 0)
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.F32):
                opcode = X64MachineOps.ADDSSrr
            elif node.value_types[0] == MachineValueType(ValueType.F64):
                opcode = X64MachineOps.ADDSDrr
            elif node.value_types[0] == MachineValueType(ValueType.V4F32):
                opcode = X64MachineOps.ADDPSrr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_add")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_fsub(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.F32):
                imm_val = DagValue(dag.add_target_constant_node(
                    ValueType.F32, op2.node.value), 0)
            elif node.value_types[0] == MachineValueType(ValueType.F64):
                imm_val = DagValue(dag.add_target_constant_node(
                    ValueType.F64, op2.node.value), 0)
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.F32):
                opcode = X64MachineOps.SUBSSrr
            elif node.value_types[0] == MachineValueType(ValueType.F64):
                opcode = X64MachineOps.SUBSDrr
            elif node.value_types[0] == MachineValueType(ValueType.V4F32):
                opcode = X64MachineOps.SUBPSrr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_add")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_sub(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.SUB32mi
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.SUB32ri
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.SUB32rr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_sub")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_fmul(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.F32):
                opcode = X64MachineOps.MULSSrr
            elif node.value_types[0] == MachineValueType(ValueType.F64):
                opcode = X64MachineOps.MULSDrr
            elif node.value_types[0] == MachineValueType(ValueType.V4F32):
                opcode = X64MachineOps.MULPSrr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_add")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_fdiv(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.F32):
                opcode = X64MachineOps.DIVSSrr
            elif node.value_types[0] == MachineValueType(ValueType.F64):
                opcode = X64MachineOps.DIVSDrr
            elif node.value_types[0] == MachineValueType(ValueType.V4F32):
                opcode = X64MachineOps.DIVPSrr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_add")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_and(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.AND32ri
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.AND32ri
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.I8):
                opcode = X64MachineOps.AND8rr
            elif node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.AND32rr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_and")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_or(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            return dag.add_machine_dag_node(X64MachineOps.OR32ri, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            return dag.add_machine_dag_node(X64MachineOps.OR32ri, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            return dag.add_machine_dag_node(X64MachineOps.OR32rr, node.value_types, op1, op2)

        print("select_or")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_xor(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            return dag.add_machine_dag_node(X64MachineOps.XOR32ri, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            return dag.add_machine_dag_node(X64MachineOps.XOR32ri, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            return dag.add_machine_dag_node(X64MachineOps.XOR32rr, node.value_types, op1, op2)

        print("select_xor")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_srl(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            return dag.add_machine_dag_node(X64MachineOps.SHR32ri, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            subreg_idx_node = dag.add_target_constant_node(ValueType.I32, 1)

            extract_subreg_node = dag.add_machine_dag_node(TargetDagOps.EXTRACT_SUBREG, [MachineValueType(ValueType.OTHER)],
                                                           op2, DagValue(subreg_idx_node, 0))

            cl = dag.add_target_register_node(ValueType.I8, CL)

            copy_to_cl_node = dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(ValueType.OTHER)],
                                           dag.entry, DagValue(cl, 0), DagValue(extract_subreg_node, 0))
            return dag.add_machine_dag_node(X64MachineOps.SHR32rCL, node.value_types, op1, DagValue(copy_to_cl_node, 0))

        print("select_and")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_shl(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            return dag.add_machine_dag_node(X64MachineOps.SHL32ri, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            subreg_idx_node = dag.add_target_constant_node(ValueType.I32, 1)

            extract_subreg_node = dag.add_machine_dag_node(TargetDagOps.EXTRACT_SUBREG, [MachineValueType(ValueType.OTHER)],
                                                           op2, DagValue(subreg_idx_node, 0))

            cl = dag.add_target_register_node(ValueType.I8, CL)

            copy_to_cl_node = dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(ValueType.OTHER)],
                                           dag.entry, DagValue(cl, 0), DagValue(extract_subreg_node, 0))
            return dag.add_machine_dag_node(X64MachineOps.SHL32rCL, node.value_types, op1, DagValue(copy_to_cl_node, 0))

        print("select_and")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_br(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        dest = new_ops[1]

        return dag.add_machine_dag_node(X64MachineOps.JMP_1, node.value_types, dest, chain)

    def select_brcond(self, node: DagNode, dag: Dag, new_ops):
        chain = node.operands[0]
        dest = node.operands[1]
        condcode = node.operands[2]
        cond = node.operands[3]

        eflags = dag.add_target_register_node(
            MachineValueType(ValueType.I32), EFLAGS)

        copy_cc_node = dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)],
                                    chain, DagValue(eflags, 0), cond)

        return dag.add_machine_dag_node(X64MachineOps.JCC_1, node.value_types, dest, condcode, DagValue(copy_cc_node, 0), DagValue(copy_cc_node, 1))

    def select_setcc(self, node: DagNode, dag: Dag, new_ops):
        condcode = new_ops[0]
        cond = new_ops[1]

        if isinstance(cond.node, DagNode):
            eflags = dag.add_target_register_node(
                MachineValueType(ValueType.I32), EFLAGS)

            copy_cc_node = dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(ValueType.OTHER)],
                                        dag.entry, DagValue(eflags, 0), cond)

            return dag.add_machine_dag_node(X64MachineOps.SETCCr, node.value_types, condcode, DagValue(copy_cc_node, 0))

        raise NotImplementedError()

    def select_bitcast(self, node: DagNode, dag: Dag, new_ops):
        src = new_ops[0]

        raise NotImplementedError()

    def select_callseq_start(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        opt = dag.add_target_constant_node(MachineValueType(ValueType.I32), 0)
        return dag.add_machine_dag_node(X64MachineOps.ADJCALLSTACKDOWN32, node.value_types, in_bytes, out_bytes, DagValue(opt, 0), chain)

    def select_callseq_end(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        in_bytes = new_ops[1]
        out_bytes = new_ops[2]
        glue = self.get_glue(new_ops)

        ops = [in_bytes, out_bytes, chain]
        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(X64MachineOps.ADJCALLSTACKUP32, node.value_types, *ops)

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

        return dag.add_machine_dag_node(X64MachineOps.CALL, node.value_types, *ops)

    def select_return(self, node: DagNode, dag: Dag, new_ops):
        chain = new_ops[0]
        ops = new_ops[1:]
        return dag.add_machine_dag_node(X64MachineOps.RET, node.value_types, *ops, chain)

    def select_x64_sub(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.SUB32mi
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.SUB32ri
            elif node.value_types[0] == MachineValueType(ValueType.I8):
                opcode = X64MachineOps.SUB8ri
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            if node.value_types[0] == MachineValueType(ValueType.I32):
                opcode = X64MachineOps.SUB32rr
            else:
                raise NotImplementedError()
            return dag.add_machine_dag_node(opcode, node.value_types, op1, op2)

        print("select_x64_sub")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_x64_cmp(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            ty = op1.ty
            if ty == MachineValueType(ValueType.F32):
                opcode = X64MachineOps.UCOMISSrr
            elif ty == MachineValueType(ValueType.F64):
                opcode = X64MachineOps.UCOMISDrr
            else:
                raise NotImplementedError()

            return dag.add_machine_dag_node(opcode, [MachineValueType(ValueType.I32)], *new_ops)

        print("select_x64_sub")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_x64_movss(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            return dag.add_machine_dag_node(X64MachineOps.MOVSSrr, node.value_types, *new_ops)

        print("select_x64_movss")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

    def select_x64_shufp(self, node: DagNode, dag: Dag, new_ops):
        op1 = new_ops[0]
        op2 = new_ops[1]
        op3 = new_ops[2]

        assert(isinstance(op3.node, ConstantDagNode))

        if isinstance(op1.node, FrameIndexDagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, ConstantDagNode):
            raise NotImplementedError()
        elif isinstance(op1.node, DagNode) and isinstance(op2.node, DagNode):
            return dag.add_machine_dag_node(X64MachineOps.SHUFPSrri, node.value_types, *new_ops)

        print("select_x64_shufp")
        print([edge.node for edge in new_ops])
        raise NotImplementedError()

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

        if src.node.opcode == VirtualDagOps.CONSTANT:
            src = DagValue(self.select_constant(src.node, dag, []), 0)
        elif src.node.opcode == VirtualDagOps.FRAME_INDEX:
            lea_ops = self.get_memory_operands(dag, src)

            if src.ty == MachineValueType(ValueType.I64):
                lea_operand = X64MachineOps.LEA64r
            elif src.ty == MachineValueType(ValueType.I32):
                lea_operand = X64MachineOps.LEA32r
            else:
                raise ValueError()
            src = DagValue(dag.add_machine_dag_node(
                lea_operand, [src.ty], *lea_ops), 0)

        glue = self.get_glue(new_ops)

        ops = [chain, dest, src]
        if glue:
            ops.append(glue)

        return dag.add_node(VirtualDagOps.COPY_TO_REG, node.value_types, *ops)

    def select_code(self, node: DagNode, dag: Dag):
        ops_table = [op for op in X64MachineOps.insts()]

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

        for pattern in x64_patterns:
            _, res = pattern.match(node, dag)
            if res:
                return pattern.construct(node, dag, res).node

        return None

    def select_constant(self, node: DagNode, dag: Dag, new_ops):
        value = DagValue(dag.add_target_constant_node(
            node.value_types[0], node.value), 0)
        ops = [value]

        if node.value_types[0] == MachineValueType(ValueType.I64):
            operand = X64MachineOps.MOV64ri
        elif node.value_types[0] == MachineValueType(ValueType.I32):
            operand = X64MachineOps.MOV32ri
        else:
            raise ValueError()

        return dag.add_machine_dag_node(operand, node.value_types, *ops)

    def select_build_vector(self, node: DagNode, dag: Dag, new_ops):
        for operand in node.operands:
            assert(is_x86_zero(operand))

        ops = []

        if node.value_types[0] == MachineValueType(ValueType.V4F32):
            operand = X64MachineOps.V_SET0
        else:
            raise ValueError()

        return dag.add_machine_dag_node(operand, node.value_types, *ops)

    def select_scalar_to_vector(self, node: DagNode, dag: Dag, new_ops):

        in_type = node.operands[0].ty
        out_type = node.value_types[0]

        if in_type == MachineValueType(ValueType.F32) and out_type == MachineValueType(ValueType.V4F32):
            regclass_id = regclasses.index(VR128)
            regclass_id_val = DagValue(dag.add_target_constant_node(
                MachineValueType(ValueType.I32), regclass_id), 0)
            return dag.add_node(TargetDagOps.COPY_TO_REGCLASS, node.value_types, node.operands[0], regclass_id_val)

        raise ValueError()

    def select(self, node: DagNode, dag: Dag):
        new_ops = node.operands

        matched = self.select_code(node, dag)
        if matched:
            return matched

        SELECT_TABLE = {
            VirtualDagOps.COPY_FROM_REG: self.select_copy_from_reg,
            VirtualDagOps.COPY_TO_REG: self.select_copy_to_reg,
            # VirtualDagOps.LOAD: self.select_load,
            # VirtualDagOps.STORE: self.select_store,
            # VirtualDagOps.ADD: self.select_add,
            # VirtualDagOps.SUB: self.select_sub,

            # VirtualDagOps.AND: self.select_and,
            # VirtualDagOps.OR: self.select_or,
            # VirtualDagOps.XOR: self.select_xor,
            # VirtualDagOps.SRL: self.select_srl,
            # VirtualDagOps.SHL: self.select_shl,

            # VirtualDagOps.FADD: self.select_fadd,
            # VirtualDagOps.FSUB: self.select_fsub,
            # VirtualDagOps.FMUL: self.select_fmul,
            # VirtualDagOps.FDIV: self.select_fdiv,

            VirtualDagOps.BITCAST: self.select_bitcast,
            # VirtualDagOps.BR: self.select_br,
            VirtualDagOps.CALLSEQ_START: self.select_callseq_start,
            VirtualDagOps.CALLSEQ_END: self.select_callseq_end,
            # VirtualDagOps.BUILD_VECTOR: self.select_build_vector,
            VirtualDagOps.SCALAR_TO_VECTOR: self.select_scalar_to_vector,
            # VirtualDagOps.CONSTANT: self.select_constant,
            # X64DagOps.SUB: self.select_x64_sub,
            # X64DagOps.CMP: self.select_x64_cmp,
            # X64DagOps.MOVSS: self.select_x64_movss,
            # X64DagOps.SHUFP: self.select_x64_shufp,
            # X64DagOps.SETCC: self.select_setcc,
            # X64DagOps.BRCOND: self.select_brcond,
            X64DagOps.CALL: self.select_call,
            X64DagOps.RETURN: self.select_return,
        }

        if node.opcode == VirtualDagOps.ENTRY:
            return dag.entry.node
        # elif node.opcode == VirtualDagOps.FRAME_INDEX:
        #     return node
        elif node.opcode == VirtualDagOps.UNDEF:
            return node
        elif node.opcode == VirtualDagOps.CONSTANT:
            return dag.add_target_constant_node(node.value_types[0], node.value)
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
        elif node.opcode == X64DagOps.WRAPPER_RIP:
            return self.lower_wrapper_rip(node, dag)
        elif node.opcode == X64DagOps.WRAPPER:
            return node
        elif node.opcode == VirtualDagOps.TARGET_CONSTANT_FP:
            return node
        elif node.opcode in SELECT_TABLE:
            select_func = SELECT_TABLE[node.opcode]
            minst = select_func(node, dag, new_ops)
        else:
            raise NotImplementedError(
                "Can't select the instruction: {}".format(node.opcode))

        return minst


class X86CallingConv(CallingConv):
    def __init__(self):
        pass

    @property
    def id(self):
        return CallingConvID.C

    def can_lower_return(self, func: Function):
        return_size, align = func.module.data_layout.get_type_size_in_bits(
            func.vty.return_ty)
        return return_size / 8 <= 16

    def lower_return(self, builder: DagBuilder, inst: ReturnInst, g: Dag):
        mfunc = builder.mfunc
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
                ret_reg = EAX
            elif ret_val.ty == MachineValueType(ValueType.I64):
                ret_reg = RAX
            else:
                raise NotImplementedError()

            reg_node = DagValue(
                g.add_target_register_node(ret_val.ty, ret_reg), 0)

            node = g.add_copy_to_reg_node(reg_node, ret_val)
            builder.root = DagValue(node, 0)

            ops = [builder.root, stack_pop_bytes, reg_node]

        node = g.add_node(X64DagOps.RETURN, [
                          MachineValueType(ValueType.OTHER)], *ops)

        builder.root = DagValue(node, 0)

        return node

    def compute_type_size_aligned(self, ty, data_layout: DataLayout):
        return data_layout.get_type_size_in_bits(ty)

    def lower_call(self, builder: DagBuilder, inst: CallInst, g: Dag):
        mfunc = builder.mfunc
        func = inst.callee
        calling_conv = mfunc.target_info.get_calling_conv()
        reg_info = mfunc.target_info.get_register_info()
        data_layout = builder.data_layout

        target_lowering = mfunc.target_info.get_lowering()

        ptr_ty = target_lowering.get_frame_index_type(data_layout)

        # Handle arguments
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

        # Estimate stack size to call function
        data_layout = builder.data_layout
        stack_bytes = 0
        for arg in inst.args:
            size, align = self.compute_type_size_aligned(arg.ty, data_layout)
            arg_size = int(size / 8)
            arg_align = int(align / 8)
            stack_bytes += arg_size

        in_bytes = g.add_target_constant_node(
            MachineValueType(ValueType.I32), stack_bytes)
        out_bytes = g.add_target_constant_node(
            MachineValueType(ValueType.I32), 0)

        callseq_start_node = g.add_node(VirtualDagOps.CALLSEQ_START, [
            MachineValueType(ValueType.OTHER)], builder.root, DagValue(in_bytes, 0), DagValue(out_bytes, 0))

        builder.root = DagValue(callseq_start_node, 0)

        stack_ptr_type = MachineValueType(ValueType.I64)
        esp_reg_node = g.add_target_register_node(stack_ptr_type, RSP)

        esp = g.add_copy_from_reg_node(
            stack_ptr_type, DagValue(esp_reg_node, 0))

        ##
        arg_parts = []
        for arg in inst.args:
            idx = 0
            arg_value = builder.get_value(arg)
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
            reg_val = DagValue(g.add_target_register_node(
                arg_val.vt, arg_val.reg), 0)
            copy_val = arg_parts[idx]

            if arg_val.loc_info == CCArgLocInfo.Full:
                pass
            elif arg_val.loc_info == CCArgLocInfo.Indirect:
                arg_mem_size = arg_val.vt.get_size_in_byte()
                arg_mem_align = int(data_layout.get_pref_type_alignment(
                    arg_val.vt.get_ir_type()) / 8)
                arg_mem_frame_idx = mfunc.frame.create_stack_object(
                    arg_mem_size, arg_mem_align)
                arg_mem_val = DagValue(builder.g.add_frame_index_node(
                    ptr_ty, arg_mem_frame_idx), 0)

                chain = DagValue(g.add_store_node(
                    g.root, arg_mem_val, copy_val), 0)
                builder.root = chain
                copy_val = arg_mem_val
            else:
                raise ValueError()

            operands = [builder.root, reg_val, copy_val]
            if idx != 0:
                operands.append(copy_to_reg_chain.get_value(1))

            copy_to_reg_chain = DagValue(builder.g.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(
                ValueType.OTHER), MachineValueType(ValueType.GLUE)], *operands), 0)

            builder.root = copy_to_reg_chain
            reg_vals.append(reg_val)

        func_address = builder.get_or_create_global_address(inst.callee, True)

        ops = [builder.root, func_address]
        if len(ccstate.values) > 0:
            ops.append(copy_to_reg_chain.get_value(1))

        call_node = g.add_node(
            X64DagOps.CALL, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)], *ops)

        builder.root = DagValue(call_node, 0)

        callseq_end_node = g.add_node(VirtualDagOps.CALLSEQ_END, [
            MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)], builder.root, DagValue(in_bytes, 0), DagValue(out_bytes, 0), DagValue(call_node, 1))

        builder.root = DagValue(callseq_end_node, 0)

        # Handle returns
        return_offsets = []
        return_vts = compute_value_types(inst.ty, data_layout, return_offsets)

        returns = []
        if not self.can_lower_return(func):
            raise NotImplementedError()
        else:
            offset_in_arg = 0

            for val_idx, vt in enumerate(return_vts):
                reg_vt = reg_info.get_register_type(vt)
                reg_count = reg_info.get_register_count(vt)

                for reg_idx in range(reg_count):
                    flags = CCArgFlags()

                    returns.append(CallingConvReturn(
                        vt, reg_vt, 0, offset_in_arg, flags))

                    offset_in_arg += reg_vt.get_size_in_byte()

        ccstate = CallingConvState(calling_conv, mfunc)
        ccstate.compute_returns_layout(returns)

        glue_val = DagValue(callseq_end_node, 1)

        # Handle return values
        ret_vals = []
        for idx, ret_val in enumerate(ccstate.values):
            assert(isinstance(ret_val, CCArgReg))
            reg = MachineRegister(ret_val.reg)

            reg_node = DagValue(
                builder.g.add_register_node(ret_val.vt, reg), 0)
            ret_val_node = DagValue(builder.g.add_node(VirtualDagOps.COPY_FROM_REG, [
                                    ret_val.vt, MachineValueType(ValueType.GLUE)], builder.root, reg_node, glue_val), 0)
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

        if len(ret_parts) == 0:
            return None

        return builder.g.add_merge_values(ret_parts)

    def allocate_return_x64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type == ValueType.I1:
            loc_vt = MachineValueType(ValueType.I8)

        if loc_vt.value_type == ValueType.I8:
            regs = [AL, DL, CL]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I16:
            regs = [AX, CX, DX]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs = [EAX, ECX, EDX]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs = [RAX, RCX, RDX]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.F32, ValueType.F64]:
            regs = [XMM0, XMM1, XMM2]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.V4F32]:
            regs = [XMM0, XMM1, XMM2]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_return_win64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type == ValueType.I1:
            loc_vt = MachineValueType(ValueType.I8)

        if loc_vt.value_type == ValueType.I8:
            regs = [AL, DL, CL]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I16:
            regs = [AX, CX, DX]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs = [EAX, ECX, EDX]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs = [RAX, RCX, RDX]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.F32, ValueType.F64]:
            regs = [XMM0, XMM1, XMM2]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.V4F32]:
            regs = [XMM0, XMM1, XMM2]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_return(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        target_info = ccstate.mfunc.target_info
        if target_info.triple.os == OS.Windows and target_info.is_64bit_mode:
            self.allocate_return_win64_cdecl(
                idx, vt, loc_vt, loc_info, flags, ccstate)
            return

        self.allocate_return_x64_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)

    def allocate_argument_x64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.I1, ValueType.I8, ValueType.I16]:
            loc_vt = MachineValueType(ValueType.I32)

        if loc_vt.value_type == ValueType.I32:
            regs = [EDI, ESI, EDX, ECX, R8D, R9D]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs = [RDI, RSI, RDX, RCX, R8, R9]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.F32, ValueType.F64, ValueType.F128]:
            regs = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.V4F32]:
            regs = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]
            reg = ccstate.alloc_reg_from_list(regs)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.I32, ValueType.I64]:
            stack_offset = ccstate.alloc_stack(8, 8)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_argument_win64_cdecl(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        if loc_vt.value_type in [ValueType.V4F32]:
            loc_vt = MachineValueType(ValueType.I64)
            loc_info = CCArgLocInfo.Indirect

        if loc_vt.value_type == ValueType.I8:
            regs1 = [CL, DL, R8B, R9B]
            regs2 = [XMM0, XMM1, XMM2, XMM3]
            reg = ccstate.alloc_reg_from_list(regs1, reg2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I16:
            regs1 = [CX, DX, R8W, R9W]
            regs2 = [XMM0, XMM1, XMM2, XMM3]
            reg = ccstate.alloc_reg_from_list(regs1, reg2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I32:
            regs1 = [ECX, EDX, R8D, R9D]
            regs2 = [XMM0, XMM1, XMM2, XMM3]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type == ValueType.I64:
            regs1 = [RCX, RDX, R8, R9]
            regs2 = [XMM0, XMM1, XMM2, XMM3]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.F32, ValueType.F64]:
            regs1 = [XMM0, XMM1, XMM2, XMM3]
            regs2 = [RCX, RDX, R8, R9]
            reg = ccstate.alloc_reg_from_list(regs1, regs2)
            if reg is not None:
                ccstate.assign_reg_value(idx, vt, loc_vt, loc_info, reg, flags)
                return False

        if loc_vt.value_type in [ValueType.I8, ValueType.I16, ValueType.I32, ValueType.I64]:
            stack_offset = ccstate.alloc_stack(8, 8)
            ccstate.assign_stack_value(
                idx, vt, loc_vt, loc_info, stack_offset, flags)
            return False

        raise NotImplementedError("The type is unsupporting.")

    def allocate_argument(self, idx, vt: MachineValueType, loc_vt, loc_info, flags: CCArgFlags, ccstate: CallingConvState):
        target_info = ccstate.mfunc.target_info
        if target_info.triple.os == OS.Windows and target_info.is_64bit_mode:
            self.allocate_argument_win64_cdecl(
                idx, vt, loc_vt, loc_info, flags, ccstate)
            return

        self.allocate_argument_x64_cdecl(
            idx, vt, loc_vt, loc_info, flags, ccstate)


class X64TargetInstInfo(TargetInstInfo):
    def __init__(self):
        super().__init__()

    def copy_phys_reg(self, src_reg, dst_reg, kill_src, inst: MachineInstruction):
        assert(isinstance(src_reg, MachineRegister))
        assert(isinstance(dst_reg, MachineRegister))

        if src_reg.spec in GR64.regs or dst_reg.spec in GR64.regs:
            opcode = X64MachineOps.MOV64rr
        elif src_reg.spec in GR32.regs or dst_reg.spec in GR32.regs:
            opcode = X64MachineOps.MOV32rr
        elif src_reg.spec in GR8.regs or dst_reg.spec in GR8.regs:
            opcode = X64MachineOps.MOV8rr
        elif src_reg.spec in VR128.regs and dst_reg.spec in VR128.regs:
            opcode = X64MachineOps.MOVAPSrr
        elif src_reg.spec in FR64.regs or dst_reg.spec in FR64.regs:
            opcode = X64MachineOps.MOVSDrr
        elif src_reg.spec in FR32.regs or dst_reg.spec in FR32.regs:
            opcode = X64MachineOps.MOVSSrr
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst = MachineInstruction(opcode)

        copy_inst.add_reg(dst_reg, RegState.Define)
        if opcode in [X64MachineOps.MOVSSrr, X64MachineOps.MOVSDrr]:
            copy_inst.add_reg(dst_reg, RegState.Non)
        copy_inst.add_reg(src_reg, RegState.Kill if kill_src else RegState.Non)

        copy_inst.insert_after(inst)

        return copy_inst

    def copy_reg_to_stack(self, reg, stack_slot, regclass, inst: MachineInstruction):
        hwmode = inst.mbb.func.target_info.hwmode

        tys = regclass.get_types(hwmode)

        align = int(regclass.align / 8)
        size = tys[0].get_size_in_bits()
        size = int(int((size + 7) / 8))

        def has_reg_regclass(reg, regclass):
            if isinstance(reg, MachineVirtualRegister):
                return reg.regclass == regclass
            else:
                return reg.spec in regclass.regs

        if size == 4:
            if has_reg_regclass(reg, GR32):
                opcode = X64MachineOps.MOV32mr
            elif has_reg_regclass(reg, FR32):
                opcode = X64MachineOps.MOVSSmr
        elif size == 1:
            if has_reg_regclass(reg, GR8):
                opcode = X64MachineOps.MOV8mr
        elif size == 8:
            if has_reg_regclass(reg, GR64):
                opcode = X64MachineOps.MOV64mr
            elif has_reg_regclass(reg, FR64):
                opcode = X64MachineOps.MOVSDmr
        elif size == 16:
            if has_reg_regclass(reg, VR128):
                opcode = X64MachineOps.MOVAPSmr
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst = MachineInstruction(opcode)

        noreg = MachineRegister(NOREG)

        copy_inst.add_frame_index(stack_slot)  # base
        copy_inst.add_imm(1)  # scale
        copy_inst.add_reg(noreg, RegState.Non)  # index
        copy_inst.add_imm(0)  # disp
        copy_inst.add_reg(noreg, RegState.Non)  # segment
        copy_inst.add_reg(reg, RegState.Non)

        copy_inst.insert_before(inst)

        return copy_inst

    def copy_reg_from_stack(self, reg, stack_slot, regclass, inst: MachineInstruction):
        hwmode = inst.mbb.func.target_info.hwmode

        tys = regclass.get_types(hwmode)

        align = int(regclass.align / 8)
        size = tys[0].get_size_in_bits()
        size = int(int((size + 7) / 8))

        def has_reg_regclass(reg, regclass):
            if isinstance(reg, MachineVirtualRegister):
                return reg.regclass == regclass
            else:
                return reg.spec in regclass.regs

        if size == 1:
            if has_reg_regclass(reg, GR8):
                opcode = X64MachineOps.MOV8rm
        elif size == 4:
            if has_reg_regclass(reg, GR32):
                opcode = X64MachineOps.MOV32rm
            elif has_reg_regclass(reg, FR32):
                opcode = X64MachineOps.MOVSSrm
        elif size == 8:
            if has_reg_regclass(reg, GR64):
                opcode = X64MachineOps.MOV64rm
            elif has_reg_regclass(reg, FR64):
                opcode = X64MachineOps.MOVSDrm
        elif size == 16:
            if has_reg_regclass(reg, VR128):
                opcode = X64MachineOps.MOVAPSrm
        else:
            raise NotImplementedError(
                "Move instructions support GR64 or GR32 at the present time.")

        copy_inst = MachineInstruction(opcode)

        noreg = MachineRegister(NOREG)

        copy_inst.add_reg(reg, RegState.Define)
        copy_inst.add_frame_index(stack_slot)  # base
        copy_inst.add_imm(1)  # scale
        copy_inst.add_reg(noreg, RegState.Non)  # index
        copy_inst.add_imm(0)  # disp
        copy_inst.add_reg(noreg, RegState.Non)  # segment

        copy_inst.insert_before(inst)

        return copy_inst

    def calculate_frame_offset(self, func: MachineFunction, idx):
        slot_size = 8
        frame = func.frame
        stack_obj = func.frame.get_stack_object(idx)
        frame_lowering = func.target_info.get_frame_lowering()
        if idx < 0:
            return stack_obj.offset + frame_lowering.frame_spill_size

        return stack_obj.offset

    def eliminate_frame_index(self, func: MachineFunction, inst: MachineInstruction, idx):
        # Analyze the frame index into a base register and a displacement.
        operand = inst.operands[idx]
        if isinstance(operand, MOFrameIndex):
            base_reg = MachineRegister(RBP)
            stack_obj = func.frame.get_stack_object(operand.index)
            offset = self.calculate_frame_offset(func, operand.index)

            inst.operands[idx] = MOReg(base_reg, RegState.Non)
            inst.operands[idx + 3] = MOImm(inst.operands[idx + 3].val + offset)

    def optimize_compare_inst(self, func: MachineFunction, inst: MachineInstruction):
        # Eliminate destination register.
        reginfo = func.reg_info
        if reginfo.is_use_empty(inst.operands[0].reg):

            if inst.opcode == X64MachineOps.SUB32ri:
                inst.opcode = X64MachineOps.CMP32ri
            elif inst.opcode == X64MachineOps.SUB32rm:
                inst.opcode = X64MachineOps.CMP32rm
            elif inst.opcode == X64MachineOps.SUB32rr:
                inst.opcode = X64MachineOps.CMP32rr
            elif inst.opcode == X64MachineOps.SUB8ri:
                inst.opcode = X64MachineOps.CMP8ri
            else:
                raise ValueError("Not supporting instruction.")

            remove_op = inst.operands[0]
            if remove_op.tied_to >= 0:
                tied = inst.operands[remove_op.tied_to]
                assert(tied.tied_to == 0)
                tied.tied_to = -1

            inst.remove_operand(0)

    def expand_post_ra_pseudo(self, inst: MachineInstruction):
        if inst.opcode == X64MachineOps.V_SET0:
            inst.opcode = X64MachineOps.XORPSrr
            reg_operand = inst.operands[0]
            inst.add_reg(reg_operand.reg, RegState.Undef)
            inst.add_reg(reg_operand.reg, RegState.Undef)


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


def get_sub_regs(reg):
    regs = MachineRegisterDef.regs

    stk = list(reg.subregs)
    subregs = set()
    while len(stk) > 0:
        poped = stk.pop()

        if poped in subregs:
            continue

        subregs.add(poped)

        for subreg in poped.subregs:
            stk.append(subreg)

    return subregs


def count_if(values, pred):
    return len([v for v in values if pred(v)])


def find_if(values, pred):
    for i, v in enumerate(values):
        if pred(v):
            return i

    return -1


class X64TargetLowering(TargetLowering):
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
                if cond in [CondCode.SETOLT, CondCode.SETOLE, CondCode.SETUGT, CondCode.SETUGE]:
                    swap = True
                    if cond == CondCode.SETOLT:
                        cond = CondCode.SETUGE
                    elif cond == CondCode.SETOLE:
                        cond = CondCode.SETUGT
                    elif cond == CondCode.SETUGT:
                        cond = CondCode.SETOLE
                    elif cond == CondCode.SETUGE:
                        cond = CondCode.SETOLT

            if cond == CondCode.SETEQ:
                node = dag.add_target_constant_node(ty, 4)
            elif cond == CondCode.SETNE:
                node = dag.add_target_constant_node(ty, 5)
            elif cond == CondCode.SETLT:
                node = dag.add_target_constant_node(ty, 12)
            elif cond == CondCode.SETGT:
                node = dag.add_target_constant_node(ty, 15)
            elif cond == CondCode.SETLE:
                node = dag.add_target_constant_node(ty, 14)
            elif cond == CondCode.SETGE:
                node = dag.add_target_constant_node(ty, 13)
            elif cond in [CondCode.SETULT, CondCode.SETOLT]:
                node = dag.add_target_constant_node(ty, 2)
            elif cond in [CondCode.SETUGT, CondCode.SETOGT]:
                node = dag.add_target_constant_node(ty, 7)
            elif cond in [CondCode.SETULE, CondCode.SETOLE]:
                node = dag.add_target_constant_node(ty, 6)
            elif cond in [CondCode.SETUGE, CondCode.SETOGE]:
                node = dag.add_target_constant_node(ty, 3)
            else:
                raise NotImplementedError()

            return node, swap

        condcode, swap = compute_condcode(cond)
        if swap:
            op1, op2 = op2, op1

        if is_fcmp:
            op = X64DagOps.CMP
            cmp_node = DagValue(dag.add_node(op,
                                             [MachineValueType(ValueType.I32)], op1, op2), 0)
        else:
            op = X64DagOps.SUB
            cmp_node = DagValue(dag.add_node(op,
                                             [op1.ty, MachineValueType(ValueType.I32)], op1, op2), 1)

        # operand 1 is eflags.
        setcc_node = dag.add_node(X64DagOps.SETCC, node.value_types,
                                  DagValue(condcode, 0), cmp_node)

        return setcc_node

    def lower_brcond(self, node: DagNode, dag: Dag):
        chain = node.operands[0]
        cond = node.operands[1]
        dest = node.operands[2]

        if cond.node.opcode == VirtualDagOps.SETCC:
            cond = DagValue(self.lower_setcc(cond.node, dag), 0)

        if cond.node.opcode == X64DagOps.SETCC:
            condcode = cond.node.operands[0]
            cond = cond.node.operands[1]
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
            cond = cond.node.operands[1]

        return dag.add_node(X64DagOps.BRCOND, node.value_types, chain, dest, condcode, cond)

    def lower_global_address(self, node: DagNode, dag: Dag):
        target_address = dag.add_global_address_node(
            node.value_types[0], node.value, True)
        wrapper_opc = X64DagOps.WRAPPER if node.value.is_thread_local else X64DagOps.WRAPPER_RIP
        return dag.add_node(wrapper_opc, node.value_types, DagValue(target_address, 0))

    def lower_global_tls_address(self, node: DagNode, dag: Dag):
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)
        global_value = node.value

        if dag.mfunc.target_info.machine.options.emulated_tls:
            raise NotImplementedError()

        if dag.mfunc.target_info.triple.os == OS.Linux:
            if global_value.thread_local == ThreadLocalMode.GeneralDynamicTLSModel:
                ga = DagValue(dag.add_global_address_node(
                    ptr_ty, global_value, True), 0)

                ops = [dag.entry, ga]

                chain = DagValue(dag.add_node(X64DagOps.TLSADDR, [
                                 MachineValueType(ValueType.OTHER)], *ops), 0)

                reg_node = DagValue(dag.add_register_node(
                    ptr_ty, MachineRegister(RAX)), 0)

                return dag.add_node(VirtualDagOps.COPY_FROM_REG, [ptr_ty, MachineValueType(
                    ValueType.OTHER)], chain, reg_node)

            raise ValueError("Not supporing TLS model.")

        if dag.mfunc.target_info.triple.os == OS.Windows:
            ptr = get_constant_null_value(PointerType(i8, 256))
            tls_array = DagValue(dag.add_constant_node(ptr_ty, 0x58), 0)

            tls_array = DagValue(dag.add_node(
                X64DagOps.WRAPPER_RIP, node.value_types, tls_array), 0)

            thread_ptr = DagValue(dag.add_load_node(
                ptr_ty, dag.entry, tls_array, False, ptr_info=MachinePointerInfo(ptr)), 0)

            if global_value.thread_local == ThreadLocalMode.LocalExecTLSModel:
                raise NotImplementedError()
            else:
                idx = DagValue(dag.add_external_symbol_node(
                    ptr_ty, "_tls_index", False), 0)

                idx = DagValue(dag.add_node(
                    X64DagOps.WRAPPER_RIP, node.value_types, idx), 0)

                idx = DagValue(dag.add_load_node(
                    ptr_ty, dag.entry, idx, False), 0)

                def log2_uint64_cail(value):
                    if value == 0:
                        return 0

                    value = value - 1
                    for i in reversed(range(63)):
                        if (value & (1 << 64)) != 0:
                            return i

                        value = value << 1

                    return 0

                scale = DagValue(dag.add_constant_node(
                    MachineValueType(ValueType.I8), log2_uint64_cail(data_layout.get_pointer_size_in_bits())), 0)

                idx = DagValue(dag.add_node(
                    VirtualDagOps.SHL, [ptr_ty], idx, scale), 0)

                thread_ptr = DagValue(dag.add_node(
                    VirtualDagOps.ADD, [ptr_ty], thread_ptr, idx), 0)

            tls_ptr = DagValue(dag.add_load_node(
                ptr_ty, dag.entry, thread_ptr, False), 0)

            # This value is the offset from the .tls section
            target_address = DagValue(dag.add_global_address_node(
                node.value_types[0], node.value, True, target_flags=X64OperandFlag.SECREL), 0)

            offset = DagValue(dag.add_node(
                X64DagOps.WRAPPER, node.value_types, target_address), 0)

            return dag.add_node(VirtualDagOps.ADD, [ptr_ty], tls_ptr, offset)

        raise NotImplementedError()

    def get_pointer_type(self, data_layout, addr_space=0):
        return get_int_value_type(data_layout.get_pointer_size_in_bits(addr_space))

    def get_frame_index_type(self, data_layout):
        return get_int_value_type(data_layout.get_pointer_size_in_bits(0))

    def lower_constant_fp(self, node: DagNode, dag: Dag):
        assert(isinstance(node, ConstantFPDagNode))
        data_layout = dag.mfunc.func_info.func.module.data_layout
        ptr_ty = self.get_pointer_type(data_layout)

        constant_pool = dag.add_constant_pool_node(ptr_ty, node.value, False)
        return dag.add_load_node(node.value_types[0], dag.entry, DagValue(constant_pool, 0), False)

    def lower_constant_pool(self, node: DagNode, dag: Dag):
        assert(isinstance(node, ConstantPoolDagNode))
        target_constant_pool = dag.add_constant_pool_node(
            node.value_types[0], node.value, True)
        return dag.add_node(X64DagOps.WRAPPER_RIP, node.value_types, DagValue(target_constant_pool, 0))

    def lower_build_vector(self, node: DagNode, dag: Dag):
        assert(node.opcode == VirtualDagOps.BUILD_VECTOR)
        operands = []
        for operand in node.operands:
            target_constant_fp = dag.add_target_constant_fp_node(
                operand.node.value_types[0], operand.node.value)
            operands.append(DagValue(target_constant_fp, 0))

        return dag.add_node(VirtualDagOps.BUILD_VECTOR, node.value_types, *operands)

    def shuffle_param(self, fp3, fp2, fp1, fp0):
        return (fp3 << 6) | (fp2 << 4) | (fp1 << 2) | fp0

    def get_x86_shuffle_mask_v4(self, mask, dag):
        mask_val = self.shuffle_param(mask[3], mask[2], mask[1], mask[0])
        return DagValue(dag.add_target_constant_node(MachineValueType(ValueType.I8), mask_val), 0)

    def _mm_set_ps1(self, val):
        vec = DagValue(dag.add_node(
            VirtualDagOps.SCALAR_TO_VECTOR, [vec.ty], val), 0)
        param = self.create_i8_constant(self.shuffle_param(0, 0, 0, 0))

        return DagValue(dag.add_node(X64DagOps.SHUFP, node.value_types, vec, vec, param), 0)

    def lower_insert_vector_elt(self, node: DagNode, dag: Dag):
        assert(node.opcode == VirtualDagOps.INSERT_VECTOR_ELT)
        vec = node.operands[0]
        elem = node.operands[1]
        idx = node.operands[2]

        if isinstance(idx.node, ConstantDagNode):
            elem_vec = DagValue(dag.add_node(
                VirtualDagOps.SCALAR_TO_VECTOR, [vec.ty], elem), 0)
            num_elems = vec.ty.get_num_vector_elems()
            idx_val = idx.node.value

            shuffle_idx = []
            for i in range(num_elems):
                if i == idx_val.value:
                    shuffle_idx.append(num_elems)
                else:
                    shuffle_idx.append(i)

            return dag.add_shuffle_vector(vec.ty, vec, elem_vec, shuffle_idx)

        raise ValueError()

    def get_scalar_value_for_vec_elem(self, vec, idx, dag: Dag):
        if vec.node.opcode == VirtualDagOps.SCALAR_TO_VECTOR and idx == 0:
            scalar_val = vec.node.operands[idx]
            return scalar_val

        raise ValueError()

    def lower_shuffle_as_elem_insertion(self, vt, vec1, vec2, mask, dag: Dag):
        vec2_idx = find_if(mask, lambda m: m >= len(mask))

        elem_vt = vt.get_vector_elem_type()
        assert(elem_vt.value_type == ValueType.F32)

        vec2_elem = self.get_scalar_value_for_vec_elem(vec2, vec2_idx, dag)

        vec2 = DagValue(dag.add_node(
            VirtualDagOps.SCALAR_TO_VECTOR, [vt], vec2_elem), 0)

        if elem_vt.value_type == ValueType.F32:
            opcode = X64DagOps.MOVSS
        else:
            opcode = X64DagOps.MOVSD

        return dag.add_node(opcode, [vt], vec1, vec2)

    def lower_shuffle_shufps(self, vt, vec1, vec2, mask, dag: Dag):
        assert(len(mask) == 4)
        num_vec2_elems = count_if(mask, lambda m: m >= 4)

        new_mask = list(mask)
        lo_vec, hi_vec = vec1, vec2

        if num_vec2_elems == 1:
            vec2_idx = find_if(mask, lambda m: m >= 4)

            # Each element of the vector is divided into groups of two elements.
            # If the index is odd, the index of the other element is even.
            vec2_idx_adj = vec2_idx ^ 1

            # Merge the vectors.
            blend_mask = [mask[vec2_idx] - 4, 0, mask[vec2_idx_adj], 0]
            vec2 = DagValue(dag.add_node(X64DagOps.SHUFP, [
                vt], vec2, vec1, self.get_x86_shuffle_mask_v4(blend_mask, dag)), 0)

            if vec2_idx < 2:
                lo_vec = vec2
                hi_vec = vec1
            else:
                lo_vec = vec1
                hi_vec = vec2

            new_mask[vec2_idx] = 0
            new_mask[vec2_idx_adj] = 2
        elif num_vec2_elems == 2:
            raise NotImplementedError()

        return dag.add_node(X64DagOps.SHUFP, [vt], lo_vec, hi_vec, self.get_x86_shuffle_mask_v4(new_mask, dag))

    def lower_v4f32_shuffle(self, node: DagNode, dag: Dag):
        vec1 = node.operands[0]
        vec2 = node.operands[1]
        mask = node.mask
        num_vec2_elems = count_if(mask, lambda m: m >= 4)

        if num_vec2_elems == 1 and mask[0] >= 4:
            return self.lower_shuffle_as_elem_insertion(MachineValueType(ValueType.V4F32), vec1, vec2, mask, dag)

        return self.lower_shuffle_shufps(MachineValueType(ValueType.V4F32), vec1, vec2, mask, dag)

    def lower_shuffle_vector(self, node: DagNode, dag: Dag):
        if node.value_types[0] == MachineValueType(ValueType.V4F32):
            return self.lower_v4f32_shuffle(node, dag)

        raise ValueError()

    def lower_sub(self, node: DagNode, dag: Dag):
        return dag.add_node(X64DagOps.SUB, node.value_types, *node.operands)

    def lower_atomic_fence(self, node: DagNode, dag: Dag):
        ordering = node.operands[1].node.value.value
        if ordering == AtomicOrdering.SequentiallyConsistent.value:
            raise NotImplementedError()

        return dag.add_node(X64DagOps.MEMBARRIER, node.value_types, node.operands[0])

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
        elif node.opcode == VirtualDagOps.GLOBAL_TLS_ADDRESS:
            return self.lower_global_tls_address(node, dag)
        elif node.opcode == VirtualDagOps.CONSTANT_FP:
            return self.lower_constant_fp(node, dag)
        elif node.opcode == VirtualDagOps.CONSTANT_POOL:
            return self.lower_constant_pool(node, dag)
        elif node.opcode == VirtualDagOps.BUILD_VECTOR:
            return self.lower_build_vector(node, dag)
        elif node.opcode == VirtualDagOps.SHUFFLE_VECTOR:
            return self.lower_shuffle_vector(node, dag)
        elif node.opcode == VirtualDagOps.INSERT_VECTOR_ELT:
            return self.lower_insert_vector_elt(node, dag)
        elif node.opcode == VirtualDagOps.ATOMIC_FENCE:
            return self.lower_atomic_fence(node, dag)
        else:
            return node

    def lower_arguments(self, func: Function, builder: DagBuilder):
        arg_load_chains = []
        chain = builder.root

        mfunc = builder.mfunc
        calling_conv = mfunc.target_info.get_calling_conv()
        reg_info = mfunc.target_info.get_register_info()
        data_layout = func.module.data_layout

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
            if isinstance(arg_val, CCArgReg):
                if arg_vt.value_type == ValueType.I8:
                    regclass = GR8
                elif arg_vt.value_type == ValueType.I16:
                    regclass = GR16
                elif arg_vt.value_type == ValueType.I32:
                    regclass = GR32
                elif arg_vt.value_type == ValueType.I64:
                    regclass = GR64
                elif arg_vt.value_type == ValueType.F32:
                    regclass = FR32
                elif arg_vt.value_type == ValueType.F64:
                    regclass = FR64
                elif arg_vt.value_type == ValueType.V4F32:
                    regclass = VR128
                else:
                    raise ValueError()

                reg = mfunc.reg_info.create_virtual_register(regclass)

                mfunc.reg_info.add_live_in(MachineRegister(arg_val.reg), reg)

                reg_node = DagValue(
                    builder.g.add_register_node(arg_vt, reg), 0)

                arg_val_node = DagValue(
                    builder.g.add_copy_from_reg_node(arg_vt, reg_node), 0)
            else:
                assert(isinstance(arg_val, CCArgMem))

                size = arg_vt.get_size_in_byte()
                offset = arg_val.offset

                frame_idx = builder.mfunc.frame.create_fixed_stack_object(
                    size, offset)

                frame_idx_node = DagValue(
                    builder.g.add_frame_index_node(frame_idx), 0)

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
                    mfunc.func_info.reg_value_map[arg] = [reg]

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
        if inst.opcode == X64MachineOps.ADJCALLSTACKDOWN32:
            return True
        if inst.opcode == X64MachineOps.ADJCALLSTACKUP32:
            return True

        return False

    def lower_prolog(self, func: MachineFunction, bb: MachineBasicBlock):
        inst_info = func.target_info.get_inst_info()
        frame_info = func.target_info.get_frame_lowering()
        reg_info = func.target_info.get_register_info()
        data_layout = func.func_info.func.module.data_layout

        front_inst = bb.insts[0]

        push_rbp_inst = MachineInstruction(X64MachineOps.PUSH64r)
        push_rbp_inst.add_reg(MachineRegister(RBP), RegState.Non)
        push_rbp_inst.add_reg(MachineRegister(RSP), RegState.ImplicitDefine)

        push_rbp_inst.insert_before(front_inst)

        mov_esp_inst = MachineInstruction(X64MachineOps.MOV64rr)
        mov_esp_inst.add_reg(MachineRegister(RBP), RegState.Define)  # To
        mov_esp_inst.add_reg(MachineRegister(RSP), RegState.Non)  # From

        mov_esp_inst.insert_before(front_inst)

        # The stack and base pointer is aligned by 16 bytes here.

        for cs_info in func.frame.calee_save_info:
            reg = cs_info.reg
            regclass = reg_info.get_regclass_from_reg(reg)
            frame_idx = cs_info.frame_idx

            inst_info.copy_reg_to_stack(MachineRegister(
                reg), frame_idx, regclass, front_inst)

        stack_size = func.frame.estimate_stack_size(
            X64MachineOps.ADJCALLSTACKDOWN32, X64MachineOps.ADJCALLSTACKUP32)

        max_align = max(func.frame.max_alignment, func.frame.stack_alignment)
        stack_size = int(
            int((stack_size + max_align - 1) / max_align) * max_align)

        sub_esp_inst = MachineInstruction(X64MachineOps.SUB64ri)
        sub_esp_inst.add_reg(MachineRegister(RSP), RegState.Define)
        sub_esp_inst.add_reg(MachineRegister(RSP), RegState.Non)
        sub_esp_inst.add_imm(stack_size)

        sub_esp_inst.insert_before(front_inst)

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

        restore_esp_inst = MachineInstruction(X64MachineOps.MOV64rr)
        restore_esp_inst.add_reg(MachineRegister(RSP), RegState.Define)  # To
        restore_esp_inst.add_reg(MachineRegister(RBP), RegState.Non)  # From

        restore_esp_inst.insert_before(front_inst)

        pop_rbp_inst = MachineInstruction(X64MachineOps.POP64r)
        pop_rbp_inst.add_reg(MachineRegister(RBP), RegState.Non)
        pop_rbp_inst.add_reg(MachineRegister(RSP), RegState.ImplicitDefine)

        pop_rbp_inst.insert_before(front_inst)

    def eliminate_call_frame_pseudo_inst(self, func, inst: MachineInstruction):
        inst.remove()

    def get_machine_vreg(self, ty: MachineValueType):
        if ty.value_type == ValueType.I8:
            return GR8
        elif ty.value_type == ValueType.I16:
            return GR16
        elif ty.value_type == ValueType.I32:
            return GR32
        elif ty.value_type == ValueType.I64:
            return GR64
        elif ty.value_type == ValueType.F32:
            return FR32
        elif ty.value_type == ValueType.F64:
            return FR64
        elif ty.value_type == ValueType.V4F32:
            return VR128

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


class X64TargetRegisterInfo(TargetRegisterInfo):
    def __init__(self):
        super().__init__()

    def get_reserved_regs(self):
        reserved = []
        reserved.extend([SPL, BPL])
        reserved.extend([ESP, EBP])
        reserved.extend([RSP, RBP])

        return reserved

    @property
    def allocatable_regs(self):
        regs = set()
        regs |= set(GR64.regs)
        regs |= set(GR32.regs)
        regs |= set(GR16.regs)
        regs |= set(GR8.regs)
        regs |= set(FR32.regs)
        regs |= set(FR64.regs)
        regs |= set(VR128.regs)

        return regs

    def get_callee_saved_regs(self):
        callee_save_regs = [RBX, R12, R13, R14, R15, RDI, RSI, XMM6,
                            XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15]

        return callee_save_regs

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


class X64FrameLowering(TargetFrameLowering):
    def __init__(self, alignment):
        super().__init__(alignment)

        self.frame_spill_size = 0

    @property
    def stack_grows_direction(self):
        return StackGrowsDirection.Down


class X64Legalizer(Legalizer):
    def __init__(self):
        super().__init__()

    def promote_integer_result_setcc(self, node, dag, legalized):
        setcc_ty = MachineValueType(ValueType.I8)

        return dag.add_node(node.opcode, [setcc_ty], *node.operands)

    def get_legalized_op(self, operand, legalized):
        if operand.node in legalized:
            return DagValue(legalized[operand.node], operand.index)

        return operand

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


class X64TargetInfo(TargetInfo):
    def __init__(self, triple, machine):
        super().__init__(triple)
        self.machine = machine

    def get_inst_info(self) -> TargetInstInfo:
        return X64TargetInstInfo()

    def is_64bit_mode(self):
        return self.triple.arch == ArchType.X86_64

    def get_lowering(self) -> TargetLowering:
        return X64TargetLowering()

    def get_register_info(self) -> TargetRegisterInfo:
        return X64TargetRegisterInfo()

    def get_calling_conv(self) -> CallingConv:
        return X86CallingConv()

    def get_instruction_selector(self):
        return X64InstructionSelector()

    def get_legalizer(self):
        return X64Legalizer()

    def get_frame_lowering(self) -> TargetFrameLowering:
        return X64FrameLowering(16)

    @property
    def hwmode(self) -> MachineHWMode:
        if self.triple.arch == ArchType.X86_64:
            return X64

        raise ValueError("Invalid arch type")


class X64TargetMachine(TargetMachine):
    def __init__(self, triple, options):
        super().__init__(options)
        self.triple = triple

    def get_target_info(self, func: Function):
        return X64TargetInfo(self.triple, self)

    def add_mc_emit_passes(self, pass_manager, mccontext, output, is_asm):
        from codegen.x64_asm_printer import X64AsmInfo, MCAsmStream, X64CodeEmitter, X64AsmBackend, X64AsmPrinter, X64IntelInstPrinter
        from codegen.coff import WinCOFFObjectWriter, WinCOFFObjectStream
        from codegen.elf import ELFObjectStream, ELFObjectWriter, X64ELFObjectWriter

        objformat = self.triple.objformat

        mccontext.asm_info = X64AsmInfo()
        if is_asm:
            printer = X64IntelInstPrinter()
            stream = MCAsmStream(mccontext, output, printer)
        else:
            emitter = X64CodeEmitter()
            backend = X64AsmBackend()

            if objformat == ObjectFormatType.COFF:
                writer = WinCOFFObjectWriter(output)
                stream = WinCOFFObjectStream(
                    mccontext, backend, writer, emitter)
            elif objformat == ObjectFormatType.ELF:
                target_writer = X64ELFObjectWriter()
                writer = ELFObjectWriter(output, target_writer)
                stream = ELFObjectStream(mccontext, backend, writer, emitter)

        pass_manager.passes.append(X64AsmPrinter(stream))
