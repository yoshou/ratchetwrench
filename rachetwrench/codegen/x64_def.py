from rachetwrench.codegen.spec import *
from rachetwrench.codegen.mir_emitter import *
from rachetwrench.codegen.matcher import *

sub_8bit = def_subreg("sub_8bit", 8)
sub_8bit_hi = def_subreg("sub_8bit_hi", 8)
sub_16bit = def_subreg("sub_16bit", 16)
sub_32bit = def_subreg("sub_32bit", 32)
sub_xmm = def_subreg("sub_xmm", 128)
sub_ymm = def_subreg("sub_ymm", 256)


# 8-bit registers
AL = def_reg("al", encoding=0)
DL = def_reg("dl", encoding=2)
CL = def_reg("cl", encoding=1)
BL = def_reg("bl", encoding=3)

AH = def_reg("ah", encoding=4)
DH = def_reg("dh", encoding=6)
CH = def_reg("ch", encoding=5)
BH = def_reg("bh", encoding=7)

# X86-64 only
SIL = def_reg("sil", encoding=6)
DIL = def_reg("dil", encoding=7)
BPL = def_reg("bpl", encoding=5)
SPL = def_reg("spl", encoding=4)

R8B = def_reg("r8b", encoding=8)
R9B = def_reg("r9b", encoding=9)
R10B = def_reg("r10b", encoding=10)
R11B = def_reg("r11b", encoding=11)
R12B = def_reg("r12b", encoding=12)
R13B = def_reg("r13b", encoding=13)
R14B = def_reg("r14b", encoding=14)
R15B = def_reg("r15b", encoding=15)

# 16-bit registers

AX = def_reg("ax", [AL, AH], [sub_8bit, sub_8bit_hi], encoding=0)
DX = def_reg("dx", [DL, DH], [sub_8bit, sub_8bit_hi], encoding=2)
CX = def_reg("cx", [CL, CH], [sub_8bit, sub_8bit_hi], encoding=1)
BX = def_reg("bx", [BL, BH], [sub_8bit, sub_8bit_hi], encoding=3)
SI = def_reg("si", [SIL], [sub_8bit], encoding=6)
DI = def_reg("di", [DIL], [sub_8bit], encoding=7)
BP = def_reg("bp", [BPL], [sub_8bit], encoding=5)
SP = def_reg("sp", [SPL], [sub_8bit], encoding=4)
IP = def_reg("ip")


R8W = def_reg("r8w", [R8B], [sub_8bit], encoding=8)
R9W = def_reg("r9w", [R9B], [sub_8bit], encoding=9)
R10W = def_reg("r10w", [R10B], [sub_8bit], encoding=10)
R11W = def_reg("r11w", [R11B], [sub_8bit], encoding=11)
R12W = def_reg("r12w", [R12B], [sub_8bit], encoding=12)
R13W = def_reg("r13w", [R13B], [sub_8bit], encoding=13)
R14W = def_reg("r14w", [R14B], [sub_8bit], encoding=14)
R15W = def_reg("r15w", [R15B], [sub_8bit], encoding=15)

# 32-bit registers
EAX = def_reg("eax", [AX], [sub_16bit], encoding=0)
EDX = def_reg("edx", [DX], [sub_16bit], encoding=2)
ECX = def_reg("ecx", [CX], [sub_16bit], encoding=1)
EBX = def_reg("ebx", [BX], [sub_16bit], encoding=3)
ESI = def_reg("esi", [SI], [sub_16bit], encoding=6)
EDI = def_reg("edi", [DI], [sub_16bit], encoding=7)
EBP = def_reg("ebp", [BP], [sub_16bit], encoding=5)
ESP = def_reg("esp", [SP], [sub_16bit], encoding=4)
EIP = def_reg("eip", [IP], [sub_16bit])

R8D = def_reg("r8d", [R8W], [sub_16bit], encoding=8)
R9D = def_reg("r9d", [R9W], [sub_16bit], encoding=9)
R10D = def_reg("r10d", [R10W], [sub_16bit], encoding=10)
R11D = def_reg("r11d", [R11W], [sub_16bit], encoding=11)
R12D = def_reg("r12d", [R12W], [sub_16bit], encoding=12)
R13D = def_reg("r13d", [R13W], [sub_16bit], encoding=13)
R14D = def_reg("r14d", [R14W], [sub_16bit], encoding=14)
R15D = def_reg("r15d", [R15W], [sub_16bit], encoding=15)

# 64-bit registers
RAX = def_reg("rax", [EAX], [sub_32bit], encoding=0)
RDX = def_reg("rdx", [EDX], [sub_32bit], encoding=2)
RCX = def_reg("rcx", [ECX], [sub_32bit], encoding=1)
RBX = def_reg("rbx", [EBX], [sub_32bit], encoding=3)
RSI = def_reg("rsi", [ESI], [sub_32bit], encoding=6)
RDI = def_reg("rdi", [EDI], [sub_32bit], encoding=7)
RBP = def_reg("rbp", [EBP], [sub_32bit], encoding=5)
RSP = def_reg("rsp", [ESP], [sub_32bit], encoding=4)

R8 = def_reg("r8", [R8D], [sub_32bit], encoding=8)
R9 = def_reg("r9", [R9D], [sub_32bit], encoding=9)
R10 = def_reg("r10", [R10D], [sub_32bit], encoding=10)
R11 = def_reg("r11", [R11D], [sub_32bit], encoding=11)
R12 = def_reg("r12", [R12D], [sub_32bit], encoding=12)
R13 = def_reg("r13", [R13D], [sub_32bit], encoding=13)
R14 = def_reg("r14", [R14D], [sub_32bit], encoding=14)
R15 = def_reg("r15", [R15D], [sub_32bit], encoding=15)

RIP = def_reg("rip")

EFLAGS = def_reg("eflags")

# Segment registers
CS = def_reg("cs", encoding=1)
DS = def_reg("ds", encoding=3)
SS = def_reg("ss", encoding=2)
ES = def_reg("es", encoding=0)
FS = def_reg("fs", encoding=4)
GS = def_reg("gs", encoding=5)

SEGMENT_REG = def_regclass("SEGMENT_REG", [ValueType.I16], 2, [
                           CS, DS, SS, ES, FS, GS])

MM0 = def_reg("mm0", encoding=0)
MM1 = def_reg("mm1", encoding=1)
MM2 = def_reg("mm2", encoding=2)
MM3 = def_reg("mm3", encoding=3)
MM4 = def_reg("mm4", encoding=4)
MM5 = def_reg("mm5", encoding=5)
MM6 = def_reg("mm6", encoding=6)
MM7 = def_reg("mm7", encoding=7)

# Pseudo Floating Point registers
FP0 = def_reg("fp0", encoding=0)
FP1 = def_reg("fp1", encoding=1)
FP2 = def_reg("fp2", encoding=2)
FP3 = def_reg("fp3", encoding=3)
FP4 = def_reg("fp4", encoding=4)
FP5 = def_reg("fp5", encoding=5)
FP6 = def_reg("fp6", encoding=6)
FP7 = def_reg("fp7", encoding=7)

XMM0 = def_reg("xmm0", encoding=0)
XMM1 = def_reg("xmm1", encoding=1)
XMM2 = def_reg("xmm2", encoding=2)
XMM3 = def_reg("xmm3", encoding=3)
XMM4 = def_reg("xmm4", encoding=4)
XMM5 = def_reg("xmm5", encoding=5)
XMM6 = def_reg("xmm6", encoding=6)
XMM7 = def_reg("xmm7", encoding=7)

# X86-64 only
XMM8 = def_reg("xmm8", encoding=8)
XMM9 = def_reg("xmm9", encoding=9)
XMM10 = def_reg("xmm10", encoding=10)
XMM11 = def_reg("xmm11", encoding=11)
XMM12 = def_reg("xmm12", encoding=12)
XMM13 = def_reg("xmm13", encoding=13)
XMM14 = def_reg("xmm14", encoding=14)
XMM15 = def_reg("xmm15", encoding=15)

XMM16 = def_reg("xmm16", encoding=16)
XMM17 = def_reg("xmm17", encoding=17)
XMM18 = def_reg("xmm18", encoding=18)
XMM19 = def_reg("xmm19", encoding=19)
XMM20 = def_reg("xmm20", encoding=20)
XMM21 = def_reg("xmm21", encoding=21)
XMM22 = def_reg("xmm22", encoding=22)
XMM23 = def_reg("xmm23", encoding=23)
XMM24 = def_reg("xmm24", encoding=24)
XMM25 = def_reg("xmm25", encoding=25)
XMM26 = def_reg("xmm26", encoding=26)
XMM27 = def_reg("xmm27", encoding=27)
XMM28 = def_reg("xmm28", encoding=28)
XMM29 = def_reg("xmm29", encoding=29)
XMM30 = def_reg("xmm30", encoding=30)
XMM31 = def_reg("xmm31", encoding=31)

for idx in range(32):
    globals()[f'YMM{idx}'] = def_reg(f"ymm{idx}", [
        globals()[f'XMM{idx}']], [sub_xmm], encoding=idx)

for idx in range(32):
    globals()[f'ZMM{idx}'] = def_reg(f"zmm{idx}", [
        globals()[f'YMM{idx}']], [sub_ymm], encoding=idx)


x64_regclasses = []


def def_x64_regclass(*args, **kwargs):
    regclass = def_regclass(*args, **kwargs)
    x64_regclasses.append(regclass)
    return regclass


# GR8 = def_x64_regclass("GR8", [ValueType.I8], 8, [
#     AL, CL, DL, AH, CH, DH, BL, BH, SIL, DIL, BPL, SPL, R8B, R9B, R10B, R11B, R14B, R15B, R12B, R13B])
GR8 = def_x64_regclass("GR8", [ValueType.I8], 8, [
    AL, CL, DL, BL, SIL, DIL, BPL, SPL, R8B, R9B, R10B, R11B, R14B, R15B, R12B, R13B])

GR16 = def_x64_regclass("GR16", [ValueType.I16], 32, [
    AX, CX, DX, SI, DI, BX, BP, SP, R8W, R9W, R10W, R11W, R14W, R15W, R12W, R13W])

GR32 = def_x64_regclass("GR32", [ValueType.I32], 32, [
    EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP, R8D, R9D, R10D, R11D, R14D, R15D, R12D, R13D])

GR64 = def_x64_regclass("GR64", [ValueType.I64], 64, [
    RAX, RCX, RDX, RSI, RDI, RBX, RBP, RSP, R8, R9, R10, R11, R14, R15, R12, R13])

VR128 = def_x64_regclass("VR128", [ValueType.V4F32], 128, [
    globals()[f"XMM{idx}"] for idx in range(16)])

FR32 = def_x64_regclass("FR32", [ValueType.F32], 32, [
    globals()[f"XMM{idx}"] for idx in range(16)])

FR64 = def_x64_regclass("FR64", [ValueType.F64], 64, [
    globals()[f"XMM{idx}"] for idx in range(16)])

CCR = def_x64_regclass("CCR", [ValueType.I32], 32, [EFLAGS])

X64 = MachineHWMode("x64")

# build register graph
reg_graph = compute_reg_graph()
reg_groups = compute_reg_groups(reg_graph)
compute_reg_subregs_all(reg_graph)
compute_reg_superregs_all(reg_graph)

# infer regclass
for regclass in x64_regclasses:
    infer_subregclass_and_subreg(regclass)

I8Imm = ValueOperandDef(ValueType.I8)
I16Imm = ValueOperandDef(ValueType.I16)
I32Imm = ValueOperandDef(ValueType.I32)
I64Imm = ValueOperandDef(ValueType.I64)

F32Imm = ValueOperandDef(ValueType.F32)
F64Imm = ValueOperandDef(ValueType.F64)

BrTarget8 = ValueOperandDef(ValueType.I8)
BrTarget16 = ValueOperandDef(ValueType.I16)
BrTarget32 = ValueOperandDef(ValueType.I32)


class X64MemOperandDef(ValueOperandDef):
    def __init__(self):
        super().__init__(ValueType.IPTR)

    @property
    def operand_info(self):
        return [GR32, I8Imm, GR32, I32Imm, SEGMENT_REG]


I8Mem = X64MemOperandDef()
I16Mem = X64MemOperandDef()
I32Mem = X64MemOperandDef()
I64Mem = X64MemOperandDef()
F32Mem = X64MemOperandDef()
F64Mem = X64MemOperandDef()
F128Mem = X64MemOperandDef()
AnyMem = X64MemOperandDef()


def match_reloc_imm8(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    value = values[idx]
    if value.node.opcode != VirtualDagOps.CONSTANT:
        return idx, None

    if value.ty.value_type != ValueType.I8:
        return idx, None

    constant = value.node.value.value
    if constant & 0xFF != constant:
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


def match_reloc_imm16(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    value = values[idx]
    if value.node.opcode != VirtualDagOps.CONSTANT:
        return idx, None

    if value.ty.value_type != ValueType.I16:
        return idx, None

    constant = value.node.value.value
    if constant & 0xFFFF != constant:
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


def match_reloc_imm32(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    if idx >= len(values):
        return idx, None

    value = values[idx]
    if value.node.opcode != VirtualDagOps.CONSTANT:
        return idx, None

    if value.ty.value_type != ValueType.I32:
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


def match_reloc_imm64(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    if idx >= len(values):
        return idx, None

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    if value.ty.value_type != ValueType.I64:
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


def match_reloc_immf32(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps
    from rachetwrench.codegen.types import ValueType

    if idx >= len(values):
        return idx, None

    value = values[idx]
    if value.node.opcode != VirtualDagOps.CONSTANT:
        return idx, None

    if value.ty.value_type != ValueType.F32:
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


reloc_imm8 = ComplexOperandMatcher(match_reloc_imm8)
reloc_imm16 = ComplexOperandMatcher(match_reloc_imm16)
reloc_imm32 = ComplexOperandMatcher(match_reloc_imm32)
reloc_imm64 = ComplexOperandMatcher(match_reloc_imm64)
reloc_immf32 = ComplexOperandMatcher(match_reloc_immf32)


def match_reloc_imm(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    if idx >= len(values):
        return idx, None

    value = values[idx]
    if value.node.opcode != VirtualDagOps.CONSTANT:
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


reloc_imm = ComplexOperandMatcher(match_reloc_imm)


def match_addr(node, operands, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.x64_gen import X64DagOps
    from rachetwrench.codegen.types import MachineValueType, ValueType
    from rachetwrench.codegen.mir import MachineRegister
    from rachetwrench.codegen.spec import NOREG
    from rachetwrench.codegen.x64_def import RIP

    if idx >= len(operands):
        return idx, None

    operand = operands[idx]

    noreg = MachineRegister(NOREG)
    MVT = MachineValueType

    if operand.node.opcode == VirtualDagOps.ADD:
        sub_op1 = operand.node.operands[0]
        sub_op2 = operand.node.operands[1]

        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        def is_constant(value):
            return value.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]

        def is_index(value):
            if value.node.opcode == VirtualDagOps.MUL:
                mul_op1 = sub_op2.node.operands[0]
                mul_op2 = sub_op2.node.operands[1]

                if mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    return True

                if mul_op2.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op2.node.value.value in [1, 2, 4, 8]:
                    return True

            return False

        if sub_op1.node.opcode in [VirtualDagOps.FRAME_INDEX] and (is_constant(sub_op2) or is_index(sub_op2)):
            base = DagValue(dag.add_frame_index_node(
                sub_op1.ty, sub_op1.node.index, True), 0)

            if sub_op2.node.opcode == VirtualDagOps.MUL:
                mul_op1 = sub_op2.node.operands[0]
                mul_op2 = sub_op2.node.operands[1]

                if mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op1
                    index = mul_op2
                elif mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op2
                    index = mul_op1
                else:
                    index = sub_op2

                disp = DagValue(dag.add_target_constant_node(
                    MVT(ValueType.I32), 0), 0)
            else:
                disp = sub_op2
        elif sub_op2.node.opcode in [VirtualDagOps.FRAME_INDEX] and (is_constant(sub_op1) or is_index(sub_op1)):
            base = DagValue(dag.add_frame_index_node(
                sub_op2.ty, sub_op2.node.index, True), 0)

            if sub_op2.node.opcode == VirtualDagOps.MUL:
                mul_op1 = sub_op2.node.operands[0]
                mul_op2 = sub_op2.node.operands[1]

                if mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op1
                    index = mul_op2
                elif mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op2
                    index = mul_op1
                else:
                    index = sub_op2

                disp = DagValue(dag.add_target_constant_node(
                    MVT(ValueType.I32), 0), 0)
            else:
                disp = sub_op1
        elif sub_op2.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            if sub_op1.node.opcode == X64DagOps.WRAPPER_RIP and sub_op2.node.is_zero:
                base = DagValue(dag.add_target_register_node(
                    MVT(ValueType.I64), RIP), 0)

                disp = sub_op1.node.operands[0]
            else:
                base = sub_op1
                disp = sub_op2
        elif sub_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            if sub_op2.node.opcode == X64DagOps.WRAPPER_RIP and sub_op1.node.is_zero:
                base = DagValue(dag.add_target_register_node(
                    MVT(ValueType.I64), RIP), 0)

                disp = sub_op2.node.operands[0]
            else:
                base = sub_op2
                disp = sub_op1
        else:
            if sub_op1.node.opcode == VirtualDagOps.MUL:
                mul_op1 = sub_op1.node.operands[0]
                mul_op2 = sub_op1.node.operands[1]

                if mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op1
                    index = mul_op2
                elif mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op2
                    index = mul_op1
                else:
                    index = sub_op2
                base = sub_op2
            elif sub_op2.node.opcode == VirtualDagOps.MUL:
                mul_op1 = sub_op2.node.operands[0]
                mul_op2 = sub_op2.node.operands[1]

                if mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op1
                    index = mul_op2
                elif mul_op1.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT] and mul_op1.node.value.value in [1, 2, 4, 8]:
                    scale = mul_op2
                    index = mul_op1
                else:
                    index = sub_op2
                base = sub_op1
            else:
                base = sub_op1
                index = sub_op2

            disp = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I32), 0), 0)

        if disp.node.opcode == VirtualDagOps.CONSTANT:
            disp = DagValue(dag.add_target_constant_node(
                disp.ty, disp.node.value), 0)

        if scale.node.opcode == VirtualDagOps.CONSTANT:
            disp = DagValue(dag.add_target_constant_node(
                disp.ty, disp.node.value), 0)

        if node.mem_operand:
            if node.mem_operand.ptr_info and node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == VirtualDagOps.SUB:
        sub_op1 = operand.node.operands[0]
        sub_op2 = operand.node.operands[1]

        base = operand
        disp = DagValue(dag.add_target_constant_node(
            MVT(ValueType.I32), 1), 0)

        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        if node.mem_operand:
            if node.mem_operand.ptr_info and node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == VirtualDagOps.FRAME_INDEX:
        base = DagValue(dag.add_frame_index_node(
            operand.ty, operand.node.index, True), 0)

        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        disp = DagValue(dag.add_target_constant_node(MVT(ValueType.I32), 0), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        if node.mem_operand:
            if node.mem_operand.ptr_info and node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == X64DagOps.WRAPPER_RIP:
        base = DagValue(dag.add_register_node(
            MVT(ValueType.I64), MachineRegister(RIP)), 0)
        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        disp = operand.node.operands[0]
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        if node.mem_operand:
            if node.mem_operand.ptr_info and node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == X64DagOps.WRAPPER:
        disp = operand.node.operands[0]

        if disp.node.opcode in [VirtualDagOps.TARGET_GLOBAL_ADDRESS, VirtualDagOps.TARGET_CONSTANT_POOL]:
            base = DagValue(dag.add_register_node(
                MVT(ValueType.I64), noreg), 0)
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)
        elif disp.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            disp = DagValue(dag.add_target_constant_node(
                disp.node.value_types[0], disp.node.value), 0)
            base = DagValue(dag.add_register_node(
                MVT(ValueType.I64), noreg), 0)
            scale = DagValue(dag.add_target_constant_node(
                MVT(ValueType.I8), 1), 0)
            index = DagValue(dag.add_register_node(
                MVT(ValueType.I32), noreg), 0)
            segment = DagValue(dag.add_register_node(
                MVT(ValueType.I16), noreg), 0)
        else:
            assert(disp.node.opcode == VirtualDagOps.TARGET_GLOBAL_TLS_ADDRESS)
            raise NotImplementedError()

        if node.mem_operand:
            if node.mem_operand.ptr_info and node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == VirtualDagOps.CONSTANT:
        disp = DagValue(dag.add_target_constant_node(
            operand.node.value_types[0], operand.node.value), 0)

        base = DagValue(dag.add_register_node(
            MVT(ValueType.I64), noreg), 0)
        scale = DagValue(dag.add_target_constant_node(
            MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(
            MVT(ValueType.I32), noreg), 0)

        segment = DagValue(dag.add_register_node(
            MVT(ValueType.I16), noreg), 0)

        if node.mem_operand:
            if node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        return idx + 1, [base, scale, index, disp, segment]
    else:
        base = operand
        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        disp = DagValue(dag.add_target_constant_node(MVT(ValueType.I32), 0), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        if node.mem_operand:
            if node.mem_operand.ptr_info and node.mem_operand.ptr_info.value.ty.addr_space == 256:
                segment = DagValue(dag.add_register_node(
                    MVT(ValueType.I16), MachineRegister(GS)), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]

    return idx, None


addr = ComplexOperandMatcher(match_addr)


def match_lea_addr(node, operands, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.x64_gen import X64DagOps
    from rachetwrench.codegen.types import MachineValueType, ValueType
    from rachetwrench.codegen.mir import MachineRegister
    from rachetwrench.codegen.spec import NOREG
    from rachetwrench.codegen.x64_def import RIP

    if idx >= len(operands):
        return idx, None

    operand = operands[idx]

    noreg = MachineRegister(NOREG)
    MVT = MachineValueType

    if operand.node.opcode == VirtualDagOps.ADD:
        sub_op1 = operand.node.operands[0]
        sub_op2 = operand.node.operands[1]

        if sub_op1.node.opcode in [VirtualDagOps.FRAME_INDEX] and sub_op2.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            base = DagValue(dag.add_frame_index_node(
                sub_op1.ty, sub_op1.node.index, True), 0)
            disp = sub_op2
        else:
            return idx, None

        disp = DagValue(dag.add_target_constant_node(
            disp.ty, disp.node.value), 0)

        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == VirtualDagOps.SUB:
        sub_op1 = operand.node.operands[0]
        sub_op2 = operand.node.operands[1]
        if sub_op1.node.opcode in [VirtualDagOps.FRAME_INDEX] and sub_op2.node.opcode in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            base = DagValue(dag.add_frame_index_node(
                sub_op1.ty, sub_op1.node.index, True), 0)
            disp = sub_op2
        else:
            return idx, None

        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == X64DagOps.WRAPPER_RIP:
        base = DagValue(dag.add_register_node(
            MVT(ValueType.I64), MachineRegister(RIP)), 0)
        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        disp = operand.node.operands[0]
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == X64DagOps.WRAPPER:
        disp = operand.node.operands[0]

        base = DagValue(dag.add_register_node(
            MVT(ValueType.I64), noreg), 0)
        scale = DagValue(dag.add_target_constant_node(
            MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(
            MVT(ValueType.I32), noreg), 0)
        segment = DagValue(dag.add_register_node(
            MVT(ValueType.I16), noreg), 0)

        return idx + 1, [base, scale, index, disp, segment]
    elif operand.node.opcode == VirtualDagOps.FRAME_INDEX:
        base = DagValue(dag.add_frame_index_node(
            operand.ty, operand.node.index, True), 0)
        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        disp = DagValue(dag.add_target_constant_node(MVT(ValueType.I32), 0), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]

    return idx, None


lea32addr = ComplexOperandMatcher(match_lea_addr)
lea64addr = ComplexOperandMatcher(match_lea_addr)


def match_tls_addr(node, operands, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.x64_gen import X64DagOps
    from rachetwrench.codegen.types import MachineValueType, ValueType
    from rachetwrench.codegen.mir import MachineRegister
    from rachetwrench.codegen.spec import NOREG
    from rachetwrench.codegen.x64_def import RIP

    if idx >= len(operands):
        return idx, None

    operand = operands[idx]

    noreg = MachineRegister(NOREG)
    MVT = MachineValueType

    if operand.node.opcode == VirtualDagOps.TARGET_GLOBAL_TLS_ADDRESS:
        base = operand
        disp = DagValue(dag.add_target_constant_node(MVT(ValueType.I32), 0), 0)
        scale = DagValue(dag.add_target_constant_node(MVT(ValueType.I8), 1), 0)
        index = DagValue(dag.add_register_node(MVT(ValueType.I32), noreg), 0)
        segment = DagValue(dag.add_register_node(MVT(ValueType.I16), noreg), 0)

        assert(base.node.opcode != X64DagOps.WRAPPER_RIP)

        return idx + 1, [base, scale, index, disp, segment]

    return idx, None


tlsaddr = ComplexOperandMatcher(match_tls_addr)


class X64DagOp(DagOp):
    def __init__(self, name):
        super().__init__(name, "x64")


class X64DagOps(Enum):
    SUB = X64DagOp("sub")
    CMP = X64DagOp("cmp")
    FCMP = X64DagOp("fcmp")
    COMI = X64DagOp("comi")
    UCOMI = X64DagOp("ucomi")
    SETCC = X64DagOp("setcc")
    BRCOND = X64DagOp("brcond")

    SHUFP = X64DagOp("shufp")
    UNPCKL = X64DagOp("unpckl")
    UNPCKH = X64DagOp("unpckh")

    MOVSS = X64DagOp("movss")
    MOVSD = X64DagOp("movsd")

    CALL = X64DagOp("call")
    RETURN = X64DagOp("return")
    WRAPPER = X64DagOp("wrapper")
    WRAPPER_RIP = X64DagOp("wrapper_rip")

    MEMBARRIER = X64DagOp("membarrier")

    TLSADDR = X64DagOp("tlsaddr")


x64brcond_ = NodePatternMatcherGen(X64DagOps.BRCOND)
x64sub_ = NodePatternMatcherGen(X64DagOps.SUB)
x64cmp_ = NodePatternMatcherGen(X64DagOps.CMP)
x64comi_ = NodePatternMatcherGen(X64DagOps.COMI)
x64ucomi_ = NodePatternMatcherGen(X64DagOps.UCOMI)
x64setcc_ = NodePatternMatcherGen(X64DagOps.SETCC)
x64movss_ = NodePatternMatcherGen(X64DagOps.MOVSS)
x64shufp_ = NodePatternMatcherGen(X64DagOps.SHUFP)
x64tlsaddr_ = NodePatternMatcherGen(X64DagOps.TLSADDR)
x64membarrier = NodePatternMatcherGen(X64DagOps.MEMBARRIER)()


class X64MachineOps:

    @classmethod
    def insts(cls):
        for member, value in cls.__dict__.items():
            if isinstance(value, MachineInstructionDef):
                yield value

    # lea32
    LEA32r = def_inst(
        "lea32_r",
        outs=[("dst", GR32)],
        ins=[("src", AnyMem)],
        patterns=[set_(("dst", GR32), ("src", lea32addr))]
    )

    LEA64r = def_inst(
        "lea64_r",
        outs=[("dst", GR64)],
        ins=[("src", AnyMem)],
        patterns=[set_(("dst", GR64), ("src", lea64addr))]
    )

    # mov8
    MOV8ri = def_inst(
        "mov8_ri",
        outs=[("dst", GR8)],
        ins=[("src", I8Imm)],
        patterns=[set_(("dst", GR8), ("src", reloc_imm))]
    )

    MOV8rr = def_inst(
        "mov8_rr",
        outs=[("dst", GR8)],
        ins=[("src", GR8)]
    )

    MOV8mr = def_inst(
        "mov8_mr",
        outs=[],
        ins=[("dst", I8Mem), ("src", GR8)],
        patterns=[store_(("src", GR8), ("dst", addr))]
    )

    MOV8rm = def_inst(
        "mov8_rm",
        outs=[("dst", GR8)],
        ins=[("src", I8Mem)],
        patterns=[set_(("dst", GR8), load_(("src", addr)))]
    )

    # mov16
    MOV16ri = def_inst(
        "mov16_ri",
        outs=[("dst", GR16)],
        ins=[("src", I16Imm)],
        patterns=[set_(("dst", GR16), ("src", reloc_imm))]
    )

    MOV16rr = def_inst(
        "mov16_rr",
        outs=[("dst", GR16)],
        ins=[("src", GR16)]
    )

    MOV16mi = def_inst(
        "mov16_mi",
        outs=[],
        ins=[("dst", I16Mem), ("src", I16Imm)],
        patterns=[store_(("src", reloc_imm16), ("dst", addr))]
    )

    MOV16mr = def_inst(
        "mov16_mr",
        outs=[],
        ins=[("dst", I16Mem), ("src", GR16)],
        patterns=[store_(("src", GR16), ("dst", addr))]
    )

    MOV16rm = def_inst(
        "mov16_rm",
        outs=[("dst", GR16)],
        ins=[("src", I16Mem)],
        patterns=[set_(("dst", GR16), load_(("src", addr)))]
    )

    # mov32
    MOV32r0 = def_inst(
        "mov32_r0",
        outs=[("dst", GR32)],
        ins=[]
    )

    MOV32ri = def_inst(
        "mov32_ri",
        outs=[("dst", GR32)],
        ins=[("src", I32Imm)],
        patterns=[set_(("dst", GR32), ("src", reloc_imm))]
    )

    MOV32rr = def_inst(
        "mov32_rr",
        outs=[("dst", GR32)],
        ins=[("src", GR32)]
    )

    MOV32mi = def_inst(
        "mov32_mi",
        outs=[],
        ins=[("dst", I32Mem), ("src", I32Imm)],
        patterns=[store_(("src", reloc_imm32), ("dst", addr))]
    )

    MOV32mr = def_inst(
        "mov32_mr",
        outs=[],
        ins=[("dst", I32Mem), ("src", GR32)],
        patterns=[store_(("src", GR32), ("dst", addr))]
    )

    MOV32rm = def_inst(
        "mov32_rm",
        outs=[("dst", GR32)],
        ins=[("src", I32Mem)],
        patterns=[set_(("dst", GR32), load_(("src", addr)))]
    )

    # mov64
    MOV64r0 = def_inst(
        "mov64_r0",
        outs=[("dst", GR64)],
        ins=[]
    )

    MOV64rm = def_inst(
        "mov64_rm",
        outs=[("dst", GR64)],
        ins=[("src", I64Mem)],
        patterns=[set_(("dst", GR64), load_(("src", addr)))]
    )

    MOV64ri = def_inst(
        "mov64_ri",
        outs=[("dst", GR64)],
        ins=[("src", I64Imm)],
        patterns=[set_(("dst", GR64), ("src", reloc_imm))]
    )

    MOV64rr = def_inst(
        "mov64_rr",
        outs=[("dst", GR64)],
        ins=[("src", GR64)]
    )

    MOV64mi = def_inst(
        "mov64_mi",
        outs=[],
        ins=[("dst", I64Mem), ("src", I64Imm)],
        patterns=[store_(("src", reloc_imm64), ("dst", addr))]
    )

    MOV64mr = def_inst(
        "mov64_mr",
        outs=[],
        ins=[("dst", I64Mem), ("src", GR64)],
        patterns=[store_(("src", GR64), ("dst", addr))]
    )

    # movss
    MOVSSrr = def_inst(
        "movss_rr",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128)],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", VR128), x64movss_(
            ("src1", VR128), ("src2", VR128)))]
    )

    MOVSSmi = def_inst(
        "movss_mi",
        outs=[],
        ins=[("dst", F32Mem), ("src", F32Imm)],
        patterns=[store_(("src", reloc_immf32), ("dst", addr))]
    )

    MOVSSmr = def_inst(
        "movss_mr",
        outs=[],
        ins=[("dst", F32Mem), ("src", FR32)],
        patterns=[store_(("src", FR32), ("dst", addr))]
    )

    MOVSSrm = def_inst(
        "movss_rm",
        outs=[("dst", FR32)],
        ins=[("src", F32Mem)],
        patterns=[set_(("dst", FR32), load_(("src", addr)))]
    )

    VMOVSSrm = def_inst(
        "movss_rm",
        outs=[("dst", VR128)],
        ins=[("src", F32Mem)],
        patterns=[set_(("dst", FR32), load_(("src", addr)))]
    )

    # movsd
    MOVSDmi = def_inst(
        "movsd_mi",
        outs=[],
        ins=[("dst", F64Mem), ("src", F64Imm)]
    )

    MOVSDmr = def_inst(
        "movsd_mr",
        outs=[],
        ins=[("dst", F64Mem), ("src", FR64)],
        patterns=[store_(("src", FR64), ("dst", addr))]
    )

    MOVSDrm = def_inst(
        "movsd_rm",
        outs=[("dst", FR64)],
        ins=[("src", F64Mem)],
        patterns=[set_(("dst", FR64), load_(("src", addr)))]
    )

    MOVSDrr = def_inst(
        "movsd_rr",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", FR64)],
        constraints=[Constraint("dst", "src1")]
    )

    # movaps
    MOVAPSrr = def_inst(
        "movaps_rr",
        outs=[("dst", VR128)],
        ins=[("src", VR128)]
    )

    MOVAPSmr = def_inst(
        "movaps_mr",
        outs=[],
        ins=[("dst", F128Mem), ("src", VR128)],
        # patterns=[store_(("src", VR128), ("dst", addr))]
    )

    MOVAPSrm = def_inst(
        "movaps_rm",
        outs=[("dst", VR128)],
        ins=[("src", F128Mem)],
        # patterns=[set_(("dst", VR128), load_(("src", addr)))]
    )

    # movups
    MOVUPSmr = def_inst(
        "movups_mr",
        outs=[],
        ins=[("dst", F128Mem), ("src", VR128)],
        patterns=[store_(("src", VR128), ("dst", addr))]
    )

    MOVUPSrm = def_inst(
        "movups_rm",
        outs=[("dst", VR128)],
        ins=[("src", F128Mem)],
        patterns=[set_(("dst", VR128), load_(("src", addr)))]
    )

    # movq
    MOVPQIto64rr = def_inst(
        "movpqito64_rr",
        outs=[("dst", GR64)],
        ins=[("src", VR128)]
    )

    # add16
    ADD16rm = def_inst(
        "add16_rm",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", I16Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR16), EFLAGS, add_(("src1", GR16), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADD16mi = def_inst(
        "add16_mi",
        outs=[],
        ins=[("dst", I16Mem), ("src", I16Imm)],
        defs=[EFLAGS]
    )

    ADD16ri = def_inst(
        "add16_ri",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", I16Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR16), EFLAGS, add_(
            ("src1", GR16), ("src2", reloc_imm16)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADD16rr = def_inst(
        "add16_rr",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", GR16)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR16), EFLAGS, add_(
            ("src1", GR16), ("src2", GR16)))],
        constraints=[Constraint("dst", "src1")]
    )

    # add32
    ADD32rm = def_inst(
        "add32_rm",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, add_(
            ("src1", GR32), load_(("src2", addr))))],
        constraints=[Constraint("dst", "src1")]
    )

    ADD32mi = def_inst(
        "add32_mi",
        outs=[],
        ins=[("dst", I32Mem), ("src", I32Imm)],
        defs=[EFLAGS]
    )

    ADD32ri = def_inst(
        "add32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, add_(
            ("src1", GR32), ("src2", reloc_imm32)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADD32rr = def_inst(
        "add32_rr",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", GR32)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, add_(
            ("src1", GR32), ("src2", GR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # add64
    ADD64rm = def_inst(
        "add64_rm",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR64), EFLAGS, add_(("src1", GR64), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADD64mi = def_inst(
        "add64_mi",
        outs=[],
        ins=[("dst", I64Mem), ("src", I64Imm)],
        defs=[EFLAGS]
    )

    ADD64ri = def_inst(
        "add64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, add_(
            ("src1", GR64), ("src2", reloc_imm64)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADD64rr = def_inst(
        "add64_rr",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", GR64)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, add_(
            ("src1", GR64), ("src2", GR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # addss
    ADDSSrm = def_inst(
        "addss_rm",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", F32Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", FR32), EFLAGS, fadd_(("src1", FR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADDSSrr = def_inst(
        "addss_rr",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", FR32)],
        defs=[EFLAGS],
        patterns=[set_(("dst", FR32), EFLAGS, fadd_(
            ("src1", FR32), ("src2", FR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # addsd
    ADDSDrm = def_inst(
        "addsd_rm",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", F64Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", FR64), EFLAGS, fadd_(
        #     ("src1", FR64), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    ADDSDrr = def_inst(
        "addsd_rr",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", FR64)],
        defs=[EFLAGS],
        patterns=[set_(("dst", FR64), EFLAGS, fadd_(
            ("src1", FR64), ("src2", FR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # addps
    ADDPSrr = def_inst(
        "addps_rr",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128)],
        patterns=[set_(("dst", VR128), EFLAGS, fadd_(
            ("src1", VR128), ("src2", VR128)))],
        constraints=[Constraint("dst", "src1")]
    )

    # sub8
    SUB8ri = def_inst(
        "sub8_ri",
        outs=[("dst", GR8)],
        ins=[("src1", GR8), ("src2", I8Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR8), EFLAGS, x64sub_(
            ("src1", GR8), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")],
        is_compare=True
    )

    # sub32
    SUB32rm = def_inst(
        "sub32_rm",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR32), EFLAGS, x64sub_(("src1", GR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")],
        is_compare=True
    )

    SUB32mi = def_inst(
        "sub32_mi",
        outs=[],
        ins=[("dst", I32Mem), ("src", I32Imm)],
        defs=[EFLAGS],
        is_compare=True
    )

    SUB32ri = def_inst(
        "sub32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Imm)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", GR32), EFLAGS, x64sub_(
            ("src1", GR32), ("src2", reloc_imm32)))],
        is_compare=True
    )

    SUB32rr = def_inst(
        "sub32_rr",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", GR32)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", GR32), EFLAGS, x64sub_(
            ("src1", GR32), ("src2", GR32)))],
        is_compare=True
    )

    # sub64
    SUB64rm = def_inst(
        "sub64_rm",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Mem)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        is_compare=True
    )

    SUB64mi = def_inst(
        "sub64_mi",
        outs=[],
        ins=[("dst", I64Mem), ("src", I64Imm)],
        defs=[EFLAGS],
        is_compare=True
    )

    SUB64ri = def_inst(
        "sub64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, x64sub_(
            ("src1", GR64), ("src2", reloc_imm64)))],
        constraints=[Constraint("dst", "src1")],
        is_compare=True
    )

    SUB64rr = def_inst(
        "sub64_rr",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", GR64)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", GR64), EFLAGS, x64sub_(
            ("src1", GR64), ("src2", GR64)))],
        is_compare=True
    )

    # subss
    SUBSSrm = def_inst(
        "subss_rm",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", F32Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", FR32), EFLAGS, fsub_(("src1", FR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    SUBSSrr = def_inst(
        "subss_rr",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", FR32)],
        defs=[EFLAGS],
        patterns=[set_(("dst", FR32), EFLAGS, fsub_(
            ("src1", FR32), ("src2", FR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # subsd
    SUBSDrm = def_inst(
        "subsd_rm",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", F64Mem)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")]
    )

    SUBSDrr = def_inst(
        "subsd_rr",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", FR64)],
        defs=[EFLAGS],
        patterns=[set_(("dst", FR64), EFLAGS, fsub_(
            ("src1", FR64), ("src2", FR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # subps
    SUBPSrr = def_inst(
        "subps_rr",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128)],
        patterns=[set_(("dst", VR128), EFLAGS, fsub_(
            ("src1", VR128), ("src2", VR128)))],
        constraints=[Constraint("dst", "src1")]
    )

    # imul16
    IMUL16rm = def_inst(
        "imul16_rm",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", I16Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR16), EFLAGS, x64mul_(("src1", GR16), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    IMUL16rr = def_inst(
        "imul16_rr",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", GR16)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", GR16), EFLAGS, mul_(
            ("src1", GR16), ("src2", GR16)))]
    )

    # imul32
    IMUL32rm = def_inst(
        "imul32_rm",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR32), EFLAGS, x64mul_(("src1", GR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    IMUL32rr = def_inst(
        "imul32_rr",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", GR32)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", GR32), EFLAGS, mul_(
            ("src1", GR32), ("src2", GR32)))]
    )

    # imul64
    IMUL64rm = def_inst(
        "imul64_rm",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Mem)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")]
    )

    IMUL64rr = def_inst(
        "imul64_rr",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", GR64)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")],
        patterns=[set_(("dst", GR64), EFLAGS, mul_(
            ("src1", GR64), ("src2", GR64)))]
    )

    # mul32
    MUL64r = def_inst(
        "mul32_r",
        outs=[],
        ins=[("src", GR32)],
        defs=[EAX, EDX, EFLAGS],
        uses=[EAX]
    )

    # mul64
    MUL64r = def_inst(
        "mul64_r",
        outs=[],
        ins=[("src", GR64)],
        defs=[RAX, RDX, EFLAGS],
        uses=[RAX]
    )

    # mulss
    MULSSrm = def_inst(
        "mulss_rm",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", F32Mem)],
        # patterns=[set_(("dst", FR32), EFLAGS, fmul_(("src1", FR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    MULSSrr = def_inst(
        "mulss_rr",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", FR32)],
        patterns=[set_(("dst", FR32), EFLAGS, fmul_(
            ("src1", FR32), ("src2", FR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # mulsd
    MULSDrm = def_inst(
        "mulsd_rm",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", F64Mem)],
        # patterns=[set_(("dst", FR32), EFLAGS, fmul_(("src1", FR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    MULSDrr = def_inst(
        "mulsd_rr",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", FR64)],
        patterns=[set_(("dst", FR64), EFLAGS, fmul_(
            ("src1", FR64), ("src2", FR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # mulps
    MULPSrm = def_inst(
        "mulps_rm",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", F128Mem)],
        constraints=[Constraint("dst", "src1")]
    )

    MULPSrr = def_inst(
        "mulps_rr",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128)],
        patterns=[set_(("dst", VR128), EFLAGS, fmul_(
            ("src1", VR128), ("src2", VR128)))],
        constraints=[Constraint("dst", "src1")]
    )

    # idiv8
    IDIV8r = def_inst(
        "idiv8_r",
        outs=[],
        ins=[("src", GR8)],
        defs=[AL, AH, EFLAGS],
        uses=[AX]
    )

    # idiv16
    IDIV16r = def_inst(
        "idiv16_r",
        outs=[],
        ins=[("src", GR16)],
        defs=[AX, DX, EFLAGS],
        uses=[AX, DX]
    )

    # idiv32
    IDIV32r = def_inst(
        "idiv32_r",
        outs=[],
        ins=[("src", GR32)],
        defs=[EAX, EDX, EFLAGS],
        uses=[EAX, EDX]
    )

    # idiv64
    IDIV64r = def_inst(
        "idiv64_r",
        outs=[],
        ins=[("src", GR64)],
        defs=[RAX, RDX, EFLAGS],
        uses=[RAX, RDX]
    )

    # div8
    DIV8r = def_inst(
        "div8_r",
        outs=[],
        ins=[("src", GR8)],
        defs=[AL, AH, EFLAGS],
        uses=[AX]
    )

    # div16
    DIV16r = def_inst(
        "div16_r",
        outs=[],
        ins=[("src", GR16)],
        defs=[AX, DX, EFLAGS],
        uses=[AX, DX]
    )

    # div32
    DIV32r = def_inst(
        "div32_r",
        outs=[],
        ins=[("src", GR32)],
        defs=[EAX, EDX, EFLAGS],
        uses=[EAX, EDX]
    )

    # div64
    DIV64r = def_inst(
        "div64_r",
        outs=[],
        ins=[("src", GR64)],
        defs=[RAX, RDX, EFLAGS],
        uses=[RAX, RDX]
    )

    # divss
    DIVSSrm = def_inst(
        "divss_rm",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", F32Mem)],
        # patterns=[set_(("dst", FR32), EFLAGS, fdiv_(("src1", FR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    DIVSSrr = def_inst(
        "divss_rr",
        outs=[("dst", FR32)],
        ins=[("src1", FR32), ("src2", FR32)],
        patterns=[set_(("dst", FR32), EFLAGS, fdiv_(
            ("src1", FR32), ("src2", FR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # divsd
    DIVSDrm = def_inst(
        "divsd_rm",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", F64Mem)],
        # patterns=[set_(("dst", FR32), EFLAGS, fdiv_(("src1", FR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    DIVSDrr = def_inst(
        "divsd_rr",
        outs=[("dst", FR64)],
        ins=[("src1", FR64), ("src2", FR64)],
        patterns=[set_(("dst", FR64), EFLAGS, fdiv_(
            ("src1", FR64), ("src2", FR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # divps
    DIVPSrm = def_inst(
        "divps_rm",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", F128Mem)],
        constraints=[Constraint("dst", "src1")]
    )

    DIVPSrr = def_inst(
        "divps_rr",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128)],
        patterns=[set_(("dst", VR128), EFLAGS, fdiv_(
            ("src1", VR128), ("src2", VR128)))],
        constraints=[Constraint("dst", "src1")]
    )

    # and8
    AND8rr = def_inst(
        "and8_rr",
        outs=[("dst", GR8)],
        ins=[("src1", GR8), ("src2", GR8)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR8), EFLAGS, and_(
            ("src1", GR8), ("src2", GR8)))],
        constraints=[Constraint("dst", "src1")]
    )

    # and32
    AND32rm = def_inst(
        "and32_rm",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")]
    )

    AND32mi = def_inst(
        "and32_mi",
        outs=[],
        ins=[("dst", I32Mem), ("src", I32Imm)],
        defs=[EFLAGS]
    )

    AND32ri = def_inst(
        "and32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, and_(
            ("src1", GR32), ("src2", reloc_imm32)))],
        constraints=[Constraint("dst", "src1")]
    )

    AND32rr = def_inst(
        "and32_rr",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", GR32)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, and_(
            ("src1", GR32), ("src2", GR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # and64
    AND64rm = def_inst(
        "and64_rm",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Mem)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")]
    )

    AND64mi = def_inst(
        "and64_mi",
        outs=[],
        ins=[("dst", I64Mem), ("src", I64Imm)],
        defs=[EFLAGS]
    )

    AND64ri = def_inst(
        "and64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Imm)],
        defs=[EFLAGS],
        constraints=[Constraint("dst", "src1")]
    )

    AND64rr = def_inst(
        "and64_rr",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", GR64)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, and_(
            ("src1", GR64), ("src2", GR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # or8
    OR8rr = def_inst(
        "or8_rr",
        outs=[("dst", GR8)],
        ins=[("src1", GR8), ("src2", GR8)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR8), EFLAGS, or_(
            ("src1", GR8), ("src2", GR8)))],
        constraints=[Constraint("dst", "src1")]
    )

    # or32
    OR32rm = def_inst(
        "or32_rm",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR32), EFLAGS, or_(("src1", GR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    OR32mi = def_inst(
        "or32_mi",
        outs=[],
        ins=[("dst", I32Mem), ("src", I32Imm)],
        defs=[EFLAGS]
    )

    OR32ri = def_inst(
        "or32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, or_(
            ("src1", GR32), ("src2", reloc_imm32)))],
        constraints=[Constraint("dst", "src1")]
    )

    OR32rr = def_inst(
        "or32_rr",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", GR32)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, or_(
            ("src1", GR32), ("src2", GR32)))],
        constraints=[Constraint("dst", "src1")]
    )

    # or64
    OR64rm = def_inst(
        "or64_rm",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR64), EFLAGS, or_(("src1", GR64), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    OR64mi = def_inst(
        "or64_mi",
        outs=[],
        ins=[("dst", I64Mem), ("src", I64Imm)],
        defs=[EFLAGS]
    )

    OR64ri = def_inst(
        "or64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, or_(
            ("src1", GR64), ("src2", reloc_imm64)))],
        constraints=[Constraint("dst", "src1")]
    )

    OR64rr = def_inst(
        "or64_rr",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", GR64)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, or_(
            ("src1", GR64), ("src2", GR64)))],
        constraints=[Constraint("dst", "src1")]
    )

    # xor8
    XOR8ri = def_inst(
        "xor8_ri",
        outs=[("dst", GR8)],
        ins=[("src1", GR8), ("src2", I8Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR8), EFLAGS, xor_(
            ("src1", GR8), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    XOR8rr = def_inst(
        "xor8_rr",
        outs=[("dst", GR8)],
        ins=[("src1", GR8), ("src2", GR8)],
        patterns=[set_(("dst", GR8), EFLAGS, xor_(
            ("src1", GR8), ("src2", GR8)))],
        defs=[EFLAGS], constraints=[Constraint("dst", "src1")]
    )

    # xor16
    XOR16ri = def_inst(
        "xor16_ri",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", I16Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR16), EFLAGS, xor_(
            ("src1", GR16), ("src2", reloc_imm16)))],
        constraints=[Constraint("dst", "src1")]
    )

    XOR16rr = def_inst(
        "xor16_rr",
        outs=[("dst", GR16)],
        ins=[("src1", GR16), ("src2", GR16)],
        patterns=[set_(("dst", GR16), EFLAGS, xor_(
            ("src1", GR16), ("src2", GR16)))],
        defs=[EFLAGS], constraints=[Constraint("dst", "src1")]
    )

    # xor32
    XOR32rm = def_inst(
        "xor32_rm",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        # patterns=[set_(("dst", GR32), EFLAGS, xor_(("src1", GR32), ("src2", addr)))],
        constraints=[Constraint("dst", "src1")]
    )

    XOR32mi = def_inst(
        "xor32_mi",
        outs=[],
        ins=[("dst", I32Mem), ("src", I32Imm)],
        defs=[EFLAGS]
    )

    XOR32ri = def_inst(
        "xor32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I32Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR32), EFLAGS, xor_(
            ("src1", GR32), ("src2", reloc_imm32)))],
        constraints=[Constraint("dst", "src1")]
    )

    XOR32rr = def_inst(
        "xor32_rr",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", GR32)],
        patterns=[set_(("dst", GR32), EFLAGS, xor_(
            ("src1", GR32), ("src2", GR32)))],
        defs=[EFLAGS], constraints=[Constraint("dst", "src1")]
    )

    # xor64
    XOR64ri = def_inst(
        "xor64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I64Imm)],
        defs=[EFLAGS],
        patterns=[set_(("dst", GR64), EFLAGS, xor_(
            ("src1", GR64), ("src2", reloc_imm64)))],
        constraints=[Constraint("dst", "src1")]
    )

    XOR64rr = def_inst(
        "xor64_rr",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", GR64)],
        patterns=[set_(("dst", GR64), EFLAGS, xor_(
            ("src1", GR64), ("src2", GR64)))],
        defs=[EFLAGS], constraints=[Constraint("dst", "src1")]
    )

    # xorps
    XORPSrr = def_inst(
        "xorps_rr",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128)],
        constraints=[Constraint("dst", "src1")]
    )

    # shr32
    SHR32ri = def_inst(
        "shr32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I8Imm)],
        patterns=[set_(("dst", GR32), srl_(
            ("src1", GR32), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    SHR32rCL = def_inst(
        "shr32_rcl",
        outs=[("dst", GR32)],
        ins=[("src", GR32)],
        uses=[CL],
        # patterns=[
        #     set_(("dst", GR32), srl_(("src", GR32), CL))],
        constraints=[Constraint("dst", "src")]
    )

    # shr64
    SHR64ri = def_inst(
        "shr64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I8Imm)],
        patterns=[set_(("dst", GR64), srl_(
            ("src1", GR64), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    SHR64rCL = def_inst(
        "shr64_rcl",
        outs=[("dst", GR64)],
        ins=[("src", GR64)],
        uses=[CL],
        # patterns=[
        #     set_(("dst", GR64), srl_(("src", GR64), CL))],
        constraints=[Constraint("dst", "src")]
    )

    # sar32
    SAR32ri = def_inst(
        "sar32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I8Imm)],
        patterns=[set_(("dst", GR32), sra_(
            ("src1", GR32), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    SAR32rCL = def_inst(
        "sar32_rcl",
        outs=[("dst", GR32)],
        ins=[("src", GR32)],
        uses=[CL],
        # patterns=[
        #     set_(("dst", GR32), sra_(("src", GR32), CL))],
        constraints=[Constraint("dst", "src")]
    )

    # sar64
    SAR64ri = def_inst(
        "sar64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I8Imm)],
        patterns=[set_(("dst", GR64), sra_(
            ("src1", GR64), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    SAR64rCL = def_inst(
        "sar64_rcl",
        outs=[("dst", GR64)],
        ins=[("src", GR64)],
        uses=[CL],
        # patterns=[
        #     set_(("dst", GR64), sra_(("src", GR64), CL))],
        constraints=[Constraint("dst", "src")]
    )

    # shl32
    SHL32ri = def_inst(
        "shl32_ri",
        outs=[("dst", GR32)],
        ins=[("src1", GR32), ("src2", I8Imm)],
        patterns=[set_(("dst", GR32), shl_(
            ("src1", GR32), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    SHL32rCL = def_inst(
        "shl32_rcl",
        outs=[("dst", GR32)],
        ins=[("src", GR32)],
        uses=[CL],
        # patterns=[
        #     set_(("dst", GR32), shl_(("src", GR32), CL))],
        constraints=[Constraint("dst", "src")]
    )

    # shl64
    SHL64ri = def_inst(
        "shl64_ri",
        outs=[("dst", GR64)],
        ins=[("src1", GR64), ("src2", I8Imm)],
        patterns=[set_(("dst", GR64), shl_(
            ("src1", GR64), ("src2", reloc_imm8)))],
        constraints=[Constraint("dst", "src1")]
    )

    SHL64rCL = def_inst(
        "shl64_rcl",
        outs=[("dst", GR64)],
        ins=[("src", GR64)],
        uses=[CL],
        # patterns=[set_(("dst", GR64), shl_(("src", GR64), CL))],
        constraints=[Constraint("dst", "src")]
    )

    # shufps
    SHUFPSrri = def_inst(
        "shufps_rri",
        outs=[("dst", VR128)],
        ins=[("src1", VR128), ("src2", VR128), ("src3", I8Imm)],
        patterns=[set_(("dst", VR128), x64shufp_(
            ("src1", VR128), ("src2", VR128), ("src3", timm)))],
        constraints=[Constraint("dst", "src1")]
    )

    # cmp8
    CMP8ri = def_inst(
        "cmp8_ri",
        outs=[],
        ins=[("src1", GR8), ("src2", I8Imm)],
        defs=[EFLAGS],
        patterns=[set_(EFLAGS, x64cmp_(
            ("src1", GR8), ("src2", reloc_imm8)))],
        is_compare=True
    )

    # cmp32
    CMP32rm = def_inst(
        "cmp32_rm",
        outs=[],
        ins=[("src1", GR32), ("src2", I32Mem)],
        defs=[EFLAGS],
        is_compare=True
    )

    CMP32rr = def_inst(
        "cmp32_rr",
        outs=[],
        ins=[("src1", GR32), ("src2", GR32)],
        defs=[EFLAGS],
        is_compare=True
    )

    CMP32ri = def_inst(
        "cmp32_ri",
        outs=[],
        ins=[("src1", GR32), ("src2", I32Imm)],
        defs=[EFLAGS],
        is_compare=True
    )

    # cmp64
    CMP64rm = def_inst(
        "cmp64_rm",
        outs=[],
        ins=[("src1", GR64), ("src2", I64Mem)],
        defs=[EFLAGS],
        is_compare=True
    )

    CMP64rr = def_inst(
        "cmp64_rr",
        outs=[],
        ins=[("src1", GR64), ("src2", GR64)],
        defs=[EFLAGS],
        is_compare=True
    )

    CMP64ri = def_inst(
        "cmp64_ri",
        outs=[],
        ins=[("src1", GR64), ("src2", I64Imm)],
        defs=[EFLAGS],
        is_compare=True
    )

    # comiss
    COMISSrr = def_inst(
        "comiss_rr",
        outs=[],
        ins=[("src1", FR32), ("src2", FR32)],
        defs=[EFLAGS],
        patterns=[set_(EFLAGS, x64comi_(
            ("src1", FR32), ("src2", FR32)))],
        is_compare=True
    )

    # comisd
    COMISDrr = def_inst(
        "comisd_rr",
        outs=[],
        ins=[("src1", FR64), ("src2", FR64)],
        defs=[EFLAGS],
        patterns=[set_(EFLAGS, x64comi_(
            ("src1", FR64), ("src2", FR64)))],
        is_compare=True
    )

    # ucomiss
    UCOMISSrr = def_inst(
        "ucomiss_rr",
        outs=[],
        ins=[("src1", FR32), ("src2", FR32)],
        defs=[EFLAGS],
        patterns=[set_(EFLAGS, x64ucomi_(
            ("src1", FR32), ("src2", FR32)))],
        is_compare=True
    )

    # ucomisd
    UCOMISDrr = def_inst(
        "ucomisd_rr",
        outs=[],
        ins=[("src1", FR64), ("src2", FR64)],
        defs=[EFLAGS],
        patterns=[set_(EFLAGS, x64ucomi_(
            ("src1", FR64), ("src2", FR64)))],
        is_compare=True
    )

    # cvtsd2ss
    CVTSD2SSrr = def_inst(
        "cvtsd2ss_rr",
        outs=[("dst", FR32)],
        ins=[("src", FR64)],
        patterns=[set_(("dst", FR32), fp_round_(
            ("src", FR64)))]
    )

    # cvtss2sd
    CVTSS2SDrr = def_inst(
        "cvtss2sd_rr",
        outs=[("dst", FR64)],
        ins=[("src", FR32)],
        patterns=[set_(("dst", FR64), fp_extend_(
            ("src", FR32)))]
    )

    # cvttss2si
    CVTTSS2SIrr = def_inst(
        "cvttss2si_rr",
        outs=[("dst", GR32)],
        ins=[("src", FR32)],
        patterns=[set_(("dst", GR32), fp_to_sint_(
            ("src", FR32)))]
    )
    CVTTSS2SI64rr = def_inst(
        "cvttss2si64_rr",
        outs=[("dst", GR64)],
        ins=[("src", FR32)],
        patterns=[set_(("dst", GR64), fp_to_sint_(
            ("src", FR32)))]
    )

    # cvttsd2si
    CVTTSD2SIrr = def_inst(
        "cvttsd2si_rr",
        outs=[("dst", GR32)],
        ins=[("src", FR64)],
        patterns=[set_(("dst", GR32), fp_to_sint_(
            ("src", FR64)))]
    )
    CVTTSD2SI64rr = def_inst(
        "cvttsd2si64_rr",
        outs=[("dst", GR64)],
        ins=[("src", FR64)],
        patterns=[set_(("dst", GR64), fp_to_sint_(
            ("src", FR64)))]
    )

    # cvtsi2ss
    CVTSI2SSrr = def_inst(
        "cvtsi2ss_rr",
        outs=[("dst", FR32)],
        ins=[("src", GR32)],
        patterns=[set_(("dst", FR32), sint_to_fp_(
            ("src", GR32)))]
    )
    CVTSI642SSrr = def_inst(
        "cvtsi642ss_rr",
        outs=[("dst", FR32)],
        ins=[("src", GR64)],
        patterns=[set_(("dst", FR32), sint_to_fp_(
            ("src", GR64)))]
    )

    # cvtsi2sd
    CVTSI2SDrr = def_inst(
        "cvtsi2sd_rr",
        outs=[("dst", FR64)],
        ins=[("src", GR32)],
        patterns=[set_(("dst", FR64), sint_to_fp_(
            ("src", GR32)))]
    )
    CVTSI642SDrr = def_inst(
        "cvtsi642sd_rr",
        outs=[("dst", FR64)],
        ins=[("src", GR64)],
        patterns=[set_(("dst", FR64), sint_to_fp_(
            ("src", GR64)))]
    )

    # movsx
    MOVSX32rr8 = def_inst(
        "movsx32_rr8",
        outs=[("dst", GR32)],
        ins=[("src", GR8)],
        patterns=[set_(("dst", GR32), sext_(("src", GR8)))]
    )

    MOVSX32rr16 = def_inst(
        "movsx32_rr16",
        outs=[("dst", GR32)],
        ins=[("src", GR16)],
        patterns=[set_(("dst", GR32), sext_(("src", GR16)))]
    )

    MOVSX64rr32 = def_inst(
        "movsx64_rr32",
        outs=[("dst", GR64)],
        ins=[("src", GR32)],
        patterns=[set_(("dst", GR64), sext_(("src", GR32)))]
    )

    # movzx
    MOVZX16rr8 = def_inst(
        "movzx16_rr8",
        outs=[("dst", GR16)],
        ins=[("src", GR8)],
        patterns=[set_(("dst", GR16), zext_(("src", GR8)))]
    )

    MOVZX32rr8 = def_inst(
        "movzx32_rr8",
        outs=[("dst", GR32)],
        ins=[("src", GR8)],
        patterns=[set_(("dst", GR32), zext_(("src", GR8)))]
    )

    MOVZX32rr16 = def_inst(
        "movzx32_rr16",
        outs=[("dst", GR32)],
        ins=[("src", GR16)],
        patterns=[set_(("dst", GR32), zext_(("src", GR16)))]
    )

    MOVZX64rr8 = def_inst(
        "movzx64_rr8",
        outs=[("dst", GR64)],
        ins=[("src", GR8)],
        patterns=[set_(("dst", GR64), zext_(("src", GR8)))]
    )

    MOVZX64rr16 = def_inst(
        "movzx64_rr16",
        outs=[("dst", GR64)],
        ins=[("src", GR16)],
        patterns=[set_(("dst", GR64), zext_(("src", GR16)))]
    )

    # extload
    MOVZX32rm8 = def_inst(
        "movzx32_rm8",
        outs=[("dst", GR32)],
        ins=[("src", I8Mem)],
        patterns=[set_(("dst", GR32), i32_(zextloadi8_(("src", addr))))]
    )
    MOVZX32rm16 = def_inst(
        "movzx32_rm16",
        outs=[("dst", GR32)],
        ins=[("src", I16Mem)],
        patterns=[set_(("dst", GR32), i32_(zextloadi16_(("src", addr))))]
    )
    MOVZX64rm8 = def_inst(
        "movzx64_rm8",
        outs=[("dst", GR64)],
        ins=[("src", I8Mem)],
        patterns=[set_(("dst", GR64), i64_(zextloadi8_(("src", addr))))]
    )
    MOVZX64rm16 = def_inst(
        "movzx64_rm16",
        outs=[("dst", GR64)],
        ins=[("src", I16Mem)],
        patterns=[set_(("dst", GR64), i64_(zextloadi16_(("src", addr))))]
    )

    MOVSX64rm8 = def_inst(
        "movsx64_rm8",
        outs=[("dst", GR64)],
        ins=[("src", I8Mem)],
        patterns=[set_(("dst", GR64), i64_(sextloadi8_(("src", addr))))]
    )
    MOVSX64rm16 = def_inst(
        "movsx64_rm16",
        outs=[("dst", GR64)],
        ins=[("src", I16Mem)],
        patterns=[set_(("dst", GR64), i64_(sextloadi16_(("src", addr))))]
    )
    MOVSX64rm32 = def_inst(
        "movsx64_rm32",
        outs=[("dst", GR64)],
        ins=[("src", I32Mem)],
        patterns=[set_(("dst", GR64), i64_(sextloadi32_(("src", addr))))]
    )

    # convert byte to word
    CBW = def_inst(
        "cbw",
        outs=[],
        ins=[],
        defs=[AX],
        uses=[AL]
    )

    # convert word to double word
    CWDE = def_inst(
        "cwde",
        outs=[],
        ins=[],
        defs=[EAX],
        uses=[AX]
    )

    # convert double word to quad word
    CDQE = def_inst(
        "cdqe",
        outs=[],
        ins=[],
        defs=[RAX],
        uses=[EAX]
    )

    # convert word to double word
    CWD = def_inst(
        "cwd",
        outs=[],
        ins=[],
        defs=[AX, DX],
        uses=[AX]
    )

    # convert double word to quad word
    CDQ = def_inst(
        "cdq",
        outs=[],
        ins=[],
        defs=[EAX, EDX],
        uses=[EAX]
    )

    # convert quad word to octet word
    CQO = def_inst(
        "cqo",
        outs=[],
        ins=[],
        defs=[RAX, RDX],
        uses=[RAX]
    )

    # jcc
    JCC_1 = def_inst(
        "jcc",
        outs=[],
        ins=[("dst", BrTarget8), ("cond", I32Imm)],
        uses=[EFLAGS],
        patterns=[x64brcond_(
            ("dst", bb), ("cond", timm), EFLAGS)],
        is_terminator=True
    )

    JCC_2 = def_inst(
        "jcc",
        outs=[],
        ins=[("dst", BrTarget16), ("cond", I32Imm)],
        uses=[EFLAGS],
        is_terminator=True
    )

    JCC_4 = def_inst(
        "jcc",
        outs=[],
        ins=[("dst", BrTarget32), ("cond", I32Imm)],
        uses=[EFLAGS],
        is_terminator=True
    )

    # jmp
    JMP_1 = def_inst(
        "jmp",
        outs=[],
        ins=[("dst", BrTarget8)],
        patterns=[br_(("dst", bb))],
        is_terminator=True
    )

    JMP_2 = def_inst(
        "jmp",
        outs=[],
        ins=[("dst", BrTarget16)],
        is_terminator=True
    )

    JMP_4 = def_inst(
        "jmp",
        outs=[],
        ins=[("dst", BrTarget32)],
        is_terminator=True
    )

    # setcc
    SETCCr = def_inst(
        "setcc_r",
        outs=[("dst", GR8)],
        ins=[("cond", I32Imm)],
        uses=[EFLAGS],
        patterns=[
            set_(("dst", GR8), x64setcc_(("cond", timm), EFLAGS))]
    )

    # push
    PUSH32r = def_inst(
        "push32_r",
        outs=[],
        ins=[("src", GR32)],
        uses=[ESP],
        defs=[ESP]
    )

    PUSH64r = def_inst(
        "push64_r",
        outs=[],
        ins=[("src", GR64)],
        uses=[RSP],
        defs=[RSP]
    )

    # pop
    POP32r = def_inst(
        "pop32_r",
        outs=[],
        ins=[("dst", GR32)],
        uses=[ESP],
        defs=[ESP]
    )

    POP64r = def_inst(
        "pop64_r",
        outs=[],
        ins=[("dst", GR64)],
        uses=[RSP],
        defs=[RSP]
    )

    V_SET0 = def_inst(
        "v_set0",
        outs=[("dst", VR128)],
        ins=[]
    )

    CALLpcrel32 = def_inst(
        "call",
        outs=[],
        ins=[("dst", GR8)],
        is_call=True,
        defs=[RSP]
    )

    RET = def_inst(
        "ret",
        outs=[],
        ins=[],
        is_terminator=True
    )

    ADJCALLSTACKDOWN32 = def_inst(
        "ADJCALLSTACKDOWN",
        outs=[],
        ins=[("amt1", I32Imm),
             ("amt2", I32Imm), ("amt3", I32Imm)]
    )

    ADJCALLSTACKUP32 = def_inst(
        "ADJCALLSTACKUP",
        outs=[],
        ins=[("amt1", I32Imm), ("amt2", I32Imm)]
    )

    MEMBARRIER = def_inst(
        "MEMBARRIER",
        outs=[],
        ins=[],
        patterns=[x64membarrier]
    )

    TLSADDR64 = def_inst(
        "TLSADDR64",
        outs=[],
        ins=[("sym", I64Mem)],
        defs=[RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
        patterns=[x64tlsaddr_(("sym", tlsaddr))]
    )

    DATA16_PREFIX = def_inst(
        "DATA16_PREFIX",
        outs=[],
        ins=[])
    REX64_PREFIX = def_inst(
        "REX64_PREFIX",
        outs=[],
        ins=[])


def loadf32_(addr): return f32_(load_(addr))


V_SET0 = def_inst_node_(X64MachineOps.V_SET0)
VMOVSSrm = def_inst_node_(X64MachineOps.VMOVSSrm)
MOV8rm = def_inst_node_(X64MachineOps.MOV8rm)

x64_patterns = []


def def_pat_x64(pattern, result):
    def_pat(pattern, result, x64_patterns)


def_pat_x64(v4f32_(imm_zero_vec), V_SET0())
def_pat_x64(v4f32_(imm_zero_vec), V_SET0())
# def_pat(v4f32_(scalar_to_vector_(loadf32_(("src", addr)))), VMOVSSrm(("src", F32Mem)))
def_pat_x64(i8_(zextloadi1_(("src", addr))), MOV8rm(("src", I8Mem)))
