from codegen.spec import *
from codegen.mir_emitter import *
from codegen.matcher import *

ssub_0 = def_subreg("ssub_0", 32)
ssub_1 = def_subreg("ssub_1", 32, 32)

dsub_0 = def_subreg("dsub_0", 64)
dsub_1 = def_subreg("dsub_1", 64, 64)

ssub_2 = def_composed_subreg("ssub_2", dsub_1, ssub_0)
ssub_3 = def_composed_subreg("ssub_3", dsub_1, ssub_1)


# integer registers
R0 = def_reg("r0", encoding=0)
R1 = def_reg("r1", encoding=1)
R2 = def_reg("r2", encoding=2)
R3 = def_reg("r3", encoding=3)
R4 = def_reg("r4", encoding=4)
R5 = def_reg("r5", encoding=5)
R6 = def_reg("r6", encoding=6)
R7 = def_reg("r7", encoding=7)
R8 = def_reg("r8", encoding=8)
R9 = def_reg("r9", encoding=9)
R10 = def_reg("r10", encoding=10)
R11 = def_reg("r11", encoding=11)
R12 = def_reg("r12", encoding=12)
SP = def_reg("sp", encoding=13)
LR = def_reg("lr", encoding=14)
PC = def_reg("pc", encoding=15)

# floating-point registers
S0 = def_reg("s0", encoding=0)
S1 = def_reg("s1", encoding=1)
S2 = def_reg("s2", encoding=2)
S3 = def_reg("s3", encoding=3)
S4 = def_reg("s4", encoding=4)
S5 = def_reg("s5", encoding=5)
S6 = def_reg("s6", encoding=6)
S7 = def_reg("s7", encoding=7)
S8 = def_reg("s8", encoding=8)
S9 = def_reg("s9", encoding=9)
S10 = def_reg("s10", encoding=10)
S11 = def_reg("s11", encoding=11)
S12 = def_reg("s12", encoding=12)
S13 = def_reg("s13", encoding=13)
S14 = def_reg("s14", encoding=14)
S15 = def_reg("s15", encoding=15)
S16 = def_reg("s16", encoding=16)
S17 = def_reg("s17", encoding=17)
S18 = def_reg("s18", encoding=18)
S19 = def_reg("s19", encoding=19)
S20 = def_reg("s20", encoding=20)
S21 = def_reg("s21", encoding=21)
S22 = def_reg("s22", encoding=22)
S23 = def_reg("s23", encoding=23)
S24 = def_reg("s24", encoding=24)
S25 = def_reg("s25", encoding=25)
S26 = def_reg("s26", encoding=26)
S27 = def_reg("s27", encoding=27)
S28 = def_reg("s28", encoding=28)
S29 = def_reg("s29", encoding=29)
S30 = def_reg("s30", encoding=30)
S31 = def_reg("s31", encoding=31)

D0 = def_reg("d0", [S0, S1], [ssub_0, ssub_1], encoding=0)
D1 = def_reg("d1", [S2, S3], [ssub_0, ssub_1], encoding=1)
D2 = def_reg("d2", [S4, S5], [ssub_0, ssub_1], encoding=2)
D3 = def_reg("d3", [S6, S7], [ssub_0, ssub_1], encoding=3)
D4 = def_reg("d4", [S8, S9], [ssub_0, ssub_1], encoding=4)
D5 = def_reg("d5", [S10, S11], [ssub_0, ssub_1], encoding=5)
D6 = def_reg("d6", [S12, S13], [ssub_0, ssub_1], encoding=6)
D7 = def_reg("d7", [S14, S15], [ssub_0, ssub_1], encoding=7)
D8 = def_reg("d8", [S16, S17], [ssub_0, ssub_1], encoding=8)
D9 = def_reg("d9", [S18, S19], [ssub_0, ssub_1], encoding=9)
D10 = def_reg("d10", [S20, S21], [ssub_0, ssub_1], encoding=10)
D11 = def_reg("d11", [S22, S23], [ssub_0, ssub_1], encoding=11)
D12 = def_reg("d12", [S24, S25], [ssub_0, ssub_1], encoding=12)
D13 = def_reg("d13", [S26, S27], [ssub_0, ssub_1], encoding=13)
D14 = def_reg("d14", [S28, S29], [ssub_0, ssub_1], encoding=14)
D15 = def_reg("d15", [S30, S31], [ssub_0, ssub_1], encoding=15)

D16 = def_reg("d16", encoding=16)
D17 = def_reg("d17", encoding=17)
D18 = def_reg("d18", encoding=18)
D19 = def_reg("d19", encoding=19)
D20 = def_reg("d20", encoding=20)
D21 = def_reg("d21", encoding=21)
D22 = def_reg("d22", encoding=22)
D23 = def_reg("d23", encoding=23)
D24 = def_reg("d24", encoding=24)
D25 = def_reg("d25", encoding=25)
D26 = def_reg("d26", encoding=26)
D27 = def_reg("d27", encoding=27)
D28 = def_reg("d28", encoding=28)
D29 = def_reg("d29", encoding=29)
D30 = def_reg("d30", encoding=30)
D31 = def_reg("d31", encoding=31)

Q0 = def_reg("q0", [D0, D1], [dsub_0, dsub_1], encoding=0 << 1)
Q1 = def_reg("q1", [D2, D3], [dsub_0, dsub_1], encoding=1 << 1)
Q2 = def_reg("q2", [D4, D5], [dsub_0, dsub_1], encoding=2 << 1)
Q3 = def_reg("q3", [D6, D7], [dsub_0, dsub_1], encoding=3 << 1)
Q4 = def_reg("q4", [D8, D9], [dsub_0, dsub_1], encoding=4 << 1)
Q5 = def_reg("q5", [D10, D11], [dsub_0, dsub_1], encoding=5 << 1)
Q6 = def_reg("q6", [D12, D13], [dsub_0, dsub_1], encoding=6 << 1)
Q7 = def_reg("q7", [D14, D15], [dsub_0, dsub_1], encoding=7 << 1)
Q8 = def_reg("q8", [D16, D17], [dsub_0, dsub_1], encoding=8 << 1)
Q9 = def_reg("q9", [D18, D19], [dsub_0, dsub_1], encoding=9 << 1)
Q10 = def_reg("q10", [D20, D20], [dsub_0, dsub_1], encoding=10 << 1)
Q11 = def_reg("q11", [D22, D23], [dsub_0, dsub_1], encoding=11 << 1)
Q12 = def_reg("q12", [D24, D25], [dsub_0, dsub_1], encoding=12 << 1)
Q13 = def_reg("q13", [D26, D27], [dsub_0, dsub_1], encoding=13 << 1)
Q14 = def_reg("q14", [D28, D29], [dsub_0, dsub_1], encoding=14 << 1)
Q15 = def_reg("q15", [D30, D31], [dsub_0, dsub_1], encoding=15 << 1)

CPSR = def_reg("cpsr", encoding=0)


arm_regclasses = []


def def_arm_regclass(*args, **kwargs):
    regclass = def_regclass(*args, **kwargs)
    arm_regclasses.append(regclass)
    return regclass


GPR = def_arm_regclass("GPR", [ValueType.I32], 32, [
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, SP, LR, PC])

GPRwoPC = def_arm_regclass("GPRwoPC", [ValueType.I32], 32, [
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, SP, LR])

GPRTh = def_arm_regclass("GPRTh", [ValueType.I32], 32, [
    R0, R1, R2, R3, R4, R5, R6, R7])

SPR = def_arm_regclass("SPR", [ValueType.F32], 32, [
    globals()[f"S{idx}"] for idx in range(32)])

DPR = def_arm_regclass("DPR", [ValueType.F64, ValueType.V2F32], 64, [
    globals()[f"D{idx}"] for idx in range(31)])

QPR = def_arm_regclass("QPR", [ValueType.V4F32, ValueType.V2F64], 128, [
    globals()[f"Q{idx}"] for idx in range(15)])

# build register graph
reg_graph = compute_reg_graph()
reg_groups = compute_reg_groups(reg_graph)
compute_reg_subregs_all(reg_graph)
compute_reg_superregs_all(reg_graph)

# infer regclass
for regclass in arm_regclasses:
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
VectorIndex32 = ValueOperandDef(ValueType.I32)


class ARMMemOperandDef(ValueOperandDef):
    def __init__(self, op_info):
        super().__init__(ValueType.I32)

        self.op_info = op_info

    @property
    def operand_info(self):
        return self.op_info


AddrMode_Imm12 = ARMMemOperandDef([GPR, I32Imm])
AddrMode5 = ARMMemOperandDef([GPR, I32Imm])
AddrMode6 = ARMMemOperandDef([GPR, I32Imm])
SORegImm = ARMMemOperandDef([GPR, I32Imm])
arm_bl_target = ValueOperandDef(ValueType.I32)
reglist = ValueOperandDef(ValueType.I32)
cmovpred = ValueOperandDef(ValueType.I32)


class SDNode:
    def __init__(self, opcode):
        self.opcode = opcode


FrameIndex = SDNode(VirtualDagOps.FRAME_INDEX)
ADD = SDNode(VirtualDagOps.ADD)
SUB = SDNode(VirtualDagOps.SUB)


class ARMDagOp(DagOp):
    def __init__(self, name):
        super().__init__(name, "arm_")


class ARMDagOps(Enum):
    SUB = ARMDagOp("sub")
    CMP = ARMDagOp("cmp")
    SETCC = ARMDagOp("setcc")
    BRCOND = ARMDagOp("brcond")

    VDUP = ARMDagOp("vdup")

    SHUFP = ARMDagOp("shufp")
    UNPCKL = ARMDagOp("unpckl")
    UNPCKH = ARMDagOp("unpckh")

    MOVSS = ARMDagOp("movss")
    MOVSD = ARMDagOp("movsd")

    CMPFP = ARMDagOp("cmpfp")
    FMSTAT = ARMDagOp("fmstat")

    CALL = ARMDagOp("call")
    RETURN = ARMDagOp("return")

    WRAPPER = ARMDagOp("wrapper")
    WRAPPER_PIC = ARMDagOp("wrapper_pic")
    WRAPPER_JT = ARMDagOp("wrapper_jt")

    PIC_ADD = ARMDagOp("pic_add")


arm_brcond_ = NodePatternMatcherGen(ARMDagOps.BRCOND)
arm_sub_ = NodePatternMatcherGen(ARMDagOps.SUB)
arm_cmp_ = NodePatternMatcherGen(ARMDagOps.CMP)
arm_cmpfp_ = NodePatternMatcherGen(ARMDagOps.CMPFP)
arm_fmstat_ = NodePatternMatcherGen(ARMDagOps.FMSTAT)
arm_setcc_ = NodePatternMatcherGen(ARMDagOps.SETCC)
arm_movss_ = NodePatternMatcherGen(ARMDagOps.MOVSS)
arm_shufp_ = NodePatternMatcherGen(ARMDagOps.SHUFP)
arm_let_ = NodePatternMatcherGen(ARMDagOps.RETURN)
arm_wrapper_ = NodePatternMatcherGen(ARMDagOps.WRAPPER)
arm_pic_add_ = NodePatternMatcherGen(ARMDagOps.PIC_ADD)


def in_bits_signed(value, bits):
    mx = (1 << (bits - 1)) - 1
    mn = -mx - 1
    return value >= mn and value <= mx


def in_bits_unsigned(value, bits):
    mx = (1 << (bits)) - 1
    return value >= 0 and value <= mx

# addrmode imm12 'reg +/- imm12'


def match_addrmode_imm12(node, values, idx, dag):
    from codegen.dag import VirtualDagOps, DagValue
    from codegen.types import ValueType

    value = values[idx]

    if value.node.opcode == VirtualDagOps.FRAME_INDEX:
        index = value.node.index
        base = DagValue(
            dag.add_frame_index_node(value.ty, index, True), 0)
        offset = DagValue(dag.add_target_constant_node(
            value.ty, 0), 0)

        return idx + 1, [base, offset]

    if value.node.opcode == VirtualDagOps.ADD:
        value1 = value.node.operands[0]
        value2 = value.node.operands[1]

        if value1.node.opcode == VirtualDagOps.CONSTANT:
            base = value2
            offset = DagValue(dag.add_target_constant_node(
                value1.ty, value1.node.value), 0)

            return idx + 1, [base, offset]
        elif value2.node.opcode == VirtualDagOps.CONSTANT:
            base = value1
            offset = DagValue(dag.add_target_constant_node(
                value2.ty, value2.node.value), 0)

            return idx + 1, [base, offset]

    if value.node.opcode == VirtualDagOps.SUB:
        if value2.node.opcode == VirtualDagOps.CONSTANT:
            raise NotImplementedError()

    # if value.node.opcode == ARMDagOps.WRAPPER:
    #     base = value.node.operands[0]
    #     offset = DagValue(dag.add_target_constant_node(
    #         value.ty, 0), 0)

    #     return idx + 1, MatcherResult([base, offset])

    # only base.
    base = value

    assert(base.node.opcode != VirtualDagOps.TARGET_CONSTANT)

    offset = DagValue(dag.add_target_constant_node(
        value.ty, 0), 0)

    return idx + 1, [base, offset]


addrmode_imm12 = ComplexOperandMatcher(match_addrmode_imm12)


# addrmode 5 'reg +/- (imm8 << 2)'
def match_addrmode5(node, values, idx, dag: Dag):
    from codegen.dag import VirtualDagOps, DagValue
    from codegen.types import ValueType

    value = values[idx]

    if value.node.opcode == VirtualDagOps.FRAME_INDEX:
        index = value.node.index
        base = DagValue(
            dag.add_frame_index_node(value.ty, index, True), 0)
        offset = DagValue(dag.add_target_constant_node(
            value.ty, 0), 0)

        return idx + 1, [base, offset]

    if value.node.opcode == VirtualDagOps.ADD:
        value1 = value.node.operands[0]
        value2 = value.node.operands[1]

        if value1.node.opcode == VirtualDagOps.CONSTANT:
            base = value2

            assert(value1.node.value.value % 4 == 0)
            offset = DagValue(dag.add_target_constant_node(
                value1.ty, ConstantInt(value1.node.value.value >> 2, value1.node.value.ty)), 0)

            return idx + 1, [base, offset]
        elif value2.node.opcode == VirtualDagOps.CONSTANT:
            base = value1

            assert(value2.node.value.value % 4 == 0)
            offset = DagValue(dag.add_target_constant_node(
                value2.ty, ConstantInt(value2.node.value.value >> 2, value2.node.value.ty)), 0)

            return idx + 1, [base, offset]

    if value.node.opcode == VirtualDagOps.SUB:
        if value2.node.opcode == VirtualDagOps.CONSTANT:
            raise NotImplementedError()

    # if value.node.opcode == ARMDagOps.WRAPPER:
    #     base = value.node.operands[0]
    #     offset = DagValue(dag.add_target_constant_node(
    #         value.ty, 0), 0)

    #     return idx + 1, MatcherResult([base, offset])

    # only base.
    base = value

    assert(base.node.opcode != VirtualDagOps.TARGET_CONSTANT)

    offset = DagValue(dag.add_target_constant_node(
        value.ty, 0), 0)

    return idx + 1, [base, offset]


addrmode5 = ComplexOperandMatcher(match_addrmode5)


# addrmode 6
def match_addrmode6(node, values, idx, dag: Dag):
    from codegen.dag import VirtualDagOps, DagValue
    from codegen.types import ValueType

    value = values[idx]

    addr = value

    align = DagValue(dag.add_target_constant_node(
        value.ty, 1), 0)  # TODO: Need alignment information

    return idx + 1, [addr, align]


addrmode6 = ComplexOperandMatcher(match_addrmode6)


def match_imm0_65535(node, values, idx, dag):
    from codegen.dag import VirtualDagOps
    from codegen.types import ValueType

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    if not in_bits_unsigned(value.node.value.value, 16):
        return idx, None

    target_value = DagValue(dag.add_target_constant_node(
        value.ty, value.node.value), 0)

    return idx + 1, target_value


imm0_65535 = ComplexOperandMatcher(match_imm0_65535)


def count_trailing_zeros(value):
    if value == 0:
        return 0

    cnt = 0
    while (value & 0x1) == 0:
        value = value >> 1
        cnt += 1
    return cnt


def rotate_right32(value, count):
    move_mask = (1 << count) - 1
    return ((value & move_mask) << (32 - count)) | ((value & ~move_mask) >> count)


def rotate_left32(value, count):
    move_mask = ((1 << count) - 1) << (32 - count)
    return ((value & ~move_mask) << count) | ((value & move_mask) >> (32 - count))


def get_mod_imm(value):
    value = value & 0xffffffff

    mask = ((1 << 8) - 1)
    if (value & ~mask) == 0:
        return value & mask

    ctz = count_trailing_zeros(value)
    shift = ctz & ~1
    if (rotate_right32(value, shift) & ~mask) == 0:
        shift = (32 - shift) & 31
        return (rotate_left32(value, shift) & mask) | ((shift >> 1) << 8)

    return -1


def match_mod_imm(node, values, idx, dag):
    from codegen.dag import VirtualDagOps
    from codegen.types import ValueType

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    constant = value.node.value.value
    if get_mod_imm(constant) != -1:
        target_value = DagValue(dag.add_target_constant_node(
            value.ty, constant), 0)

        return idx + 1, target_value

    return idx, None


mod_imm = ComplexOperandMatcher(match_mod_imm)


def get_shift_opcode(opc):
    if opc == VirtualDagOps.SHL:
        return 2
    elif opc == VirtualDagOps.SRL:
        return 3
    elif opc == VirtualDagOps.SRA:
        return 1
    elif opc == VirtualDagOps.ROTR:
        return 4

    return 0


def match_imm_shifter_operand(node, values, idx, dag):
    from codegen.dag import VirtualDagOps
    from codegen.types import ValueType

    value = values[idx]

    opc = get_shift_opcode(value.node.opcode)
    if opc == 0:
        return idx, None

    rhs = value.node.operands[1]
    if rhs.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    rhs_val = rhs.node.value.value

    base = value.node.operands[0]
    opc = DagValue(dag.add_target_constant_node(
        value.ty, opc | (rhs_val << 3)), 0)

    return idx + 1, [base, opc]


so_reg_imm = ComplexOperandMatcher(match_imm_shifter_operand)


class ARMMachineOps:
    @classmethod
    def insts(cls):
        for member, value in cls.__dict__.items():
            if isinstance(value, MachineInstructionDef):
                yield value

    LDRi12 = def_inst(
        "ldr_i12",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)],
        patterns=[set_(("dst", GPR), load_(("src", addrmode_imm12)))]
    )

    STRi12 = def_inst(
        "str_i12",
        outs=[],
        ins=[("src", GPR), ("dst", AddrMode_Imm12)],
        patterns=[store_(("src", GPR), ("dst", addrmode_imm12))]
    )

    MOVsi = def_inst(
        "mov_si",
        outs=[("dst", GPR)],
        ins=[("src", SORegImm)],
        patterns=[set_(("dst", GPR), ("src", so_reg_imm))]
    )

    MOVi = def_inst(
        "mov_i",
        outs=[("dst", GPR)],
        ins=[("src", I32Imm)],
        patterns=[set_(("dst", GPR), ("src", mod_imm))]
    )

    MOVi16 = def_inst(
        "mov_i16",
        outs=[("dst", GPR)],
        ins=[("src", I32Imm)],
        patterns=[set_(("dst", GPR), ("src", imm0_65535))]
    )

    MOVTi16 = def_inst(
        "movt_i16",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("imm", I32Imm)],
        constraints=[Constraint("dst", "src")]
    )

    MOVr = def_inst(
        "mov_r",
        outs=[("dst", GPR)],
        ins=[("src", GPR)]
    )

    STR_PRE_IMM = def_inst(
        "str_pre_imm",
        outs=[("rn_web", GPR)],
        ins=[("rt", GPR), ("addr", AddrMode_Imm12)]
    )

    LDR_PRE_IMM = def_inst(
        "ldr_pre_imm",
        outs=[("rt", GPR), ("rn_web", GPR)],
        ins=[("addr", AddrMode_Imm12)]
    )

    STR_POST_IMM = def_inst(
        "str_post_imm",
        outs=[("rn_web", GPR)],
        ins=[("rt", GPR), ("addr", AddrMode_Imm12)]
    )

    LDR_POST_IMM = def_inst(
        "ldr_post_imm",
        outs=[("rt", GPR), ("rn_web", GPR)],
        ins=[("addr", AddrMode_Imm12)]
    )

    # load/store multiple and increment after
    LDMIA = def_inst(
        "ldm_ia",
        outs=[],
        ins=[("src", GPR), ("regs", reglist)]
    )

    STMIA = def_inst(
        "stm_ia",
        outs=[],
        ins=[("src", GPR), ("regs", reglist)]
    )

    LDMIA_UPD = def_inst(
        "ldm_ia_upd",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("regs", reglist)],
        constraints=[Constraint("dst", "src")]
    )

    STMIA_UPD = def_inst(
        "stm_ia_upd",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("regs", reglist)],
        constraints=[Constraint("dst", "src")]
    )

    # load/store multiple and decrement before
    LDMDB = def_inst(
        "ldm_db",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("regs", reglist)]
    )

    STMDB = def_inst(
        "stm_db",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("regs", reglist)]
    )

    LDMDB_UPD = def_inst(
        "ldm_db_upd",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("regs", reglist)],
        constraints=[Constraint("dst", "src")]
    )

    STMDB_UPD = def_inst(
        "stm_db_upd",
        outs=[("dst", GPR)],
        ins=[("src", GPR), ("regs", reglist)],
        constraints=[Constraint("dst", "src")]
    )

    ADDri = def_inst(
        "add_ri",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), add_(("src1", GPR), ("src2", mod_imm)))]
    )

    ADDrsi = def_inst(
        "add_rsi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), add_(
            ("src1", GPR), ("src2", so_reg_imm)))]
    )

    ADDrr = def_inst(
        "add_rr",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[set_(("dst", GPR), add_(("src1", GPR), ("src2", GPR)))]
    )

    SUBri = def_inst(
        "sub_ri",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), arm_sub_(
            ("src1", GPR), ("src2", mod_imm)))]
    )

    SUBrsi = def_inst(
        "sub_rsi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), arm_sub_(
            ("src1", GPR), ("src2", so_reg_imm)))]
    )

    SUBrr = def_inst(
        "sub_rr",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[set_(("dst", GPR), arm_sub_(("src1", GPR), ("src2", GPR)))]
    )

    ANDri = def_inst(
        "and_ri",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), and_(("src1", GPR), ("src2", mod_imm)))]
    )

    ANDrsi = def_inst(
        "and_rsi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), and_(
            ("src1", GPR), ("src2", so_reg_imm)))]
    )

    ANDrr = def_inst(
        "and_rr",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[set_(("dst", GPR), and_(("src1", GPR), ("src2", GPR)))]
    )

    ORri = def_inst(
        "or_ri",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), or_(("src1", GPR), ("src2", mod_imm)))]
    )

    ORrsi = def_inst(
        "or_rsi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), or_(("src1", GPR), ("src2", so_reg_imm)))]
    )

    ORrr = def_inst(
        "or_rr",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[set_(("dst", GPR), or_(("src1", GPR), ("src2", GPR)))]
    )

    XORri = def_inst(
        "xor_ri",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), xor_(("src1", GPR), ("src2", mod_imm)))]
    )

    XORrsi = def_inst(
        "xor_rsi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), xor_(
            ("src1", GPR), ("src2", so_reg_imm)))]
    )

    XORrr = def_inst(
        "xor_rr",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[set_(("dst", GPR), xor_(("src1", GPR), ("src2", GPR)))]
    )

    CMPri = def_inst(
        "cmp_ri",
        outs=[],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[arm_cmp_(("src1", GPR), ("src2", mod_imm))],
        is_terminator=True
    )

    CMPrsi = def_inst(
        "cmp_rsi",
        outs=[],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[arm_cmp_(("src1", GPR), ("src2", so_reg_imm))],
        is_terminator=True
    )

    CMPrr = def_inst(
        "cmp_rr",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[arm_cmp_(("src1", GPR), ("src2", GPR))],
        is_terminator=True
    )

    # vfp instructions
    VMOVS = def_inst(
        "vmovs",
        outs=[("dst", SPR)],
        ins=[("src", SPR)]
    )

    VMOVSR = def_inst(
        "vmovsr",
        outs=[("dst", SPR)],
        ins=[("src", GPR)],
        patterns=[set_(("dst", SPR), bitconvert_(("src", GPR)))]
    )

    VLDRS = def_inst(
        "vldrs",
        outs=[("dst", SPR)],
        ins=[("src", AddrMode5)],
        patterns=[set_(("dst", SPR), load_(("src", addrmode5)))]
    )

    VSTRS = def_inst(
        "vstrs",
        outs=[],
        ins=[("src", SPR), ("dst", AddrMode5)],
        patterns=[store_(("src", SPR), ("dst", addrmode5))]
    )

    VADDS = def_inst(
        "vadds",
        outs=[("dst", SPR)],
        ins=[("src1", SPR), ("src2", SPR)],
        patterns=[set_(("dst", SPR), fadd_(("src1", SPR), ("src2", SPR)))]
    )

    VSUBS = def_inst(
        "vsubs",
        outs=[("dst", SPR)],
        ins=[("src1", SPR), ("src2", SPR)],
        patterns=[set_(("dst", SPR), fsub_(("src1", SPR), ("src2", SPR)))]
    )

    VMULS = def_inst(
        "vmuls",
        outs=[("dst", SPR)],
        ins=[("src1", SPR), ("src2", SPR)],
        patterns=[set_(("dst", SPR), fmul_(("src1", SPR), ("src2", SPR)))]
    )

    VDIVS = def_inst(
        "vdivs",
        outs=[("dst", SPR)],
        ins=[("src1", SPR), ("src2", SPR)],
        patterns=[set_(("dst", SPR), fdiv_(("src1", SPR), ("src2", SPR)))]
    )

    VCMPS = def_inst(
        "vcmps",
        outs=[],
        ins=[("src1", SPR), ("src2", SPR)],
        patterns=[arm_cmpfp_(("src1", SPR), ("src2", SPR))]
    )

    VLDRD = def_inst(
        "vldrd",
        outs=[("dst", DPR)],
        ins=[("src", AddrMode5)],
        patterns=[set_(("dst", DPR), load_(("src", addrmode5)))]
    )

    VSTRD = def_inst(
        "vstrd",
        outs=[],
        ins=[("src", DPR), ("dst", AddrMode5)],
        patterns=[store_(("src", DPR), ("dst", addrmode5))]
    )

    FMSTAT = def_inst(
        "fmstat",
        outs=[],
        ins=[],
        patterns=[arm_fmstat_()]
    )

    # neon instructions
    VLD1q64 = def_inst(
        "vld1_q64",
        outs=[("dst", QPR)],
        ins=[("src", AddrMode6)],
        patterns=[set_(("dst", QPR), load_(("src", addrmode6)))]
    )
    VST1q64 = def_inst(
        "vst1_q64",
        outs=[],
        ins=[("src", QPR), ("dst", AddrMode6)],
        patterns=[store_(("src", QPR), ("dst", addrmode6))]
    )

    VADDfq = def_inst(
        "vadd_fq",
        outs=[("dst", QPR)],
        ins=[("src1", QPR), ("src2", QPR)],
        patterns=[set_(("dst", QPR), fadd_(("src1", QPR), ("src2", QPR)))]
    )

    VSUBfq = def_inst(
        "vsub_fq",
        outs=[("dst", QPR)],
        ins=[("src1", QPR), ("src2", QPR)],
        patterns=[set_(("dst", QPR), fsub_(("src1", QPR), ("src2", QPR)))]
    )

    VMULfq = def_inst(
        "vmul_fq",
        outs=[("dst", QPR)],
        ins=[("src1", QPR), ("src2", QPR)],
        patterns=[set_(("dst", QPR), fmul_(("src1", QPR), ("src2", QPR)))]
    )

    VORRq = def_inst(
        "vorr_q",
        outs=[("dst", QPR)],
        ins=[("src1", QPR), ("src2", QPR)],
        patterns=[set_(("dst", QPR), or_(("src1", QPR), ("src2", QPR)))]
    )

    # # jcc
    Bcc = def_inst(
        "bcc",
        outs=[],
        ins=[("dst", BrTarget8), ("cond", I32Imm)],
        patterns=[arm_brcond_(("dst", bb), ("cond", timm))],
        is_terminator=True
    )

    B = def_inst(
        "b",
        outs=[],
        ins=[("dst", BrTarget8)],
        patterns=[br_(("dst", bb))],
        is_terminator=True
    )

    BL = def_inst(
        "bl",
        outs=[],
        ins=[("dst", arm_bl_target)],
        is_call=True
    )

    MOVPCLR = def_inst(
        "movpclr",
        outs=[],
        ins=[],
        patterns=[(arm_let_())],
        is_terminator=True
    )

    # pseudo instructions
    ADJCALLSTACKDOWN = def_inst(
        "ADJCALLSTACKDOWN",
        outs=[],
        ins=[("amt1", I32Imm), ("amt2", I32Imm), ("amt3", I32Imm)]
    )

    ADJCALLSTACKUP = def_inst(
        "ADJCALLSTACKUP",
        outs=[],
        ins=[("amt1", I32Imm), ("amt2", I32Imm)]
    )

    MOVCCr = def_inst(
        "movcc_r",
        outs=[("dst", GPR)],
        ins=[("false", GPR), ("true", GPR), ("p", cmovpred)],
        constraints=[Constraint("dst", "false")],
        is_terminator=True
    )

    MOVCCi16 = def_inst(
        "movcc_i16",
        outs=[("dst", GPR)],
        ins=[("false", GPR), ("true", I32Imm), ("p", cmovpred)],
        constraints=[Constraint("dst", "false")],
        is_terminator=True
    )

    MOVCCi = def_inst(
        "movcc_i",
        outs=[("dst", GPR)],
        ins=[("false", GPR), ("true", I32Imm), ("p", cmovpred)],
        constraints=[Constraint("dst", "false")],
        is_terminator=True
    )

    VDUPLN32q = def_inst(
        "udup_ln32q",
        outs=[("dst", QPR)],
        ins=[("src", SPR), ("lane", VectorIndex32)]
    )

    LDRLIT_ga_abs = def_inst(
        "LDRLIT_ga_abs",
        outs=[("dst", GPR)],
        ins=[("src", I32Imm)],
        # patterns=[set_(("dst", GPR), arm_wrapper_(("src", tglobaladdr_)))]
    )

    PIC_ADD = def_inst(
        "pic_add",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR), arm_pic_add_(
            ("src1", GPR), ("src2", timm)))]
    )

    MOVi32imm = def_inst(
        "mov_i32imm",
        outs=[("dst", GPR)],
        ins=[("src", I32Imm)],
        patterns=[set_(("dst", GPR), arm_wrapper_(("src", tglobaladdr_)))]
    )

    LEApcrel = def_inst(
        "LEApcrel",
        outs=[("dst", GPR)],
        ins=[("label", I32Imm)],
        patterns=[set_(("dst", GPR), arm_wrapper_(("label", tconstpool_)))]
    )

    CPEntry = def_inst(
        "CPEntry",
        outs=[],
        ins=[("instid", I32Imm), ("cpidx", I32Imm), ("size", I32Imm)]
    )
