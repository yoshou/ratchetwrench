from codegen.spec import *
from codegen.mir_emitter import *
from codegen.matcher import *

sub_32 = def_subreg("sub_32", 32)

bsub = def_subreg("bsub", 8)
hsub = def_subreg("hsub", 16)
ssub = def_subreg("ssub", 32)
dsub = def_subreg("dsub", 64)


# integer registers
W0 = def_reg("w0", encoding=0)
W1 = def_reg("w1", encoding=1)
W2 = def_reg("w2", encoding=2)
W3 = def_reg("w3", encoding=3)
W4 = def_reg("w4", encoding=4)
W5 = def_reg("w5", encoding=5)
W6 = def_reg("w6", encoding=6)
W7 = def_reg("w7", encoding=7)
W8 = def_reg("w8", encoding=8)
W9 = def_reg("w9", encoding=9)
W10 = def_reg("w10", encoding=10)
W11 = def_reg("w11", encoding=11)
W12 = def_reg("w12", encoding=12)
W13 = def_reg("w13", encoding=13)
W14 = def_reg("w14", encoding=14)
W15 = def_reg("w15", encoding=15)
W16 = def_reg("w16", encoding=16)
W17 = def_reg("w17", encoding=17)
W18 = def_reg("w18", encoding=18)
W19 = def_reg("w19", encoding=19)
W20 = def_reg("w20", encoding=20)
W21 = def_reg("w21", encoding=21)
W22 = def_reg("w22", encoding=22)
W23 = def_reg("w23", encoding=23)
W24 = def_reg("w24", encoding=24)
W25 = def_reg("w25", encoding=25)
W26 = def_reg("w26", encoding=26)
W27 = def_reg("w27", encoding=27)
W28 = def_reg("w28", encoding=28)
W29 = def_reg("w29", encoding=29)
W30 = def_reg("w30", encoding=30)
WSP = def_reg("wsp", encoding=31)
WZR = def_reg("wzr", encoding=31)

X0 = def_reg("x0", [W0], [sub_32], encoding=0)
X1 = def_reg("x1", [W1], [sub_32], encoding=1)
X2 = def_reg("x2", [W2], [sub_32], encoding=2)
X3 = def_reg("x3", [W3], [sub_32], encoding=3)
X4 = def_reg("x4", [W4], [sub_32], encoding=4)
X5 = def_reg("x5", [W5], [sub_32], encoding=5)
X6 = def_reg("x6", [W6], [sub_32], encoding=6)
X7 = def_reg("x7", [W7], [sub_32], encoding=7)
X8 = def_reg("x8", [W8], [sub_32], encoding=8)
X9 = def_reg("x9", [W9], [sub_32], encoding=9)
X10 = def_reg("x10", [W10], [sub_32], encoding=10)
X11 = def_reg("x11", [W11], [sub_32], encoding=11)
X12 = def_reg("x12", [W12], [sub_32], encoding=12)
X13 = def_reg("x13", [W13], [sub_32], encoding=13)
X14 = def_reg("x14", [W14], [sub_32], encoding=14)
X15 = def_reg("x15", [W15], [sub_32], encoding=15)
X16 = def_reg("x16", [W16], [sub_32], encoding=16)
X17 = def_reg("x17", [W17], [sub_32], encoding=17)
X18 = def_reg("x18", [W18], [sub_32], encoding=18)
X19 = def_reg("x19", [W19], [sub_32], encoding=19)
X20 = def_reg("x20", [W20], [sub_32], encoding=20)
X21 = def_reg("x21", [W21], [sub_32], encoding=21)
X22 = def_reg("x22", [W22], [sub_32], encoding=22)
X23 = def_reg("x23", [W23], [sub_32], encoding=23)
X24 = def_reg("x24", [W24], [sub_32], encoding=24)
X25 = def_reg("x25", [W25], [sub_32], encoding=25)
X26 = def_reg("x26", [W26], [sub_32], encoding=26)
X27 = def_reg("x27", [W27], [sub_32], encoding=27)
X28 = def_reg("x28", [W28], [sub_32], encoding=28)
FP = def_reg("x29", [W29], [sub_32], encoding=29)
LR = def_reg("x30", [W30], [sub_32], encoding=30)
SP = def_reg("sp", [WSP], [sub_32], encoding=31)
XZR = def_reg("xzr", [WZR], [sub_32], encoding=31)

PC = def_reg("pc", encoding=15)

# floating-point registers
B0 = def_reg("b0", encoding=0)
B1 = def_reg("b1", encoding=1)
B2 = def_reg("b2", encoding=2)
B3 = def_reg("b3", encoding=3)
B4 = def_reg("b4", encoding=4)
B5 = def_reg("b5", encoding=5)
B6 = def_reg("b6", encoding=6)
B7 = def_reg("b7", encoding=7)
B8 = def_reg("b8", encoding=8)
B9 = def_reg("b9", encoding=9)
B10 = def_reg("b10", encoding=10)
B11 = def_reg("b11", encoding=11)
B12 = def_reg("b12", encoding=12)
B13 = def_reg("b13", encoding=13)
B14 = def_reg("b14", encoding=14)
B15 = def_reg("b15", encoding=15)
B16 = def_reg("b16", encoding=16)
B17 = def_reg("b17", encoding=17)
B18 = def_reg("b18", encoding=18)
B19 = def_reg("b19", encoding=19)
B20 = def_reg("b20", encoding=20)
B21 = def_reg("b21", encoding=21)
B22 = def_reg("b22", encoding=22)
B23 = def_reg("b23", encoding=23)
B24 = def_reg("b24", encoding=24)
B25 = def_reg("b25", encoding=25)
B26 = def_reg("b26", encoding=26)
B27 = def_reg("b27", encoding=27)
B28 = def_reg("b28", encoding=28)
B29 = def_reg("b29", encoding=29)
B30 = def_reg("b30", encoding=30)
B31 = def_reg("b31", encoding=31)

H0 = def_reg("h0", [B0], [bsub], encoding=0)
H1 = def_reg("h1", [B1], [bsub], encoding=1)
H2 = def_reg("h2", [B2], [bsub], encoding=2)
H3 = def_reg("h3", [B3], [bsub], encoding=3)
H4 = def_reg("h4", [B4], [bsub], encoding=4)
H5 = def_reg("h5", [B5], [bsub], encoding=5)
H6 = def_reg("h6", [B6], [bsub], encoding=6)
H7 = def_reg("h7", [B7], [bsub], encoding=7)
H8 = def_reg("h8", [B8], [bsub], encoding=8)
H9 = def_reg("h9", [B9], [bsub], encoding=9)
H10 = def_reg("h10", [B10], [bsub], encoding=10)
H11 = def_reg("h11", [B11], [bsub], encoding=11)
H12 = def_reg("h12", [B12], [bsub], encoding=12)
H13 = def_reg("h13", [B13], [bsub], encoding=13)
H14 = def_reg("h14", [B14], [bsub], encoding=14)
H15 = def_reg("h15", [B15], [bsub], encoding=15)
H16 = def_reg("h16", [B16], [bsub], encoding=16)
H17 = def_reg("h17", [B17], [bsub], encoding=17)
H18 = def_reg("h18", [B18], [bsub], encoding=18)
H19 = def_reg("h19", [B19], [bsub], encoding=19)
H20 = def_reg("h20", [B20], [bsub], encoding=20)
H21 = def_reg("h21", [B21], [bsub], encoding=21)
H22 = def_reg("h22", [B22], [bsub], encoding=22)
H23 = def_reg("h23", [B23], [bsub], encoding=23)
H24 = def_reg("h24", [B24], [bsub], encoding=24)
H25 = def_reg("h25", [B25], [bsub], encoding=25)
H26 = def_reg("h26", [B26], [bsub], encoding=26)
H27 = def_reg("h27", [B27], [bsub], encoding=27)
H28 = def_reg("h28", [B28], [bsub], encoding=28)
H29 = def_reg("h29", [B29], [bsub], encoding=29)
H30 = def_reg("h30", [B30], [bsub], encoding=30)
H31 = def_reg("h31", [B31], [bsub], encoding=31)

S0 = def_reg("s0", [H0], [hsub], encoding=0)
S1 = def_reg("s1", [H1], [hsub], encoding=1)
S2 = def_reg("s2", [H2], [hsub], encoding=2)
S3 = def_reg("s3", [H3], [hsub], encoding=3)
S4 = def_reg("s4", [H4], [hsub], encoding=4)
S5 = def_reg("s5", [H5], [hsub], encoding=5)
S6 = def_reg("s6", [H6], [hsub], encoding=6)
S7 = def_reg("s7", [H7], [hsub], encoding=7)
S8 = def_reg("s8", [H8], [hsub], encoding=8)
S9 = def_reg("s9", [H9], [hsub], encoding=9)
S10 = def_reg("s10", [H10], [hsub], encoding=10)
S11 = def_reg("s11", [H11], [hsub], encoding=11)
S12 = def_reg("s12", [H12], [hsub], encoding=12)
S13 = def_reg("s13", [H13], [hsub], encoding=13)
S14 = def_reg("s14", [H14], [hsub], encoding=14)
S15 = def_reg("s15", [H15], [hsub], encoding=15)
S16 = def_reg("s16", [H16], [hsub], encoding=16)
S17 = def_reg("s17", [H17], [hsub], encoding=17)
S18 = def_reg("s18", [H18], [hsub], encoding=18)
S19 = def_reg("s19", [H19], [hsub], encoding=19)
S20 = def_reg("s20", [H20], [hsub], encoding=20)
S21 = def_reg("s21", [H21], [hsub], encoding=21)
S22 = def_reg("s22", [H22], [hsub], encoding=22)
S23 = def_reg("s23", [H23], [hsub], encoding=23)
S24 = def_reg("s24", [H24], [hsub], encoding=24)
S25 = def_reg("s25", [H25], [hsub], encoding=25)
S26 = def_reg("s26", [H26], [hsub], encoding=26)
S27 = def_reg("s27", [H27], [hsub], encoding=27)
S28 = def_reg("s28", [H28], [hsub], encoding=28)
S29 = def_reg("s29", [H29], [hsub], encoding=29)
S30 = def_reg("s30", [H30], [hsub], encoding=30)
S31 = def_reg("s31", [H31], [hsub], encoding=31)

D0 = def_reg("d0", [S0], [ssub], encoding=0)
D1 = def_reg("d1", [S1], [ssub], encoding=1)
D2 = def_reg("d2", [S2], [ssub], encoding=2)
D3 = def_reg("d3", [S3], [ssub], encoding=3)
D4 = def_reg("d4", [S4], [ssub], encoding=4)
D5 = def_reg("d5", [S5], [ssub], encoding=5)
D6 = def_reg("d6", [S6], [ssub], encoding=6)
D7 = def_reg("d7", [S7], [ssub], encoding=7)
D8 = def_reg("d8", [S8], [ssub], encoding=8)
D9 = def_reg("d9", [S9], [ssub], encoding=9)
D10 = def_reg("d10", [S10], [ssub], encoding=10)
D11 = def_reg("d11", [S11], [ssub], encoding=11)
D12 = def_reg("d12", [S12], [ssub], encoding=12)
D13 = def_reg("d13", [S13], [ssub], encoding=13)
D14 = def_reg("d14", [S14], [ssub], encoding=14)
D15 = def_reg("d15", [S15], [ssub], encoding=15)
D16 = def_reg("d16", [S16], [ssub], encoding=16)
D17 = def_reg("d17", [S17], [ssub], encoding=17)
D18 = def_reg("d18", [S18], [ssub], encoding=18)
D19 = def_reg("d19", [S19], [ssub], encoding=19)
D20 = def_reg("d20", [S20], [ssub], encoding=20)
D21 = def_reg("d21", [S21], [ssub], encoding=21)
D22 = def_reg("d22", [S22], [ssub], encoding=22)
D23 = def_reg("d23", [S23], [ssub], encoding=23)
D24 = def_reg("d24", [S24], [ssub], encoding=24)
D25 = def_reg("d25", [S25], [ssub], encoding=25)
D26 = def_reg("d26", [S26], [ssub], encoding=26)
D27 = def_reg("d27", [S27], [ssub], encoding=27)
D28 = def_reg("d28", [S28], [ssub], encoding=28)
D29 = def_reg("d29", [S29], [ssub], encoding=29)
D30 = def_reg("d30", [S30], [ssub], encoding=30)
D31 = def_reg("d31", [S31], [ssub], encoding=31)

Q0 = def_reg("q0", [D0], [dsub], encoding=0)
Q1 = def_reg("q1", [D1], [dsub], encoding=1)
Q2 = def_reg("q2", [D2], [dsub], encoding=2)
Q3 = def_reg("q3", [D3], [dsub], encoding=3)
Q4 = def_reg("q4", [D4], [dsub], encoding=4)
Q5 = def_reg("q5", [D5], [dsub], encoding=5)
Q6 = def_reg("q6", [D6], [dsub], encoding=6)
Q7 = def_reg("q7", [D7], [dsub], encoding=7)
Q8 = def_reg("q8", [D8], [dsub], encoding=8)
Q9 = def_reg("q9", [D9], [dsub], encoding=9)
Q10 = def_reg("q10", [D10], [dsub], encoding=10)
Q11 = def_reg("q11", [D11], [dsub], encoding=11)
Q12 = def_reg("q12", [D12], [dsub], encoding=12)
Q13 = def_reg("q13", [D13], [dsub], encoding=13)
Q14 = def_reg("q14", [D14], [dsub], encoding=14)
Q15 = def_reg("q15", [D15], [dsub], encoding=15)
Q16 = def_reg("q16", [D16], [dsub], encoding=16)
Q17 = def_reg("q17", [D17], [dsub], encoding=17)
Q18 = def_reg("q18", [D18], [dsub], encoding=18)
Q19 = def_reg("q19", [D19], [dsub], encoding=19)
Q20 = def_reg("q20", [D20], [dsub], encoding=20)
Q21 = def_reg("q21", [D21], [dsub], encoding=21)
Q22 = def_reg("q22", [D22], [dsub], encoding=22)
Q23 = def_reg("q23", [D23], [dsub], encoding=23)
Q24 = def_reg("q24", [D24], [dsub], encoding=24)
Q25 = def_reg("q25", [D25], [dsub], encoding=25)
Q26 = def_reg("q26", [D26], [dsub], encoding=26)
Q27 = def_reg("q27", [D27], [dsub], encoding=27)
Q28 = def_reg("q28", [D28], [dsub], encoding=28)
Q29 = def_reg("q29", [D29], [dsub], encoding=29)
Q30 = def_reg("q30", [D30], [dsub], encoding=30)
Q31 = def_reg("q31", [D31], [dsub], encoding=31)

NZCV = def_reg("cpsr", encoding=0)


aarch64_regclasses = []


def def_aarch64_regclass(*args, **kwargs):
    regclass = def_regclass(*args, **kwargs)
    aarch64_regclasses.append(regclass)
    return regclass


def sequence(format_str, start, end):
    seq = [globals()[format_str.format(i)]
           for i in range(start, (end - start))]
    return seq


GPR32 = def_aarch64_regclass(
    "GPR32", [ValueType.I32], 32, sequence("W{0}", 0, 30))


GPR32sp = def_aarch64_regclass(
    "GPR32sp", [ValueType.I32], 32, sequence("W{0}", 0, 30) + [SP])

GPR64 = def_aarch64_regclass(
    "GPR64", [ValueType.I64], 64, sequence("X{0}", 0, 28) + [FP, LR])

GPR64sp = def_aarch64_regclass(
    "GPR64sp", [ValueType.I64], 64, sequence("X{0}", 0, 28) + [FP, LR, SP])

FPR16 = def_aarch64_regclass(
    "SPR", [ValueType.F16], 16, sequence("H{0}", 0, 31))

FPR32 = def_aarch64_regclass(
    "SPR", [ValueType.F32], 32, sequence("S{0}", 0, 31))

FPR64 = def_aarch64_regclass(
    "DPR", [ValueType.F64, ValueType.V2F32], 64, sequence("D{0}", 0, 31))

FPR128 = def_aarch64_regclass(
    "QPR", [ValueType.V4F32, ValueType.V2F64], 128, sequence("Q{0}", 0, 31))

AArch64 = MachineHWMode("aarch64")

# build register graph
reg_graph = compute_reg_graph()
reg_groups = compute_reg_groups(reg_graph)
compute_reg_subregs_all(reg_graph)
compute_reg_superregs_all(reg_graph)

# infer regclass
for regclass in aarch64_regclasses:
    infer_subregclass_and_subreg(regclass)

I8Imm = ValueOperandDef(ValueType.I8)
I16Imm = ValueOperandDef(ValueType.I16)
I32Imm = ValueOperandDef(ValueType.I32)
I64Imm = ValueOperandDef(ValueType.I64)
AdrLabel = ValueOperandDef(ValueType.I64)

F32Imm = ValueOperandDef(ValueType.F32)
F64Imm = ValueOperandDef(ValueType.F64)

BrTarget8 = ValueOperandDef(ValueType.I8)
BrTarget16 = ValueOperandDef(ValueType.I16)
BrTarget32 = ValueOperandDef(ValueType.I32)
VectorIndex32 = ValueOperandDef(ValueType.I32)


class AArch64MemOperandDef(ValueOperandDef):
    def __init__(self, op_info):
        super().__init__(ValueType.I32)

        self.op_info = op_info

    @property
    def operand_info(self):
        return self.op_info


class AArch64DagOp(DagOp):
    def __init__(self, name):
        super().__init__(name, "aarch64_")


class AArch64DagOps(Enum):
    SUB = AArch64DagOp("sub")
    CMP = AArch64DagOp("cmp")
    SETCC = AArch64DagOp("setcc")
    BRCOND = AArch64DagOp("brcond")

    VDUP = AArch64DagOp("vdup")

    SHUFP = AArch64DagOp("shufp")
    UNPCKL = AArch64DagOp("unpckl")
    UNPCKH = AArch64DagOp("unpckh")

    MOVSS = AArch64DagOp("movss")
    MOVSD = AArch64DagOp("movsd")

    CMPFP = AArch64DagOp("cmpfp")
    FMSTAT = AArch64DagOp("fmstat")

    CALL = AArch64DagOp("call")
    RETURN = AArch64DagOp("return")

    ADRP = AArch64DagOp("adrp")

    PIC_ADD = AArch64DagOp("pic_add")


aarch64_brcond_ = NodePatternMatcherGen(AArch64DagOps.BRCOND)
aarch64_sub_ = NodePatternMatcherGen(AArch64DagOps.SUB)
aarch64_cmp_ = NodePatternMatcherGen(AArch64DagOps.CMP)
aarch64_cmpfp_ = NodePatternMatcherGen(AArch64DagOps.CMPFP)
aarch64_fmstat_ = NodePatternMatcherGen(AArch64DagOps.FMSTAT)
aarch64_setcc_ = NodePatternMatcherGen(AArch64DagOps.SETCC)
aarch64_movss_ = NodePatternMatcherGen(AArch64DagOps.MOVSS)
aarch64_shufp_ = NodePatternMatcherGen(AArch64DagOps.SHUFP)
aarch64_let_ = NodePatternMatcherGen(AArch64DagOps.RETURN)
aarch64_adrp_ = NodePatternMatcherGen(AArch64DagOps.ADRP)
aarch64_pic_add_ = NodePatternMatcherGen(AArch64DagOps.PIC_ADD)


def in_bits_signed(value, bits):
    mx = (1 << (bits - 1)) - 1
    mn = -mx - 1
    return value >= mn and value <= mx


def in_bits_unsigned(value, bits):
    mx = (1 << (bits)) - 1
    return value >= 0 and value <= mx

# addrmode imm12 'reg +/- imm12'


def match_addrmode_indexed(node, values, idx, dag, size):
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

    # only base.
    base = value

    assert(base.node.opcode != VirtualDagOps.TARGET_CONSTANT)

    offset = DagValue(dag.add_target_constant_node(
        value.ty, 0), 0)

    return idx + 1, [base, offset]


def match_addrmode_indexed8(node, values, idx, dag):
    return match_addrmode_indexed(node, values, idx, dag, 1)


def match_addrmode_indexed16(node, values, idx, dag):
    return match_addrmode_indexed(node, values, idx, dag, 2)


def match_addrmode_indexed32(node, values, idx, dag):
    return match_addrmode_indexed(node, values, idx, dag, 4)


def match_addrmode_indexed64(node, values, idx, dag):
    return match_addrmode_indexed(node, values, idx, dag, 8)


def match_addrmode_indexed128(node, values, idx, dag):
    return match_addrmode_indexed(node, values, idx, dag, 16)


am_indexed8 = ComplexPatternMatcher(i64_, 2, match_addrmode_indexed8, [])
am_indexed16 = ComplexPatternMatcher(i64_, 2, match_addrmode_indexed16, [])
am_indexed32 = ComplexPatternMatcher(i64_, 2, match_addrmode_indexed32, [])
am_indexed64 = ComplexPatternMatcher(i64_, 2, match_addrmode_indexed64, [])
am_indexed128 = ComplexPatternMatcher(i64_, 2, match_addrmode_indexed128, [])


class AArch64MachineOps:
    @classmethod
    def insts(cls):
        for member, value in cls.__dict__.items():
            if isinstance(value, MachineInstructionDef):
                yield value

    LDRWui = def_inst(
        "ldrw_ui",
        outs=[("rt", GPR32)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR32), load_(
            am_indexed32(("rn", GPR64sp), ("offset", imm))))]
    )

    LDRXui = def_inst(
        "ldrx_ui",
        outs=[("dst", GPR64)],
        ins=[("src", GPR64sp)],
        patterns=[set_(("dst", GPR64), load_(
            am_indexed32(("rn", GPR64sp), ("offset", imm))))]
    )

    STRWui = def_inst(
        "strw_ui",
        outs=[],
        ins=[("rt", GPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", GPR32), am_indexed32(
            ("rn", GPR64sp), ("offset", imm)))]
    )

    STRXui = def_inst(
        "strx_ui",
        outs=[],
        ins=[("rt", GPR64), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", GPR64), am_indexed32(
            ("rn", GPR64sp), ("offset", imm)))]
    )

    ADDXri = def_inst(
        "addx_ri",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR32), add_(("src1", GPR32), ("src2", imm)))]
    )

    SUBSWrr = def_inst(
        "subsw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[
            set_(("dst", GPR32), sub_(("src1", GPR32), ("src2", GPR32))),
            aarch64_cmp_(("src1", GPR32), ("src2", GPR32))
        ]
    )

    ANDWrr = def_inst(
        "andw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), and_(("src1", GPR32), ("src2", GPR32)))]
    )

    EORWrr = def_inst(
        "eorw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), xor_(("src1", GPR32), ("src2", GPR32)))]
    )

    ORRWrr = def_inst(
        "orrw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), or_(("src1", GPR32), ("src2", GPR32)))]
    )

    UBFMWri = def_inst(
        "ubfmw_ri",
        outs=[("rd", GPR32)],
        ins=[("rn", GPR32), ("immr", I32Imm), ("imms", I32Imm)]
    )

    MOVi32imm = def_inst(
        "mov_i32imm",
        outs=[("dst", GPR32)],
        ins=[("src", I32Imm)],
        patterns=[set_(("dst", GPR32), ("src", imm))]
    )

    MOVi64imm = def_inst(
        "mov_i64imm",
        outs=[("dst", GPR64)],
        ins=[("src", I64Imm)],
        patterns=[set_(("dst", GPR64), ("src", imm))]
    )

    ADRP = def_inst(
        "auipc",
        outs=[("dst", GPR64)],
        ins=[("src", AdrLabel)],
        patterns=[set_(("dst", GPR64), aarch64_adrp_(("src", tglobaladdr_)))]
    )

    Bcc = def_inst(
        "bcc",
        outs=[],
        ins=[("dst", BrTarget8), ("cond", I32Imm)],
        patterns=[aarch64_brcond_(("dst", bb), ("cond", timm))],
        is_terminator=True
    )

    B = def_inst(
        "b",
        outs=[],
        ins=[("addr", BrTarget8)],
        patterns=[br_(("addr", bb))],
        is_terminator=True
    )

