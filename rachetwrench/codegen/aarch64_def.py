from rachetwrench.codegen.spec import *
from rachetwrench.codegen.mir_emitter import *
from rachetwrench.codegen.matcher import *

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

NZCV = def_reg("nzcv", encoding=0)


aarch64_regclasses = []


def def_aarch64_regclass(*args, **kwargs):
    regclass = def_regclass(*args, **kwargs)
    aarch64_regclasses.append(regclass)
    return regclass


def sequence(format_str, start, end):
    seq = [globals()[format_str.format(i)]
           for i in range(start, (end - start + 1))]
    return seq


GPR32 = def_aarch64_regclass(
    "GPR32", [ValueType.I32], 32, sequence("W{0}", 0, 30))

GPR32sp = def_aarch64_regclass(
    "GPR32sp", [ValueType.I32], 32, sequence("W{0}", 0, 30) + [SP])

GPR64 = def_aarch64_regclass(
    "GPR64", [ValueType.I64], 64, sequence("X{0}", 0, 28) + [FP, LR])

GPR64sp = def_aarch64_regclass(
    "GPR64sp", [ValueType.I64], 64, sequence("X{0}", 0, 28) + [FP, LR, SP])

FPR8 = def_aarch64_regclass(
    "FPR8", [ValueType.I8], 8, sequence("B{0}", 0, 31))

FPR16 = def_aarch64_regclass(
    "FPR16", [ValueType.F16], 16, sequence("H{0}", 0, 31))

FPR32 = def_aarch64_regclass(
    "FPR32", [ValueType.F32], 32, sequence("S{0}", 0, 31))

FPR64 = def_aarch64_regclass(
    "FPR64", [ValueType.F64, ValueType.V2F32], 64, sequence("D{0}", 0, 31))

FPR128 = def_aarch64_regclass(
    "FPR128", [ValueType.V16I8, ValueType.V8I16, ValueType.V4I32, ValueType.V4F32, ValueType.V2F64, ValueType.F128], 128, sequence("Q{0}", 0, 31))

AArch64 = MachineHWMode("aarch64")

# build register graph
reg_graph = compute_reg_graph()
reg_groups = compute_reg_groups(reg_graph)
compute_reg_subregs_all(reg_graph)
compute_reg_superregs_all(reg_graph)

# infer regclass
for regclass in aarch64_regclasses:
    infer_subregclass_and_subreg(regclass)

# system registers


class AArch64SysRefDef:
    def __init__(self, op0, op1, crn, crm, op2):
        self.op0 = op0
        self.op1 = op1
        self.crn = crn
        self.crm = crm
        self.op2 = op2


class AArch64SysReg(Enum):
    TPIDR_EL0 = AArch64SysRefDef(0b11, 0b011, 0b1101, 0b0000, 0b010)


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
mrs_sysreg_op = ValueOperandDef(ValueType.I32)


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
    ADDS = AArch64DagOp("adds")
    SUBS = AArch64DagOp("subs")
    SETCC = AArch64DagOp("setcc")
    BRCOND = AArch64DagOp("brcond")

    DUP = AArch64DagOp("DUP")

    SHUFP = AArch64DagOp("shufp")
    UNPCKL = AArch64DagOp("unpckl")
    UNPCKH = AArch64DagOp("unpckh")

    MOVSS = AArch64DagOp("movss")
    MOVSD = AArch64DagOp("movsd")

    CMPFP = AArch64DagOp("cmpfp")
    CSINV = AArch64DagOp("csinv")
    CSNEG = AArch64DagOp("csneg")
    CSINC = AArch64DagOp("csinc")
    CSEL = AArch64DagOp("csel")

    CALL = AArch64DagOp("call")
    RETURN = AArch64DagOp("return")

    ADR = AArch64DagOp("adr")
    ADRP = AArch64DagOp("adrp")
    ADDlow = AArch64DagOp("addlow")

    TLSDESC_CALLSEQ = AArch64DagOp("tlsdesc_callseq")
    THREAD_POINTER = AArch64DagOp("thread_pointer")


aarch64_brcond_ = NodePatternMatcherGen(AArch64DagOps.BRCOND)
aarch64_adds_ = NodePatternMatcherGen(AArch64DagOps.ADDS)
aarch64_subs_ = NodePatternMatcherGen(AArch64DagOps.SUBS)
aarch64_fcmp_ = NodePatternMatcherGen(AArch64DagOps.CMPFP)
aarch64_csel_ = NodePatternMatcherGen(AArch64DagOps.CSEL)
aarch64_csinc_ = NodePatternMatcherGen(AArch64DagOps.CSINC)
# aarch64_setcc_ = NodePatternMatcherGen(AArch64DagOps.SETCC)
# aarch64_movss_ = NodePatternMatcherGen(AArch64DagOps.MOVSS)
# aarch64_shufp_ = NodePatternMatcherGen(AArch64DagOps.SHUFP)
aarch64_let_ = NodePatternMatcherGen(AArch64DagOps.RETURN)
aarch64_adr_ = NodePatternMatcherGen(AArch64DagOps.ADR)
aarch64_adrp_ = NodePatternMatcherGen(AArch64DagOps.ADRP)
aarch64_addlow_ = NodePatternMatcherGen(AArch64DagOps.ADDlow)
aarch64_tlsdesc_callseq_ = NodePatternMatcherGen(AArch64DagOps.TLSDESC_CALLSEQ)
aarch64_thread_pointer_ = NodePatternMatcherGen(AArch64DagOps.THREAD_POINTER)


def in_bits_signed(value, bits):
    mx = (1 << (bits - 1)) - 1
    mn = -mx - 1
    return value >= mn and value <= mx


def in_bits_unsigned(value, bits):
    mx = (1 << (bits)) - 1
    return value >= 0 and value <= mx


def match_addrmode_indexed(node, values, idx, dag, size):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    value = values[idx]

    if value.node.opcode == VirtualDagOps.ADD:
        value1 = value.node.operands[0]
        value2 = value.node.operands[1]

        if value1.node.opcode == VirtualDagOps.CONSTANT:
            offset_val = value1.node.value

            if offset_val.value % size == 0:
                base = value2
                offset = DagValue(dag.add_target_constant_node(
                    value1.ty, ConstantInt(int(offset_val.value / size), offset_val.ty)), 0)

                return idx + 1, [base, offset]
        elif value2.node.opcode == VirtualDagOps.CONSTANT:
            offset_val = value2.node.value

            if offset_val.value % size == 0:
                base = value1
                offset = DagValue(dag.add_target_constant_node(
                    value2.ty, ConstantInt(int(offset_val.value / size), offset_val.ty)), 0)

                return idx + 1, [base, offset]
    elif value.node.opcode == VirtualDagOps.FRAME_INDEX:
        base = DagValue(dag.add_frame_index_node(
            value.ty, value.node.index, True), 0)

        offset = DagValue(dag.add_target_constant_node(
            value.ty, 0), 0)

        return idx + 1, [base, offset]

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


def match_addrmode_unscaled(node, values, idx, dag, size):
    from rachetwrench.codegen.dag import VirtualDagOps, DagValue
    from rachetwrench.codegen.types import ValueType

    value = values[idx]

    # if value.node.opcode == VirtualDagOps.ADD:
    #     value1 = value.node.operands[0]
    #     value2 = value.node.operands[1]

    #     if value1.node.opcode == VirtualDagOps.CONSTANT and value1.node.value.value > 0:
    #         base = value2
    #         offset = DagValue(dag.add_target_constant_node(
    #             value1.ty, value1.node.value), 0)

    #         return idx + 1, [base, offset]
    #     elif value2.node.opcode == VirtualDagOps.CONSTANT and value2.node.value.value > 0:
    #         base = value1
    #         offset = DagValue(dag.add_target_constant_node(
    #             value2.ty, value2.node.value), 0)

    #         return idx + 1, [base, offset]
    # elif value.node.opcode == VirtualDagOps.FRAME_INDEX:
    #     base = DagValue(dag.add_frame_index_node(
    #         value.ty, value.node.index, True), 0)

    #     offset = DagValue(dag.add_target_constant_node(
    #         value.ty, 0), 0)

    #     return idx + 1, [base, offset]

    # only base.
    base = value

    assert(base.node.opcode != VirtualDagOps.TARGET_CONSTANT)

    offset = DagValue(dag.add_target_constant_node(
        value.ty, 0), 0)

    return idx + 1, [base, offset]


def match_addrmode_unscaled8(node, values, idx, dag):
    return match_addrmode_unscaled(node, values, idx, dag, 1)


def match_addrmode_unscaled16(node, values, idx, dag):
    return match_addrmode_unscaled(node, values, idx, dag, 2)


def match_addrmode_unscaled32(node, values, idx, dag):
    return match_addrmode_unscaled(node, values, idx, dag, 4)


def match_addrmode_unscaled64(node, values, idx, dag):
    return match_addrmode_unscaled(node, values, idx, dag, 8)


def match_addrmode_unscaled128(node, values, idx, dag):
    return match_addrmode_unscaled(node, values, idx, dag, 16)


am_unscaled8 = ComplexPatternMatcher(i64_, 2, match_addrmode_unscaled8, [])
am_unscaled16 = ComplexPatternMatcher(i64_, 2, match_addrmode_unscaled16, [])
am_unscaled32 = ComplexPatternMatcher(i64_, 2, match_addrmode_unscaled32, [])
am_unscaled64 = ComplexPatternMatcher(i64_, 2, match_addrmode_unscaled64, [])
am_unscaled128 = ComplexPatternMatcher(i64_, 2, match_addrmode_unscaled128, [])


def match_imm0_31(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps
    from rachetwrench.codegen.types import ValueType

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    constant = value.node.value.value
    if constant >= 0 and constant < 32:
        target_value = DagValue(dag.add_target_constant_node(
            value.ty, constant), 0)

        return idx + 1, target_value

    return idx, None


imm0_31 = ComplexOperandMatcher(match_imm0_31)


def match_imm0_63(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps
    from rachetwrench.codegen.types import ValueType

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    constant = value.node.value.value
    if constant >= 0 and constant < 64:
        target_value = DagValue(dag.add_target_constant_node(
            value.ty, constant), 0)

        return idx + 1, target_value

    return idx, None


imm0_63 = ComplexOperandMatcher(match_imm0_63)


def is_power_of_2(value):
    return value != 0 and ((value - 1) & value) == 0

# Return true if value has all one bit


def is_mask(value):
    return value != 0 and ((value + 1) & value) == 0


def is_shifted_mask(value):
    return value != 0 and is_mask((value - 1) | value)


def count_trailing_zeros(value):
    if value == 0:
        return 0

    cnt = 0
    while (value & 0x1) == 0:
        value = value >> 1
        cnt += 1
    return cnt


def count_trailing_ones(value):
    cnt = 0
    while (value & 0x1) == 1:
        value = value >> 1
        cnt += 1
    return cnt


def is_logical_imm(value, bits):
    if value == 0:
        return False

    size = bits

    while True:
        size = size >> 1
        mask = (1 << size) - 1

        if (value & mask) != ((value >> size) & mask):
            size = size << 1
            break

        if size <= 2:
            break

    mask = ((1 << 64) - 1) >> (64 - size)
    value = value & mask

    if is_shifted_mask(value):
        ctz = count_trailing_zeros(value)
        assert(ctz < 64)
        cto = count_trailing_ones(value)
    else:
        return False
        value = value | (mask ^ ((1 << 64) - 1))

    immr = (size - ctz) & (size - 1)

    nimms = ((size - 1) ^ ((1 << 32) - 1)) << 1

    nimms |= (cto - 1)

    n = ((nimms >> 6) & 1) ^ 1

    encoding = (n << 12) | (immr << 6) | (nimms & 0x3f)

    return True


def match_logical_imm32(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps
    from rachetwrench.codegen.types import ValueType

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
        return idx, None

    constant = value.node.value.value
    if is_logical_imm(constant, 32):
        target_value = DagValue(dag.add_target_constant_node(
            value.ty, constant), 0)

        return idx + 1, target_value

    return idx, None


logical_imm32 = ComplexOperandMatcher(match_logical_imm32)


def is_uimm12_offset(value, scale):
    return value % scale == 0 and value >= 0 and value / scale < 0x1000


def match_uimm12_scaled(scale):
    def match_func(node, values, idx, dag):
        from rachetwrench.codegen.dag import VirtualDagOps
        from rachetwrench.codegen.types import ValueType

        value = values[idx]
        if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            return idx, None

        constant = value.node.value.value
        if is_uimm12_offset(constant, scale):
            target_value = DagValue(dag.add_target_constant_node(
                value.ty, constant), 0)

            return idx + 1, target_value

        return idx, None

    return match_func


uimm12s1 = ComplexOperandMatcher(match_uimm12_scaled(1))
uimm12s2 = ComplexOperandMatcher(match_uimm12_scaled(2))
uimm12s4 = ComplexOperandMatcher(match_uimm12_scaled(4))
uimm12s8 = ComplexOperandMatcher(match_uimm12_scaled(8))
uimm12s16 = ComplexOperandMatcher(match_uimm12_scaled(16))


def is_imm_scaled(value, width, scale, signed):
    if signed:
        min_val = -(1 << (width - 1)) * scale
        max_val = ((1 << (width - 1)) - 1) * scale
    else:
        min_val = 0
        max_val = ((1 << width) - 1) * scale

    return value % scale == 0 and value >= min_val and value <= max_val


def match_simm_scaled_memory_indexed(width, scale):
    def match_func(node, values, idx, dag):
        from rachetwrench.codegen.dag import VirtualDagOps
        from rachetwrench.codegen.types import ValueType

        value = values[idx]
        if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            return idx, None

        constant = value.node.value.value
        if is_imm_scaled(constant, width, scale, True):
            target_value = DagValue(dag.add_target_constant_node(
                value.ty, constant), 0)

            return idx + 1, target_value

        return idx, None

    return match_func


def is_simm_scaled(value, width, scale):
    return is_imm_scaled(value, width, scale, True)


def is_simm(value, width):
    return is_simm_scaled(value, width, 1)


def match_simm(width):
    def match_func(node, values, idx, dag):
        from rachetwrench.codegen.dag import VirtualDagOps
        from rachetwrench.codegen.types import ValueType

        value = values[idx]
        if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            return idx, None

        constant = value.node.value.value
        if is_simm(constant, width):
            target_value = DagValue(dag.add_target_constant_node(
                value.ty, constant), 0)

            return idx + 1, target_value

        return idx, None

    return match_func


simm9 = ComplexOperandMatcher(match_simm(9))


def addsub_shifted_imm(width):
    def match_func(node, values, idx, dag):
        from rachetwrench.codegen.dag import VirtualDagOps
        from rachetwrench.codegen.types import ValueType

        value = values[idx]
        if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            return idx, None

        constant = value.node.value.value
        if constant >= 0 and constant < 4096:
            target_value = DagValue(dag.add_target_constant_node(
                value.ty, constant), 0)

            return idx + 1, target_value

        return idx, None

    return match_func


addsub_shifted_imm32 = ComplexOperandMatcher(addsub_shifted_imm(32))
addsub_shifted_imm64 = ComplexOperandMatcher(addsub_shifted_imm(64))


def addsub_shifted_imm_neg(width):
    def match_func(node, values, idx, dag):
        from rachetwrench.codegen.dag import VirtualDagOps
        from rachetwrench.codegen.types import ValueType

        value = values[idx]
        if value.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.TARGET_CONSTANT]:
            return idx, None

        constant = value.node.value.value
        if constant < 0 and constant > -4096:
            target_value = DagValue(dag.add_target_constant_node(
                value.ty, constant), 0)

            return idx + 1, target_value

        return idx, None

    return match_func


addsub_shifted_imm32_neg = ComplexOperandMatcher(addsub_shifted_imm_neg(32))
addsub_shifted_imm64_neg = ComplexOperandMatcher(addsub_shifted_imm_neg(64))


def match_fpimm0(node, values, idx, dag):
    from rachetwrench.codegen.dag import VirtualDagOps
    from rachetwrench.codegen.types import ValueType

    value = values[idx]
    if value.node.opcode not in [VirtualDagOps.CONSTANT_FP, VirtualDagOps.TARGET_CONSTANT_FP]:
        return idx, None

    constant = value.node.value.value
    if constant == 0:
        target_value = DagValue(dag.add_target_constant_fp_node(
            value.ty, value.node.value), 0)

        return idx + 1, target_value

    return idx, None


fpimm0 = ComplexOperandMatcher(match_fpimm0)


class SchedReadWrite:
    pass


class SchedWrite(SchedReadWrite):
    pass


class ProcResource:
    def __init__(self, num):
        self.num = num


class ProcResGroup:
    def __init__(self, resources):
        self.resources = resources


A72UnitB = ProcResource(1)
A72UnitI = ProcResource(1)
A72UnitM = ProcResource(1)
A72UnitL = ProcResource(1)
A72UnitS = ProcResource(1)
A72UnitX = ProcResource(1)
A72UnitW = ProcResource(1)
A72UnitV = ProcResGroup([A72UnitX, A72UnitW])


class ProcWriteResources:
    def __init__(self, resources, latency=1):
        self.resources = resources
        self.latency = latency


class SchedWriteRes(ProcWriteResources):
    def __init__(self, resources, latency=1, num_uops=1):
        super().__init__(resources, latency)


A72Write_1cyc_1B = SchedWriteRes([A72UnitB], latency=1)
A72Write_1cyc_1I = SchedWriteRes([A72UnitI], latency=1)
A72Write_2cyc_1M = SchedWriteRes([A72UnitM], latency=1)
A72Write_19cyc_1M = SchedWriteRes([A72UnitM], latency=19)
A72Write_35cyc_1M = SchedWriteRes([A72UnitM], latency=35)
A72Write_4cyc_1L = SchedWriteRes([A72UnitL], latency=4)
A72Write_5cyc_1L = SchedWriteRes([A72UnitL], latency=5)
A72Write_1cyc_1S = SchedWriteRes([A72UnitS], latency=1)
A72Write_4cyc_1I_1L = SchedWriteRes(
    [A72UnitI, A72UnitL], latency=4, num_uops=2)
A72Write_1cyc_1I_1S = SchedWriteRes(
    [A72UnitI, A72UnitS], latency=1, num_uops=2)
A72Write_3cyc_1V = SchedWriteRes([A72UnitV], latency=3)
A72Write_5cyc_1V = SchedWriteRes([A72UnitV], latency=5)
A72Write_17cyc_1W = SchedWriteRes([A72UnitW], latency=17)

WriteImm = A72Write_1cyc_1I
WriteI = A72Write_1cyc_1I
WriteISReg = A72Write_2cyc_1M
WriteIEReg = A72Write_2cyc_1M
WriteExtr = A72Write_1cyc_1I
WriteIS = A72Write_1cyc_1I
WriteID32 = A72Write_19cyc_1M
WriteID64 = A72Write_35cyc_1M
WriteBr = A72Write_1cyc_1B
WriteBrReg = A72Write_1cyc_1B
WriteLD = A72Write_4cyc_1L
WriteST = A72Write_1cyc_1S
WriteSTP = A72Write_1cyc_1S
WriteAdr = A72Write_1cyc_1I
WriteLDIdx = A72Write_4cyc_1I_1L
WriteSTIdx = A72Write_1cyc_1I_1S
WriteF = A72Write_3cyc_1V
WriteFCmp = A72Write_3cyc_1V
WriteFCvt = A72Write_5cyc_1V
WriteFCopy = A72Write_5cyc_1L
WriteFImm = A72Write_3cyc_1V
WriteFMul = A72Write_5cyc_1V
WriteFDiv = A72Write_17cyc_1W
WriteV = A72Write_3cyc_1V
WriteVLD = A72Write_5cyc_1L
WriteVST = A72Write_1cyc_1S


class SchedMachineMode:
    def __init(self):
        pass

    @property
    def issue_width(self):
        # Triple-issue
        return 3

    @property
    def uop_buffer_size(self):
        # ROB entry size
        return 128

    @property
    def mispredict_penalty(self):
        return 15


class AArch64MachineOps:
    @classmethod
    def insts(cls):
        for member, value in cls.__dict__.items():
            if isinstance(value, MachineInstructionDef):
                yield value

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

    LDRBBui = def_inst(
        "ldrbb_ui",
        outs=[("rt", GPR32)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR32), zextloadi8_(
            am_indexed8(("rn", GPR64sp), ("offset", uimm12s1))))],
        sched=WriteLD,
    )

    LDRHHui = def_inst(
        "ldrhh_ui",
        outs=[("rt", GPR32)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR32), zextloadi16_(
            am_indexed16(("rn", GPR64sp), ("offset", uimm12s2))))],
        sched=WriteLD,
    )

    LDRWui = def_inst(
        "ldrw_ui",
        outs=[("rt", GPR32)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR32), load_(
            am_indexed32(("rn", GPR64sp), ("offset", uimm12s4))))],
        sched=WriteLD,
    )

    LDRXui = def_inst(
        "ldrx_ui",
        outs=[("dst", GPR64)],
        ins=[("src", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR64), load_(
            am_indexed64(("src", GPR64sp), ("offset", uimm12s8))))],
        sched=WriteLD,
    )

    LDRBui = def_inst(
        "ldrb_ui",
        outs=[("dst", FPR8)],
        ins=[("src", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR16), load_(
            am_indexed8(("src", GPR64sp), ("offset", uimm12s1))))],
        sched=WriteLD,
    )

    LDRHui = def_inst(
        "ldrh_ui",
        outs=[("dst", FPR16)],
        ins=[("src", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR16), load_(
            am_indexed16(("src", GPR64sp), ("offset", uimm12s2))))],
        sched=WriteLD,
    )

    LDRSui = def_inst(
        "ldrs_ui",
        outs=[("dst", FPR32)],
        ins=[("src", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR32), load_(
            am_indexed32(("src", GPR64sp), ("offset", uimm12s4))))],
        sched=WriteLD,
    )

    LDRDui = def_inst(
        "ldrd_ui",
        outs=[("dst", FPR64)],
        ins=[("src", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR64), load_(
            am_indexed64(("src", GPR64sp), ("offset", uimm12s8))))],
        sched=WriteLD,
    )

    LDRQui = def_inst(
        "ldrq_ui",
        outs=[("dst", FPR128)],
        ins=[("src", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR128), load_(
            am_indexed128(("src", GPR64sp), ("offset", uimm12s16))))],
        sched=WriteLD,
    )

    LDURWi = def_inst(
        "ldurw_i",
        outs=[("rt", GPR32)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR32), load_(
            am_unscaled32(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDURXi = def_inst(
        "ldurx_i",
        outs=[("rt", GPR64)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", GPR64), load_(
            am_unscaled64(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDURBi = def_inst(
        "ldurb_i",
        outs=[("rt", FPR8)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR8), load_(
            am_unscaled8(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDURHi = def_inst(
        "ldurh_i",
        outs=[("rt", FPR16)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR16), load_(
            am_unscaled16(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDURSi = def_inst(
        "ldurs_i",
        outs=[("rt", FPR32)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR32), load_(
            am_unscaled32(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDURDi = def_inst(
        "ldurd_i",
        outs=[("rt", FPR64)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR64), load_(
            am_unscaled64(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDURQi = def_inst(
        "ldurq_i",
        outs=[("rt", FPR128)],
        ins=[("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[set_(("dst", FPR128), load_(
            am_unscaled128(("rn", GPR64sp), ("offset", simm9))))]
    )

    LDPWi = def_inst(
        "ldpw_i",
        outs=[],
        ins=[("rt1", GPR32), ("rt2", GPR32),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    LDPXi = def_inst(
        "ldpx_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    LDPSi = def_inst(
        "ldps_i",
        outs=[],
        ins=[("rt1", FPR32), ("rt2", FPR32),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    LDPDi = def_inst(
        "ldpd_i",
        outs=[],
        ins=[("rt1", FPR64), ("rt2", FPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    LDPQi = def_inst(
        "ldpq_i",
        outs=[],
        ins=[("rt1", FPR128), ("rt2", FPR128),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    LDPXprei = def_inst(
        "ldpxpre_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    LDPXposti = def_inst(
        "ldpxpost_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STRBBui = def_inst(
        "strbb_ui",
        outs=[],
        ins=[("rt", GPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[truncstorei8_(("rt", GPR32), am_indexed8(
            ("rn", GPR64sp), ("offset", uimm12s1)))]
    )

    STRHHui = def_inst(
        "strhh_ui",
        outs=[],
        ins=[("rt", GPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[truncstorei16_(("rt", GPR32), am_indexed16(
            ("rn", GPR64sp), ("offset", uimm12s2)))]
    )

    STRWui = def_inst(
        "strw_ui",
        outs=[],
        ins=[("rt", GPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", GPR32), am_indexed32(
            ("rn", GPR64sp), ("offset", uimm12s4)))]
    )

    STRXui = def_inst(
        "strx_ui",
        outs=[],
        ins=[("rt", GPR64), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", GPR64), am_indexed64(
            ("rn", GPR64sp), ("offset", uimm12s8)))]
    )

    STRBui = def_inst(
        "strb_ui",
        outs=[],
        ins=[("rt", FPR8), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR8), am_indexed8(
            ("rn", GPR64sp), ("offset", uimm12s1)))]
    )

    STRHui = def_inst(
        "strh_ui",
        outs=[],
        ins=[("rt", FPR16), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR16), am_indexed16(
            ("rn", GPR64sp), ("offset", uimm12s2)))]
    )

    STRSui = def_inst(
        "strs_ui",
        outs=[],
        ins=[("rt", FPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR32), am_indexed32(
            ("rn", GPR64sp), ("offset", uimm12s4)))]
    )

    STRDui = def_inst(
        "strd_ui",
        outs=[],
        ins=[("rt", FPR64), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR64), am_indexed64(
            ("rn", GPR64sp), ("offset", uimm12s8)))]
    )

    STRQui = def_inst(
        "strq_ui",
        outs=[],
        ins=[("rt", FPR128), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR128), am_indexed128(
            ("rn", GPR64sp), ("offset", uimm12s16)))]
    )

    STURWi = def_inst(
        "sturw_i",
        outs=[],
        ins=[("rt", GPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", GPR32), am_unscaled32(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STURXi = def_inst(
        "sturx_i",
        outs=[],
        ins=[("rt", GPR64), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", GPR64), am_unscaled64(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STURBi = def_inst(
        "sturb_i",
        outs=[],
        ins=[("rt", FPR8), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR8), am_unscaled8(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STURHi = def_inst(
        "sturh_i",
        outs=[],
        ins=[("rt", FPR16), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR16), am_unscaled16(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STURSi = def_inst(
        "sturs_i",
        outs=[],
        ins=[("rt", FPR32), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR32), am_unscaled32(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STURDi = def_inst(
        "sturd_i",
        outs=[],
        ins=[("rt", FPR64), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR64), am_unscaled64(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STURQi = def_inst(
        "sturq_i",
        outs=[],
        ins=[("rt", FPR128), ("rn", GPR64sp), ("offset", I32Imm)],
        patterns=[store_(("rt", FPR128), am_unscaled128(
            ("rn", GPR64sp), ("offset", simm9)))]
    )

    STPXi = def_inst(
        "stpx_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPXi = def_inst(
        "stpx_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPXprei = def_inst(
        "stpxpre_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPXposti = def_inst(
        "stpxpost_i",
        outs=[],
        ins=[("rt1", GPR64), ("rt2", GPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPWi = def_inst(
        "stpw_i",
        outs=[],
        ins=[("rt1", GPR32), ("rt2", GPR32),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPWprei = def_inst(
        "stpwpre_i",
        outs=[],
        ins=[("rt1", GPR32), ("rt2", GPR32),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPWposti = def_inst(
        "stpwpost_i",
        outs=[],
        ins=[("rt1", GPR32), ("rt2", GPR32),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    STPDi = def_inst(
        "stpd_i",
        outs=[],
        ins=[("rt1", FPR64), ("rt2", FPR64),
             ("rn", GPR64sp), ("offset", I32Imm)]
    )

    ADDWri = def_inst(
        "addw_ri",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR32), add_(
            ("src1", GPR32), ("src2", addsub_shifted_imm32)))]
    )

    ADDXri = def_inst(
        "addx_ri",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", I64Imm)],
        patterns=[set_(("dst", GPR64), add_(
            ("src1", GPR64), ("src2", addsub_shifted_imm64)))]
    )

    SUBWri = def_inst(
        "subw_ri",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR32), sub_(
            ("src1", GPR32), ("src2", addsub_shifted_imm32)))]
    )

    SUBXri = def_inst(
        "subx_ri",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", I64Imm)],
        patterns=[set_(("dst", GPR64), sub_(
            ("src1", GPR64), ("src2", addsub_shifted_imm32)))]
    )

    MADDWrrr = def_inst(
        "maddw_rrr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32), ("src3", GPR32)]
    )

    MADDXrrr = def_inst(
        "maddx_rrr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64), ("src3", GPR64)]
    )

    UDIVWrr = def_inst(
        "udivw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), udiv_(
            ("src1", GPR32), ("src2", GPR32)))]
    )

    UDIVXrr = def_inst(
        "udivx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), udiv_(
            ("src1", GPR64), ("src2", GPR64)))]
    )

    SDIVWrr = def_inst(
        "sdivw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), sdiv_(
            ("src1", GPR32), ("src2", GPR32)))]
    )

    SDIVXrr = def_inst(
        "sdivx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), sdiv_(
            ("src1", GPR64), ("src2", GPR64)))]
    )

    MSUBWrrr = def_inst(
        "msubw_rrr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32), ("src3", GPR32)]
    )

    MSUBXrrr = def_inst(
        "msubx_rrr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64), ("src3", GPR64)]
    )

    ADDWrs = def_inst(
        "addw_rs",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[
            set_(("dst", GPR32), add_(
                ("src1", GPR32), ("src2", GPR32)))
        ]
    )

    ADDXrs = def_inst(
        "addx_rs",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[
            set_(("dst", GPR64), add_(
                ("src1", GPR64), ("src2", GPR64)))
        ]
    )

    SUBWrs = def_inst(
        "subw_rs",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[
            set_(("dst", GPR32), sub_(
                ("src1", GPR32), ("src2", GPR32)))
        ]
    )

    SUBXrs = def_inst(
        "subx_rs",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[
            set_(("dst", GPR64), sub_(
                ("src1", GPR64), ("src2", GPR64)))
        ]
    )

    ADDSWrs = def_inst(
        "addsw_rs",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        defs=[NZCV],
        patterns=[
            set_(("dst", GPR32), aarch64_adds_(
                ("src1", GPR32), ("src2", GPR32)))
        ]
    )

    ADDSXrs = def_inst(
        "addsx_rs",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        defs=[NZCV],
        patterns=[
            set_(("dst", GPR64), aarch64_adds_(
                ("src1", GPR64), ("src2", GPR64)))
        ]
    )

    SUBSWrs = def_inst(
        "subsw_rs",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        defs=[NZCV],
        patterns=[
            set_(("dst", GPR32), aarch64_subs_(
                ("src1", GPR32), ("src2", GPR32)))
        ]
    )

    SUBSXrs = def_inst(
        "subsx_rs",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        defs=[NZCV],
        patterns=[
            set_(("dst", GPR64), aarch64_subs_(
                ("src1", GPR64), ("src2", GPR64)))
        ]
    )

    ANDWrr = def_inst(
        "andw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), and_(("src1", GPR32), ("src2", GPR32)))]
    )

    ANDXrr = def_inst(
        "andx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), and_(("src1", GPR64), ("src2", GPR64)))]
    )

    EORWrr = def_inst(
        "eorw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), xor_(("src1", GPR32), ("src2", GPR32)))]
    )

    EORXrr = def_inst(
        "eorx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), xor_(("src1", GPR64), ("src2", GPR64)))]
    )

    ORRWrr = def_inst(
        "orrw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), or_(("src1", GPR32), ("src2", GPR32)))]
    )

    ORRXrr = def_inst(
        "orrx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), or_(("src1", GPR64), ("src2", GPR64)))]
    )

    ORRWri = def_inst(
        "orrw_ri",
        outs=[("dst", GPR32sp)],
        ins=[("src1", GPR32), ("src2", I32Imm)],
        patterns=[set_(("dst", GPR32sp), or_(
            ("src1", GPR32), ("src2", logical_imm32)))]
    )

    ORRXri = def_inst(
        "orrx_ri",
        outs=[("dst", GPR64sp)],
        ins=[("src1", GPR64), ("src2", I64Imm)],
        patterns=[set_(("dst", GPR64sp), or_(
            ("src1", GPR64), ("src2", logical_imm32)))]
    )

    ASRVWrr = def_inst(
        "asrvw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), sra_(("src1", GPR32), ("src2", GPR32)))]
    )

    ASRVXrr = def_inst(
        "asrvx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), sra_(("src1", GPR64), ("src2", GPR64)))]
    )

    LSLVWrr = def_inst(
        "lslvw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), shl_(("src1", GPR32), ("src2", GPR32)))]
    )

    LSLVXrr = def_inst(
        "lslvx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), shl_(("src1", GPR64), ("src2", GPR64)))]
    )

    LSRVWrr = def_inst(
        "lsrvw_rr",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32)],
        patterns=[set_(("dst", GPR32), srl_(("src1", GPR32), ("src2", GPR32)))]
    )

    LSRVXrr = def_inst(
        "lsrvx_rr",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64)],
        patterns=[set_(("dst", GPR64), srl_(("src1", GPR64), ("src2", GPR64)))]
    )

    MOVZWi = def_inst(
        "movzw_i",
        outs=[("dst", GPR32)],
        ins=[("imm", I32Imm), ("shift", I32Imm)]
    )

    MOVZXi = def_inst(
        "movzx_i",
        outs=[("dst", GPR64)],
        ins=[("imm", I32Imm), ("shift", I32Imm)]
    )

    MOVKWi = def_inst(
        "movkw_i",
        outs=[("dst", GPR32)],
        ins=[("src", GPR32), ("imm", I32Imm), ("shift", I32Imm)],
        constraints=[Constraint("dst", "src")]
    )

    MOVKXi = def_inst(
        "movkx_i",
        outs=[("dst", GPR64)],
        ins=[("src", GPR64), ("imm", I32Imm), ("shift", I32Imm)],
        constraints=[Constraint("dst", "src")]
    )

    SBFMWri = def_inst(
        "sbfmw_ri",
        outs=[("rd", GPR32)],
        ins=[("rn", GPR32), ("immr", I32Imm), ("imms", I32Imm)]
    )

    SBFMXri = def_inst(
        "sbfmx_ri",
        outs=[("rd", GPR64)],
        ins=[("rn", GPR64), ("immr", I32Imm), ("imms", I32Imm)]
    )

    UBFMWri = def_inst(
        "ubfmw_ri",
        outs=[("rd", GPR32)],
        ins=[("rn", GPR32), ("immr", I32Imm), ("imms", I32Imm)]
    )

    UBFMXri = def_inst(
        "ubfmx_ri",
        outs=[("rd", GPR64)],
        ins=[("rn", GPR64), ("immr", I32Imm), ("imms", I32Imm)]
    )

    ADR = def_inst(
        "adr",
        outs=[("dst", GPR64)],
        ins=[("src", AdrLabel)],
        patterns=[set_(("dst", GPR64), aarch64_adr_(("src", tglobaladdr_)))]
    )

    ADRP = def_inst(
        "adrp",
        outs=[("dst", GPR64)],
        ins=[("src", AdrLabel)],
        patterns=[
            set_(("dst", GPR64), aarch64_adrp_(("src", tglobaladdr_))),
            set_(("dst", GPR64), aarch64_adrp_(("src", tconstpool_)))]
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

    BL = def_inst(
        "bl",
        outs=[],
        ins=[("addr", BrTarget8)],
        patterns=[br_(("addr", bb))],
        is_call=True
    )

    BLR = def_inst(
        "blr",
        outs=[],
        ins=[("addr", GPR64)],
        is_call=True
    )

    RET = def_inst(
        "ret",
        outs=[],
        ins=[("rn", GPR64)],
        is_terminator=True
    )

    FADDHrr = def_inst(
        "faddh_rr",
        outs=[("dst", FPR16)],
        ins=[("src1", FPR16), ("src2", FPR16)],
        patterns=[set_(("dst", FPR16), fadd_(
            ("src1", FPR16), ("src2", FPR16)))]
    )

    FADDSrr = def_inst(
        "fadds_rr",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fadd_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FADDDrr = def_inst(
        "faddd_rr",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fadd_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FSUBHrr = def_inst(
        "fsubh_rr",
        outs=[("dst", FPR16)],
        ins=[("src1", FPR16), ("src2", FPR16)],
        patterns=[set_(("dst", FPR16), fsub_(
            ("src1", FPR16), ("src2", FPR16)))]
    )

    FSUBSrr = def_inst(
        "fsubs_rr",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fsub_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FSUBDrr = def_inst(
        "fsubd_rr",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fsub_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FMULHrr = def_inst(
        "fmulh_rr",
        outs=[("dst", FPR16)],
        ins=[("src1", FPR16), ("src2", FPR16)],
        patterns=[set_(("dst", FPR16), fmul_(
            ("src1", FPR16), ("src2", FPR16)))]
    )

    FMULSrr = def_inst(
        "fmuls_rr",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fmul_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FMULDrr = def_inst(
        "fmuld_rr",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fmul_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FDIVHrr = def_inst(
        "fdivh_rr",
        outs=[("dst", FPR16)],
        ins=[("src1", FPR16), ("src2", FPR16)],
        patterns=[set_(("dst", FPR16), fdiv_(
            ("src1", FPR16), ("src2", FPR16)))]
    )

    FDIVSrr = def_inst(
        "fdivs_rr",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fdiv_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FDIVDrr = def_inst(
        "fdivd_rr",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fdiv_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FCMPHrr = def_inst(
        "fcmph_rr",
        outs=[],
        ins=[("src1", FPR16), ("src2", FPR16)],
        defs=[NZCV],
        patterns=[
            aarch64_fcmp_(("src1", FPR16), ("src2", FPR16))
        ]
    )

    FCMPSrr = def_inst(
        "fcmps_rr",
        outs=[],
        ins=[("src1", FPR32), ("src2", FPR32)],
        defs=[NZCV],
        patterns=[
            aarch64_fcmp_(("src1", FPR32), ("src2", FPR32))
        ]
    )

    FCMPDrr = def_inst(
        "fcmpd_rr",
        outs=[],
        ins=[("src1", FPR64), ("src2", FPR64)],
        defs=[NZCV],
        patterns=[
            aarch64_fcmp_(("src1", FPR64), ("src2", FPR64))
        ]
    )

    FMOVS0 = def_inst(
        "fmovs_0",
        outs=[("dst", FPR32)],
        ins=[],
        patterns=[set_(("dst", FPR32), fpimm0)]
    )

    FMOVD0 = def_inst(
        "fmovd_0",
        outs=[("dst", FPR64)],
        ins=[],
        patterns=[set_(("dst", FPR64), fpimm0)]
    )

    FMOVHr = def_inst(
        "fmovh_r",
        outs=[("dst", FPR16)],
        ins=[("src", FPR16)]
    )

    FMOVSr = def_inst(
        "fmovs_r",
        outs=[("dst", FPR32)],
        ins=[("src", FPR32)]
    )

    FMOVDr = def_inst(
        "fmovd_r",
        outs=[("dst", FPR64)],
        ins=[("src", FPR64)]
    )

    FMOVWHr = def_inst(
        "fmovwh_r",
        outs=[("dst", FPR16)],
        ins=[("src", GPR32)]
    )

    FMOVXHr = def_inst(
        "fmovxh_r",
        outs=[("dst", FPR16)],
        ins=[("src", GPR64)]
    )

    FMOVWSr = def_inst(
        "fmovws_r",
        outs=[("dst", FPR32)],
        ins=[("src", GPR32)]
    )

    FMOVXDr = def_inst(
        "fmovxd_r",
        outs=[("dst", FPR64)],
        ins=[("src", GPR64)]
    )

    CSINCWr = def_inst(
        "csincw_r",
        outs=[("dst", GPR32)],
        ins=[("src1", GPR32), ("src2", GPR32), ("ccond", I32Imm)]
    )

    CSINCXr = def_inst(
        "csincx_r",
        outs=[("dst", GPR64)],
        ins=[("src1", GPR64), ("src2", GPR64), ("ccond", I32Imm)]
    )

    # fcvt

    FCVTSHr = def_inst(
        "fcvtsh_r",
        outs=[("dst", FPR32)],
        ins=[("src", FPR16)],
        patterns=[set_(("dst", FPR32), fp_round_(("src", FPR16)))]
    )

    FCVTDHr = def_inst(
        "fcvtdh_r",
        outs=[("dst", FPR64)],
        ins=[("src", FPR16)],
        patterns=[set_(("dst", FPR64), fp_round_(("src", FPR16)))]
    )

    FCVTHSr = def_inst(
        "fcvths_r",
        outs=[("dst", FPR16)],
        ins=[("src", FPR32)],
        patterns=[set_(("dst", FPR16), fp_extend_(("src", FPR32)))]
    )

    FCVTDSr = def_inst(
        "fcvtds_r",
        outs=[("dst", FPR64)],
        ins=[("src", FPR32)],
        patterns=[set_(("dst", FPR64), fp_extend_(("src", FPR32)))]
    )

    FCVTHDr = def_inst(
        "fcvthd_r",
        outs=[("dst", FPR16)],
        ins=[("src", FPR64)],
        patterns=[set_(("dst", FPR16), fp_round_(("src", FPR64)))]
    )

    FCVTSDr = def_inst(
        "fcvtsd_r",
        outs=[("dst", FPR32)],
        ins=[("src", FPR64)],
        patterns=[set_(("dst", FPR32), fp_round_(("src", FPR64)))]
    )

    # unscaled
    FCVTZSUWHr = def_inst(
        "fcvtzsuwh_r",
        outs=[("dst", GPR32)],
        ins=[("src", FPR16)],
        patterns=[set_(("dst", GPR32), fp_to_sint_(("src", FPR16)))]
    )

    FCVTZSUXHr = def_inst(
        "fcvtzsuxh_r",
        outs=[("dst", GPR64)],
        ins=[("src", FPR16)],
        patterns=[set_(("dst", GPR64), fp_to_sint_(("src", FPR16)))]
    )

    FCVTZSUWSr = def_inst(
        "fcvtzsuws_r",
        outs=[("dst", GPR32)],
        ins=[("src", FPR32)],
        patterns=[set_(("dst", GPR32), fp_to_sint_(("src", FPR32)))]
    )

    FCVTZSUXSr = def_inst(
        "fcvtzsuxs_r",
        outs=[("dst", GPR64)],
        ins=[("src", FPR32)],
        patterns=[set_(("dst", GPR64), fp_to_sint_(("src", FPR32)))]
    )

    FCVTZSUWDr = def_inst(
        "fcvtzsuwd_r",
        outs=[("dst", GPR32)],
        ins=[("src", FPR64)],
        patterns=[set_(("dst", GPR32), fp_to_sint_(("src", FPR64)))]
    )

    FCVTZSUXDr = def_inst(
        "fcvtzsuxd_r",
        outs=[("dst", GPR64)],
        ins=[("src", FPR64)],
        patterns=[set_(("dst", GPR64), fp_to_sint_(("src", FPR64)))]
    )

    # unscaled
    # signed
    SCVTFUWHr = def_inst(
        "scvtfuwh_r",
        outs=[("dst", FPR16)],
        ins=[("src", GPR32)],
        patterns=[set_(("dst", FPR16), sint_to_fp_(("src", GPR32)))]
    )

    SCVTFUWSr = def_inst(
        "scvtfuws_r",
        outs=[("dst", FPR32)],
        ins=[("src", GPR32)],
        patterns=[set_(("dst", FPR32), sint_to_fp_(("src", GPR32)))]
    )

    SCVTFUWDr = def_inst(
        "scvtfuwd_r",
        outs=[("dst", FPR64)],
        ins=[("src", GPR32)],
        patterns=[set_(("dst", FPR64), sint_to_fp_(("src", GPR32)))]
    )

    SCVTFUXHr = def_inst(
        "scvtfuxh_r",
        outs=[("dst", FPR16)],
        ins=[("src", GPR64)],
        patterns=[set_(("dst", FPR16), sint_to_fp_(("src", GPR64)))]
    )

    SCVTFUXSr = def_inst(
        "scvtfuxs_r",
        outs=[("dst", FPR32)],
        ins=[("src", GPR64)],
        patterns=[set_(("dst", FPR32), sint_to_fp_(("src", GPR64)))]
    )

    SCVTFUXDr = def_inst(
        "scvtfuxd_r",
        outs=[("dst", FPR64)],
        ins=[("src", GPR64)],
        patterns=[set_(("dst", FPR64), sint_to_fp_(("src", GPR64)))]
    )

    # unsigned
    UCVTFUWHr = def_inst(
        "ucvtfuwh_r",
        outs=[("dst", FPR16)],
        ins=[("src", GPR32)],
        patterns=[set_(("dst", FPR16), uint_to_fp_(("src", GPR32)))]
    )

    UCVTFUWSr = def_inst(
        "ucvtfuws_r",
        outs=[("dst", FPR32)],
        ins=[("src", GPR32)],
        patterns=[set_(("dst", FPR32), uint_to_fp_(("src", GPR32)))]
    )

    UCVTFUWDr = def_inst(
        "ucvtfuwd_r",
        outs=[("dst", FPR64)],
        ins=[("src", GPR32)],
        patterns=[set_(("dst", FPR64), uint_to_fp_(("src", GPR32)))]
    )

    UCVTFUXHr = def_inst(
        "ucvtfuxh_r",
        outs=[("dst", FPR16)],
        ins=[("src", GPR64)],
        patterns=[set_(("dst", FPR16), uint_to_fp_(("src", GPR64)))]
    )

    UCVTFUXSr = def_inst(
        "ucvtfuxs_r",
        outs=[("dst", FPR32)],
        ins=[("src", GPR64)],
        patterns=[set_(("dst", FPR32), uint_to_fp_(("src", GPR64)))]
    )

    UCVTFUXDr = def_inst(
        "ucvtfuxd_r",
        outs=[("dst", FPR64)],
        ins=[("src", GPR64)],
        patterns=[set_(("dst", FPR64), uint_to_fp_(("src", GPR64)))]
    )

    # simd

    ORRv16i8 = def_inst(
        "orr_v16i8",
        outs=[("dst", FPR128)],
        ins=[("src1", FPR128), ("src2", FPR128)]
    )

    FADDv4f32 = def_inst(
        "fadd_v4f32",
        outs=[("dst", FPR128)],
        ins=[("src1", FPR128), ("src2", FPR128)],
        patterns=[set_(("dst", FPR128), fadd_(
            ("src1", FPR128), ("src2", FPR128)))]
    )

    FSUBv4f32 = def_inst(
        "fsub_v4f32",
        outs=[("dst", FPR128)],
        ins=[("src1", FPR128), ("src2", FPR128)],
        patterns=[set_(("dst", FPR128), fsub_(
            ("src1", FPR128), ("src2", FPR128)))]
    )

    FMULv4f32 = def_inst(
        "fmul_v4f32",
        outs=[("dst", FPR128)],
        ins=[("src1", FPR128), ("src2", FPR128)],
        patterns=[set_(("dst", FPR128), fmul_(
            ("src1", FPR128), ("src2", FPR128)))]
    )

    FDIVv4f32 = def_inst(
        "fdiv_v4f32",
        outs=[("dst", FPR128)],
        ins=[("src1", FPR128), ("src2", FPR128)],
        patterns=[set_(("dst", FPR128), fdiv_(
            ("src1", FPR128), ("src2", FPR128)))]
    )

    DUPv4i32lane = def_inst(
        "dup_v4i32_lane",
        outs=[("dst", FPR128)],
        ins=[("src", FPR128), ("idx", VectorIndex32)],
        constraints=[Constraint("dst", "src")]
    )

    INSvi32lane = def_inst(
        "ins_vi32_lane",
        outs=[("dst", FPR128)],
        ins=[("src", FPR128), ("idx", VectorIndex32),
             ("elem", FPR128), ("idx2", VectorIndex32)],
        constraints=[Constraint("dst", "src")]
    )

    MRS = def_inst(
        "mrs",
        outs=[("dst", GPR64)],
        ins=[("sysreg", mrs_sysreg_op)]
    )

    # pseudo instructions
    RET_ReallyLR = def_inst(
        "ret_reallylr",
        outs=[],
        ins=[],
        patterns=[(aarch64_let_())],
        is_terminator=True
    )

    MOVaddr = def_inst(
        "mov_addr",
        outs=[("dst", GPR64)],
        ins=[("hi", I64Imm), ("lo", I64Imm)],
        patterns=[set_(("dst", GPR64), aarch64_addlow_(aarch64_adrp_(
            ("hi", tglobaladdr_)), ("lo", tglobaladdr_)))]
    )

    MOVaddrCP = def_inst(
        "mov_addr_cp",
        outs=[("dst", GPR64)],
        ins=[("hi", I64Imm), ("lo", I64Imm)],
        patterns=[set_(("dst", GPR64), aarch64_addlow_(aarch64_adrp_(
            ("hi", tconstpool_)), ("lo", tconstpool_)))]
    )

    TLSDESC_CALLSEQ = def_inst(
        "tlsdesc_callseq",
        outs=[],
        ins=[("sym", I64Imm)],
        defs=[LR, X0, X1],
        patterns=[aarch64_tlsdesc_callseq_(("sym", tglobaltlsaddr_))]
    )

    TLSDESCCALL = def_inst(
        "tlsdesccall",
        outs=[],
        ins=[("sym", I64Imm)]
    )

    MOVbaseTLS = def_inst(
        "mov_base_tls",
        outs=[("dst", GPR64)],
        ins=[],
        patterns=[set_(("dst", GPR64), aarch64_thread_pointer_())]
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


i32shift_a = def_node_xform_(I64Imm, lambda value, dag: DagValue(dag.add_target_constant_node(
    value.ty, ConstantInt((32 - value.node.value.value) & 0x1f, value.node.value.ty)), 0))

i32shift_b = def_node_xform_(I64Imm, lambda value, dag: DagValue(dag.add_target_constant_node(
    value.ty, ConstantInt(31 - value.node.value.value, value.node.value.ty)), 0))


aarch64_patterns = []


def def_pat_aarch64(pattern, result):
    def_pat(pattern, result, aarch64_patterns)


UBFMWri = def_inst_node_(AArch64MachineOps.UBFMWri)
UBFMXri = def_inst_node_(AArch64MachineOps.UBFMXri)

def_pat_aarch64(shl_(("rn", GPR32), ("imm", imm0_31)),
                UBFMWri(("rn", GPR32), i32shift_a(("imm", I32Imm)), i32shift_b(("imm", I32Imm))))


def_pat_aarch64(srl_(("rn", GPR32), ("imm", imm0_31)),
                UBFMWri(("rn", GPR32), ("imm", I32Imm), 31))

def_pat_aarch64(srl_(("rn", GPR64), ("imm", imm0_63)),
                UBFMXri(("rn", GPR64), ("imm", I32Imm), 63))

SBFMWri = def_inst_node_(AArch64MachineOps.SBFMWri)
SBFMXri = def_inst_node_(AArch64MachineOps.SBFMXri)

def_pat_aarch64(sext_inreg_(("rn", GPR32), i1_),
                SBFMWri(("rn", GPR32), 0, 0))

def_pat_aarch64(sext_inreg_(("rn", GPR32), i8_),
                SBFMWri(("rn", GPR32), 0, 7))

def_pat_aarch64(sext_inreg_(("rn", GPR32), i16_),
                SBFMWri(("rn", GPR32), 0, 15))

def_pat_aarch64(sext_inreg_(("rn", GPR64), i1_),
                SBFMXri(("rn", GPR64), 0, 0))

def_pat_aarch64(sext_inreg_(("rn", GPR64), i8_),
                SBFMXri(("rn", GPR64), 0, 7))

def_pat_aarch64(sext_inreg_(("rn", GPR64), i16_),
                SBFMXri(("rn", GPR64), 0, 15))

def_pat_aarch64(sext_inreg_(("rn", GPR64), i32_),
                SBFMXri(("rn", GPR64), 0, 31))

CSINCWr = def_inst_node_(AArch64MachineOps.CSINCWr)
CSINCXr = def_inst_node_(AArch64MachineOps.CSINCXr)

def_pat_aarch64(aarch64_csel_(i32_(0), i32_(1), ("imm", timm)),
                CSINCWr(WZR, WZR, ("imm", I32Imm)))

def_pat_aarch64(f128_(bitconvert_(("src", FPR128))), f128_(("src", FPR128)))

INSvi32lane = def_inst_node_(AArch64MachineOps.INSvi32lane)

# def_pat_aarch64(v4f32_(vector_insert_(("vec", FPR128),
#                                       ("elm", FPR32), ("idx", imm))), INSvi32lane(("vec", FPR128), ("idx", I32Imm)))


MADDWrrr = def_inst_node_(AArch64MachineOps.MADDWrrr)
MADDXrrr = def_inst_node_(AArch64MachineOps.MADDXrrr)

def_pat_aarch64(i32_(mul_(("rn", GPR32), ("rm", GPR32))),
                MADDWrrr(("rn", GPR32), ("rm", GPR32), WZR))
def_pat_aarch64(i64_(mul_(("rn", GPR64), ("rm", GPR64))),
                MADDXrrr(("rn", GPR64), ("rm", GPR64), XZR))
