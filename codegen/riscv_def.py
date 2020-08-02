from codegen.spec import *
from codegen.mir_emitter import *
from codegen.matcher import *

sub_32 = def_subreg("sub_32", 32)


# integer registers
X0 = def_reg("x0", aliases=["zero"], encoding=0)
X1 = def_reg("x1", aliases=["ra"], encoding=1)
X2 = def_reg("x2", aliases=["sp"], encoding=2)
X3 = def_reg("x3", aliases=["gp"], encoding=3)
X4 = def_reg("x4", aliases=["tp"], encoding=4)
X5 = def_reg("x5", aliases=["t0"], encoding=5)
X6 = def_reg("x6", aliases=["t1"], encoding=6)
X7 = def_reg("x7", aliases=["t2"], encoding=7)
X8 = def_reg("x8", aliases=["s0", "fp"], encoding=8)
X9 = def_reg("x9", aliases=["s1"], encoding=9)
X10 = def_reg("x10", aliases=["a0"], encoding=10)
X11 = def_reg("x11", aliases=["a1"], encoding=11)
X12 = def_reg("x12", aliases=["a2"], encoding=12)
X13 = def_reg("x13", aliases=["a3"], encoding=13)
X14 = def_reg("x14", aliases=["a4"], encoding=14)
X15 = def_reg("x15", aliases=["a5"], encoding=15)
X16 = def_reg("x16", aliases=["a6"], encoding=16)
X17 = def_reg("x17", aliases=["a7"], encoding=17)
X18 = def_reg("x18", aliases=["s2"], encoding=18)
X19 = def_reg("x19", aliases=["s3"], encoding=19)
X20 = def_reg("x20", aliases=["s4"], encoding=20)
X21 = def_reg("x21", aliases=["s5"], encoding=21)
X22 = def_reg("x22", aliases=["s6"], encoding=22)
X23 = def_reg("x23", aliases=["s7"], encoding=23)
X24 = def_reg("x24", aliases=["s8"], encoding=24)
X25 = def_reg("x25", aliases=["s9"], encoding=25)
X26 = def_reg("x26", aliases=["s10"], encoding=26)
X27 = def_reg("x27", aliases=["s11"], encoding=27)
X28 = def_reg("x28", aliases=["t3"], encoding=28)
X29 = def_reg("x29", aliases=["t4"], encoding=29)
X30 = def_reg("x30", aliases=["t5"], encoding=30)
X31 = def_reg("x31", aliases=["t6"], encoding=31)

# floating-point registers
F0_F = def_reg("f0", aliases=["ft0"], encoding=0)
F1_F = def_reg("f1", aliases=["ft1"], encoding=1)
F2_F = def_reg("f2", aliases=["ft2"], encoding=2)
F3_F = def_reg("f3", aliases=["ft3"], encoding=3)
F4_F = def_reg("f4", aliases=["ft4"], encoding=4)
F5_F = def_reg("f5", aliases=["ft5"], encoding=5)
F6_F = def_reg("f6", aliases=["ft6"], encoding=6)
F7_F = def_reg("f7", aliases=["ft7"], encoding=7)
F8_F = def_reg("f8", aliases=["fs0"], encoding=8)
F9_F = def_reg("f9", aliases=["fs1"], encoding=9)
F10_F = def_reg("f10", aliases=["fa0"], encoding=10)
F11_F = def_reg("f11", aliases=["fa1"], encoding=11)
F12_F = def_reg("f12", aliases=["fa2"], encoding=12)
F13_F = def_reg("f13", aliases=["fa3"], encoding=13)
F14_F = def_reg("f14", aliases=["fa4"], encoding=14)
F15_F = def_reg("f15", aliases=["fa5"], encoding=15)
F16_F = def_reg("f16", aliases=["fa6"], encoding=16)
F17_F = def_reg("f17", aliases=["fa7"], encoding=17)
F18_F = def_reg("f18", aliases=["fs2"], encoding=18)
F19_F = def_reg("f19", aliases=["fs3"], encoding=19)
F20_F = def_reg("f20", aliases=["fs4"], encoding=20)
F21_F = def_reg("f21", aliases=["fs5"], encoding=21)
F22_F = def_reg("f22", aliases=["fs6"], encoding=22)
F23_F = def_reg("f23", aliases=["fs7"], encoding=23)
F24_F = def_reg("f24", aliases=["fs8"], encoding=24)
F25_F = def_reg("f25", aliases=["fs9"], encoding=25)
F26_F = def_reg("f26", aliases=["fs10"], encoding=26)
F27_F = def_reg("f27", aliases=["fs11"], encoding=27)
F28_F = def_reg("f28", aliases=["ft8"], encoding=28)
F29_F = def_reg("f29", aliases=["ft9"], encoding=29)
F30_F = def_reg("f30", aliases=["ft10"], encoding=30)
F31_F = def_reg("f31", aliases=["ft11"], encoding=31)

F0_D = def_reg("d0", [F0_F], [sub_32], encoding=0)
F1_D = def_reg("d1", [F1_F], [sub_32], encoding=1)
F2_D = def_reg("d2", [F2_F], [sub_32], encoding=2)
F3_D = def_reg("d3", [F3_F], [sub_32], encoding=3)
F4_D = def_reg("d4", [F4_F], [sub_32], encoding=4)
F5_D = def_reg("d5", [F5_F], [sub_32], encoding=5)
F6_D = def_reg("d6", [F6_F], [sub_32], encoding=6)
F7_D = def_reg("d7", [F7_F], [sub_32], encoding=7)
F8_D = def_reg("d8", [F8_F], [sub_32], encoding=8)
F9_D = def_reg("d9", [F9_F], [sub_32], encoding=9)
F10_D = def_reg("d10", [F10_F], [sub_32], encoding=10)
F11_D = def_reg("d11", [F11_F], [sub_32], encoding=11)
F12_D = def_reg("d12", [F12_F], [sub_32], encoding=12)
F13_D = def_reg("d13", [F13_F], [sub_32], encoding=13)
F14_D = def_reg("d14", [F14_F], [sub_32], encoding=14)
F15_D = def_reg("d15", [F15_F], [sub_32], encoding=15)
F16_D = def_reg("d16", [F16_F], [sub_32], encoding=16)
F17_D = def_reg("d17", [F17_F], [sub_32], encoding=17)
F18_D = def_reg("d18", [F18_F], [sub_32], encoding=18)
F19_D = def_reg("d19", [F19_F], [sub_32], encoding=19)
F20_D = def_reg("d20", [F20_F], [sub_32], encoding=20)
F21_D = def_reg("d21", [F21_F], [sub_32], encoding=21)
F22_D = def_reg("d22", [F22_F], [sub_32], encoding=22)
F23_D = def_reg("d23", [F23_F], [sub_32], encoding=23)
F24_D = def_reg("d24", [F24_F], [sub_32], encoding=24)
F25_D = def_reg("d25", [F25_F], [sub_32], encoding=25)
F26_D = def_reg("d26", [F26_F], [sub_32], encoding=26)
F27_D = def_reg("d27", [F27_F], [sub_32], encoding=27)
F28_D = def_reg("d28", [F28_F], [sub_32], encoding=28)
F29_D = def_reg("d29", [F29_F], [sub_32], encoding=29)
F30_D = def_reg("d30", [F30_F], [sub_32], encoding=30)
F31_D = def_reg("d31", [F31_F], [sub_32], encoding=31)

riscv_regclasses = []


def def_riscv_regclass(*args, **kwargs):
    regclass = def_regclass(*args, **kwargs)
    riscv_regclasses.append(regclass)
    return regclass


RV32 = MachineHWMode("rv32")
RV64 = MachineHWMode("rv64")
RV128 = MachineHWMode("rv128")

XLenVT = ValueTypeByHWMode({
    RV32: ValueType.I32,
    RV64: ValueType.I64,
    RV128: ValueType.I128,
})


def sequence(format_str, start, end):
    seq = [globals()[format_str.format(i)]
           for i in range(start, (end - start))]
    return seq


GPR = def_riscv_regclass("GPR", [XLenVT], 32, [
                         globals()[f"X{idx}"] for idx in range(32)])


GPRX0 = def_riscv_regclass("GPRX0", [ValueType.I32], 32, [X0])

GPRwoPC = def_riscv_regclass("GPRwoX0", [ValueType.I32], 32, [
    globals()[f"X{idx}"] for idx in range(1, 32)])


SP = def_riscv_regclass("SP", [ValueType.I32], 32, [X2])

FPR32 = def_riscv_regclass("FPR", [ValueType.F32], 32, [
    globals()[f"F{idx}_F"] for idx in range(32)])

FPR64 = def_riscv_regclass("FPR", [ValueType.F64], 64, [
    globals()[f"F{idx}_D"] for idx in range(32)])

# build register graph
reg_graph = compute_reg_graph()
reg_groups = compute_reg_groups(reg_graph)
compute_reg_subregs_all(reg_graph)
compute_reg_superregs_all(reg_graph)

# infer regclass
for regclass in riscv_regclasses:
    infer_subregclass_and_subreg(regclass)


class RISCVMemOperandDef(ValueOperandDef):
    def __init__(self, op_info):
        super().__init__(ValueType.I32)

        self.op_info = op_info

    @property
    def operand_info(self):
        return self.op_info


AddrMode_Imm12 = RISCVMemOperandDef([GPR, I32Imm])
AddrMode5 = RISCVMemOperandDef([GPR, I32Imm])
AddrMode6 = RISCVMemOperandDef([GPR, I32Imm])
SORegImm = RISCVMemOperandDef([GPR, I32Imm])
riscv_bl_target = ValueOperandDef(ValueType.I32)
reglist = ValueOperandDef(ValueType.I32)
cmovpred = ValueOperandDef(ValueType.I32)
frmarg = ValueOperandDef(ValueType.I32)


class RISCVDagOp(DagOp):
    def __init__(self, name):
        super().__init__(name, "riscv_")


class RISCVDagOps(Enum):
    LUI = RISCVDagOp("lui")
    ADDI = RISCVDagOp("addi")
    SLLW = RISCVDagOp("sllw")
    SRAW = RISCVDagOp("sraw")
    SRLW = RISCVDagOp("srlw")

    SUB = RISCVDagOp("sub")

    CALL = RISCVDagOp("call")
    RETURN = RISCVDagOp("return")

    WRAPPER = RISCVDagOp("wrapper")
    WRAPPER_PIC = RISCVDagOp("wrapper_pic")
    WRAPPER_JT = RISCVDagOp("wrapper_jt")

    PIC_ADD = RISCVDagOp("pic_add")


riscv_sub_ = NodePatternMatcherGen(RISCVDagOps.SUB)
riscv_let_ = NodePatternMatcherGen(RISCVDagOps.RETURN)
riscv_call_ = NodePatternMatcherGen(RISCVDagOps.CALL)
riscv_wrapper_ = NodePatternMatcherGen(RISCVDagOps.WRAPPER)
riscv_pic_add_ = NodePatternMatcherGen(RISCVDagOps.PIC_ADD)

riscv_sllw_ = NodePatternMatcherGen(RISCVDagOps.SLLW)
riscv_sraw_ = NodePatternMatcherGen(RISCVDagOps.SRAW)
riscv_srlw_ = NodePatternMatcherGen(RISCVDagOps.SRLW)


def in_bits_signed(value, bits):
    mx = (1 << (bits - 1)) - 1
    mn = -mx - 1
    return value >= mn and value <= mx


def in_bits_unsigned(value, bits):
    mx = (1 << (bits)) - 1
    return value >= 0 and value <= mx


def match_addr(node, values, idx, dag):
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


addr = ComplexOperandMatcher(match_addr)


def match_fi_addr(node, values, idx, dag):
    from codegen.dag import VirtualDagOps, DagValue
    from codegen.types import ValueType

    value = values[idx]

    if value.node.opcode == VirtualDagOps.FRAME_INDEX:
        index = value.node.index
        fi = DagValue(
            dag.add_frame_index_node(value.ty, index, True), 0)

        return idx + 1, fi

    return idx, None


fi_addr = ComplexOperandMatcher(match_fi_addr)

CallSymbol = ValueOperandDef(ValueType.I32)


class RISCVMachineOps:

    @classmethod
    def insts(cls):
        for member, value in cls.__dict__.items():
            if isinstance(value, MachineInstructionDef):
                yield value

    LB = def_inst(
        "lb",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)]
    )

    LH = def_inst(
        "lh",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)]
    )

    LBU = def_inst(
        "lbu",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)]
    )

    LHU = def_inst(
        "lhu",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)]
    )

    LW = def_inst(
        "lw",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)]
    )

    LD = def_inst(
        "ld",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)]
    )

    FLW = def_inst(
        "flw",
        outs=[("dst", FPR32)],
        ins=[("src", AddrMode_Imm12)],
        patterns=[set_(("dst", FPR32), load_(("src", addr)))]
    )

    FLD = def_inst(
        "fld",
        outs=[("dst", FPR64)],
        ins=[("src", AddrMode_Imm12)],
        patterns=[set_(("dst", FPR64), load_(("src", addr)))]
    )

    SB = def_inst(
        "sb",
        outs=[],
        ins=[("dst", GPR), ("src", AddrMode_Imm12)]
    )

    SH = def_inst(
        "sh",
        outs=[],
        ins=[("dst", GPR), ("src", AddrMode_Imm12)]
    )

    SW = def_inst(
        "sw",
        outs=[],
        ins=[("dst", GPR), ("src", AddrMode_Imm12)]
    )

    SD = def_inst(
        "sd",
        outs=[],
        ins=[("dst", GPR), ("src", AddrMode_Imm12)]
    )

    FSW = def_inst(
        "fsw",
        outs=[],
        ins=[("dst", FPR32), ("src", AddrMode_Imm12)],
        patterns=[store_(("dst", FPR32), ("src", addr))]
    )

    FSD = def_inst(
        "fsd",
        outs=[],
        ins=[("dst", FPR64), ("src", AddrMode_Imm12)],
        patterns=[store_(("dst", FPR64), ("src", addr))]
    )

    LUI = def_inst(
        "lui",
        outs=[("dst", GPR)],
        ins=[("src", I32Imm)],
        patterns=[]
    )

    AUIPC = def_inst(
        "auipc",
        outs=[("dst", GPR)],
        ins=[("src", I32Imm)],
        patterns=[]
    )

    ADDI = def_inst(
        "addi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[]
    )

    SLTI = def_inst(
        "slti",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[]
    )

    SLTIU = def_inst(
        "sltiu",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[]
    )

    XORI = def_inst(
        "xori",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[]
    )

    ORI = def_inst(
        "ori",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[]
    )

    ANDI = def_inst(
        "andi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", I32Imm)],
        patterns=[]
    )

    ADD = def_inst(
        "add",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SUB = def_inst(
        "sub",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SLL = def_inst(
        "sll",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SLT = def_inst(
        "slt",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SLTU = def_inst(
        "sltu",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    XOR = def_inst(
        "xor",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SRL = def_inst(
        "srl",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SRA = def_inst(
        "sra",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    OR = def_inst(
        "or",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    AND = def_inst(
        "and",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SLLW = def_inst(
        "sllw",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SRLW = def_inst(
        "srlw",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    SRAW = def_inst(
        "sraw",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)],
        patterns=[]
    )

    # M extension

    MUL = def_inst(
        "mul",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)]
    )

    DIV = def_inst(
        "div",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)]
    )

    DIVU = def_inst(
        "divu",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)]
    )

    REM = def_inst(
        "rem",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)]
    )

    REMU = def_inst(
        "remu",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", GPR)]
    )

    # F, D extension

    FADD_S = def_inst(
        "fadd_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fadd_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FMUL_S = def_inst(
        "fmul_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fmul_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FDIV_S = def_inst(
        "fdiv_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fdiv_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FSUB_S = def_inst(
        "fsub_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fsub_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    FSGNJ_S = def_inst(
        "fsgnj_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)]
    )

    FSGNJN_S = def_inst(
        "fsgnjn_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)]
    )

    FSGNJX_S = def_inst(
        "fsgnjx_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)]
    )

    FEQ_S = def_inst(
        "feq_s",
        outs=[("dst", GPR)],
        ins=[("src1", FPR32), ("src2", FPR32)]
    )

    FLT_S = def_inst(
        "flt_s",
        outs=[("dst", GPR)],
        ins=[("src1", FPR32), ("src2", FPR32)]
    )

    FLE_S = def_inst(
        "fle_s",
        outs=[("dst", GPR)],
        ins=[("src1", FPR32), ("src2", FPR32)]
    )

    FADD_D = def_inst(
        "fadd_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fadd_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FMUL_D = def_inst(
        "fmul_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fmul_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FDIV_D = def_inst(
        "fdiv_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fdiv_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FSUB_D = def_inst(
        "fsub_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)],
        patterns=[set_(("dst", FPR64), fsub_(
            ("src1", FPR64), ("src2", FPR64)))]
    )

    FEQ_D = def_inst(
        "feq_d",
        outs=[("dst", GPR)],
        ins=[("src1", FPR64), ("src2", FPR64)]
    )

    FLT_D = def_inst(
        "flt_d",
        outs=[("dst", GPR)],
        ins=[("src1", FPR64), ("src2", FPR64)]
    )

    FLE_D = def_inst(
        "fle_d",
        outs=[("dst", GPR)],
        ins=[("src1", FPR64), ("src2", FPR64)]
    )

    FSGNJ_D = def_inst(
        "fsgnj_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)]
    )

    FSGNJN_D = def_inst(
        "fsgnjn_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)]
    )

    FSGNJX_D = def_inst(
        "fsgnjx_d",
        outs=[("dst", FPR64)],
        ins=[("src1", FPR64), ("src2", FPR64)]
    )

    FCLASS_S = def_inst(
        "fclass_s",
        outs=[("dst", GPR)],
        ins=[("src", FPR32)]
    )

    FCVT_W_S = def_inst(
        "fcvt_w_s",
        outs=[("dst", GPR)],
        ins=[("src", FPR32), ("frm", frmarg)]
    )

    FCVT_WU_S = def_inst(
        "fcvt_wu_s",
        outs=[("dst", GPR)],
        ins=[("src", FPR32), ("frm", frmarg)]
    )

    FCVT_S_W = def_inst(
        "fcvt_s_w",
        outs=[("dst", FPR32)],
        ins=[("src", GPR), ("frm", frmarg)]
    )

    FCVT_S_WU = def_inst(
        "fcvt_s_wu",
        outs=[("dst", FPR32)],
        ins=[("src", GPR), ("frm", frmarg)]
    )

    FCVT_L_S = def_inst(
        "fcvt_l_s",
        outs=[("dst", GPR)],
        ins=[("src", FPR32), ("frm", frmarg)]
    )

    FCVT_LU_S = def_inst(
        "fcvt_lu_s",
        outs=[("dst", GPR)],
        ins=[("src", FPR32), ("frm", frmarg)]
    )

    FCVT_S_L = def_inst(
        "fcvt_s_l",
        outs=[("dst", FPR32)],
        ins=[("src", GPR), ("frm", frmarg)]
    )

    FCVT_S_LU = def_inst(
        "fcvt_s_lu",
        outs=[("dst", FPR32)],
        ins=[("src", GPR), ("frm", frmarg)]
    )

    FMV_X_W = def_inst(
        "fmv_x_w",
        outs=[("dst", GPR)],
        ins=[("src", FPR32)]
    )

    FMV_W_X = def_inst(
        "fmv_w_x",
        outs=[("dst", FPR32)],
        ins=[("src", GPR)]
    )

    FCLASS_D = def_inst(
        "fclass_d",
        outs=[("dst", GPR)],
        ins=[("src", FPR64)]
    )

    FCVT_W_D = def_inst(
        "fcvt_w_d",
        outs=[("dst", GPR)],
        ins=[("src", FPR64), ("frm", frmarg)]
    )

    FCVT_WU_D = def_inst(
        "fcvt_wu_d",
        outs=[("dst", GPR)],
        ins=[("src", FPR64), ("frm", frmarg)]
    )

    FCVT_D_W = def_inst(
        "fcvt_d_w",
        outs=[("dst", FPR64)],
        ins=[("src", GPR)]
    )

    FCVT_D_WU = def_inst(
        "fcvt_d_wu",
        outs=[("dst", FPR64)],
        ins=[("src", GPR)]
    )

    FCVT_L_D = def_inst(
        "fcvt_l_d",
        outs=[("dst", GPR)],
        ins=[("src", FPR64), ("frm", frmarg)]
    )

    FCVT_LU_D = def_inst(
        "fcvt_lu_d",
        outs=[("dst", GPR)],
        ins=[("src", FPR64), ("frm", frmarg)]
    )

    FCVT_D_L = def_inst(
        "fcvt_d_l",
        outs=[("dst", FPR64)],
        ins=[("src", GPR), ("frm", frmarg)]
    )

    FCVT_D_LU = def_inst(
        "fcvt_d_lu",
        outs=[("dst", FPR64)],
        ins=[("src", GPR), ("frm", frmarg)]
    )

    FMV_X_D = def_inst(
        "fmv_x_d",
        outs=[("dst", GPR)],
        ins=[("src", FPR64)]
    )

    FMV_D_X = def_inst(
        "fmv_d_x",
        outs=[("dst", FPR64)],
        ins=[("src", GPR)]
    )

    FCVT_S_D = def_inst(
        "fcvt_s_d",
        outs=[("dst", FPR32)],
        ins=[("src", FPR64), ("frm", frmarg)]
    )

    FCVT_D_S = def_inst(
        "fcvt_d_s",
        outs=[("dst", FPR64)],
        ins=[("src", FPR32)]
    )

    JAL = def_inst(
        "jal",
        outs=[],
        ins=[("rs1", GPR), ("imm", BrTarget8)],
        is_terminator=True
    )

    JALR = def_inst(
        "jalr",
        outs=[],
        ins=[("rs1", GPR), ("rs2", GPR), ("imm", BrTarget8)],
        is_terminator=True
    )

    BEQ = def_inst(
        "beq",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR), ("dst", BrTarget8)],
        is_terminator=True,
        is_branch=True
    )

    BNE = def_inst(
        "bne",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR), ("dst", BrTarget8)],
        is_terminator=True,
        is_branch=True
    )

    BLT = def_inst(
        "blt",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR), ("dst", BrTarget8)],
        is_terminator=True,
        is_branch=True
    )

    BGE = def_inst(
        "bge",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR), ("dst", BrTarget8)],
        is_terminator=True,
        is_branch=True
    )

    BLTU = def_inst(
        "blgu",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR), ("dst", BrTarget8)],
        is_terminator=True,
        is_branch=True
    )

    BGEU = def_inst(
        "bgeu",
        outs=[],
        ins=[("src1", GPR), ("src2", GPR), ("dst", BrTarget8)],
        is_terminator=True,
        is_branch=True
    )

    PseudoBR = def_inst(
        "br",
        outs=[],
        ins=[("dst", BrTarget8)],
        patterns=[br_(("dst", bb))],
        is_terminator=True,
        is_branch=True,
        is_barrier=True
    )

    PseudoRET = def_inst(
        "ret",
        outs=[],
        ins=[],
        patterns=[(riscv_let_())],
        is_terminator=True
    )

    PseudoCALL = def_inst(
        "call",
        outs=[],
        ins=[("func", CallSymbol)],
        is_terminator=True,
        is_call=True,
    )

    PseudoLLA = def_inst(
        "lla",
        outs=[("dst", GPR)],
        ins=[("src", GPR)]
    )

    PseudoLA_TLS_GD = def_inst(
        "la_tls_gd",
        outs=[("dst", GPR)],
        ins=[("src", GPR)]
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

LB = def_inst_node_(RISCVMachineOps.LB)
LH = def_inst_node_(RISCVMachineOps.LH)
LBU = def_inst_node_(RISCVMachineOps.LBU)
LHU = def_inst_node_(RISCVMachineOps.LHU)
LW = def_inst_node_(RISCVMachineOps.LW)
LD = def_inst_node_(RISCVMachineOps.LD)

SB = def_inst_node_(RISCVMachineOps.SB)
SH = def_inst_node_(RISCVMachineOps.SH)
SW = def_inst_node_(RISCVMachineOps.SW)
SD = def_inst_node_(RISCVMachineOps.SD)

ADD = def_inst_node_(RISCVMachineOps.ADD)
SUB = def_inst_node_(RISCVMachineOps.SUB)
AND = def_inst_node_(RISCVMachineOps.AND)
OR = def_inst_node_(RISCVMachineOps.OR)
XOR = def_inst_node_(RISCVMachineOps.XOR)
SRA = def_inst_node_(RISCVMachineOps.SRA)
SRL = def_inst_node_(RISCVMachineOps.SRL)
SLL = def_inst_node_(RISCVMachineOps.SLL)
SLT = def_inst_node_(RISCVMachineOps.SLT)
SLTU = def_inst_node_(RISCVMachineOps.SLTU)

SRAW = def_inst_node_(RISCVMachineOps.SRAW)
SRLW = def_inst_node_(RISCVMachineOps.SRLW)
SLLW = def_inst_node_(RISCVMachineOps.SLLW)

ADDI = def_inst_node_(RISCVMachineOps.ADDI)
XORI = def_inst_node_(RISCVMachineOps.XORI)
SLTI = def_inst_node_(RISCVMachineOps.SLTI)
SLTIU = def_inst_node_(RISCVMachineOps.SLTIU)
LUI = def_inst_node_(RISCVMachineOps.LUI)

BNE = def_inst_node_(RISCVMachineOps.BNE)

PseudoCALL = def_inst_node_(RISCVMachineOps.PseudoCALL)

FEQ_S = def_inst_node_(RISCVMachineOps.FEQ_S)
FLT_S = def_inst_node_(RISCVMachineOps.FLT_S)
FLE_S = def_inst_node_(RISCVMachineOps.FLE_S)

FEQ_D = def_inst_node_(RISCVMachineOps.FEQ_D)
FLT_D = def_inst_node_(RISCVMachineOps.FLT_D)
FLE_D = def_inst_node_(RISCVMachineOps.FLE_D)

FCVT_W_S = def_inst_node_(RISCVMachineOps.FCVT_W_S)
FCVT_WU_S = def_inst_node_(RISCVMachineOps.FCVT_WU_S)
FCVT_S_W = def_inst_node_(RISCVMachineOps.FCVT_S_W)
FCVT_S_WU = def_inst_node_(RISCVMachineOps.FCVT_S_WU)

FCVT_L_S = def_inst_node_(RISCVMachineOps.FCVT_L_S)
FCVT_LU_S = def_inst_node_(RISCVMachineOps.FCVT_LU_S)
FCVT_S_L = def_inst_node_(RISCVMachineOps.FCVT_S_L)
FCVT_S_LU = def_inst_node_(RISCVMachineOps.FCVT_S_LU)

FCVT_W_D = def_inst_node_(RISCVMachineOps.FCVT_W_D)
FCVT_WU_D = def_inst_node_(RISCVMachineOps.FCVT_WU_D)
FCVT_D_W = def_inst_node_(RISCVMachineOps.FCVT_D_W)
FCVT_D_WU = def_inst_node_(RISCVMachineOps.FCVT_D_WU)

FCVT_L_D = def_inst_node_(RISCVMachineOps.FCVT_L_D)
FCVT_LU_D = def_inst_node_(RISCVMachineOps.FCVT_LU_D)
FCVT_D_L = def_inst_node_(RISCVMachineOps.FCVT_D_L)
FCVT_D_LU = def_inst_node_(RISCVMachineOps.FCVT_D_LU)

FCVT_D_S = def_inst_node_(RISCVMachineOps.FCVT_D_S)
FCVT_S_D = def_inst_node_(RISCVMachineOps.FCVT_S_D)

FMV_D_X = def_inst_node_(RISCVMachineOps.FMV_D_X)
FMV_X_D = def_inst_node_(RISCVMachineOps.FMV_X_D)

MUL = def_inst_node_(RISCVMachineOps.MUL)
DIV = def_inst_node_(RISCVMachineOps.DIV)
DIVU = def_inst_node_(RISCVMachineOps.DIVU)
REM = def_inst_node_(RISCVMachineOps.REM)
REMU = def_inst_node_(RISCVMachineOps.REMU)

from codegen.dag import DagValue


HI20 = def_node_xform_(I32Imm, lambda value, dag: DagValue(dag.add_target_constant_node(
    value.node.value_types[0], ConstantInt((value.node.value.value & 0xFFFFFFFF) >> 12, value.node.value.ty)), 0))


def get_bits_sext(value, bits):
    minus = value < 0
    value = abs(value)

    value = value & ((1 << bits) - 1)

    if minus:
        value = -value

    return value


LO12Sext = def_node_xform_(I32Imm, lambda value, dag: DagValue(dag.add_target_constant_node(
    value.node.value_types[0], ConstantInt(get_bits_sext(value.node.value.value, 12), value.node.value.ty)), 0))


riscv_patterns = []


def def_pat_riscv(pattern, result, enabled=None):
    def_pat(pattern, result, riscv_patterns, enabled)

def_pat_riscv(extloadi8_(("src", addr)),
              LBU(("src", AddrMode_Imm12)))

def_pat_riscv(extloadi16_(("src", addr)),
              LHU(("src", AddrMode_Imm12)))

def_pat_riscv(extloadi32_(("src", addr)),
              LW(("src", AddrMode_Imm12)))

def_pat_riscv(load_(("src", addr)),
              LD(("src", AddrMode_Imm12)))

def_pat_riscv(truncstorei8_(("dst", GPR), ("src", addr)),
              SB(("dst", GPR), ("src", AddrMode_Imm12)))

def_pat_riscv(truncstorei16_(("dst", GPR), ("src", addr)),
              SH(("dst", GPR), ("src", AddrMode_Imm12)))

def_pat_riscv(truncstorei32_(("dst", GPR), ("src", addr)),
              SW(("dst", GPR), ("src", AddrMode_Imm12)))

def_pat_riscv(store_(("dst", GPR), ("src", addr)),
              SD(("dst", GPR), ("src", AddrMode_Imm12)))


# constant
def_pat_riscv(("imm", imm), ADDI(
    LUI(HI20(("imm", I32Imm))), LO12Sext(("imm", I32Imm))))

def_pat_riscv(add_(("rs", fi_addr), ("imm", imm)),
              ADDI(("rs", GPR), ("imm", I32Imm)))

def_pat_riscv(add_(("src1", GPR), ("imm", imm)),
              ADDI(("src1", GPR), ("imm", I32Imm)))
def_pat_riscv(add_(("src1", GPR), ("imm", timm)),
              ADDI(("src1", GPR), ("imm", I32Imm)))

def_pat_riscv(xor_(("src1", GPR), ("imm", imm)),
              XORI(("src1", GPR), ("imm", I32Imm)))
def_pat_riscv(xor_(("src1", GPR), ("imm", timm)),
              XORI(("src1", GPR), ("imm", I32Imm)))

# binary arithematic
def_pat_riscv(add_(("src1", GPR), ("src2", GPR)),
              ADD(("src1", GPR), ("src2", GPR)))
def_pat_riscv(sub_(("src1", GPR), ("src2", GPR)),
              SUB(("src1", GPR), ("src2", GPR)))
def_pat_riscv(and_(("src1", GPR), ("src2", GPR)),
              AND(("src1", GPR), ("src2", GPR)))
def_pat_riscv(or_(("src1", GPR), ("src2", GPR)),
              OR(("src1", GPR), ("src2", GPR)))
def_pat_riscv(xor_(("src1", GPR), ("src2", GPR)),
              XOR(("src1", GPR), ("src2", GPR)))
def_pat_riscv(sra_(("src1", GPR), ("src2", GPR)),
              SRA(("src1", GPR), ("src2", GPR)))
def_pat_riscv(srl_(("src1", GPR), ("src2", GPR)),
              SRL(("src1", GPR), ("src2", GPR)))
def_pat_riscv(shl_(("src1", GPR), ("src2", GPR)),
              SLL(("src1", GPR), ("src2", GPR)))

def_pat_riscv(riscv_sraw_(("src1", GPR), ("src2", GPR)),
              SRAW(("src1", GPR), ("src2", GPR)))
def_pat_riscv(riscv_srlw_(("src1", GPR), ("src2", GPR)),
              SRLW(("src1", GPR), ("src2", GPR)))
def_pat_riscv(riscv_sllw_(("src1", GPR), ("src2", GPR)),
              SLLW(("src1", GPR), ("src2", GPR)))

def_pat_riscv(mul_(("src1", GPR), ("src2", GPR)),
              MUL(("src1", GPR), ("src2", GPR)))

def_pat_riscv(sdiv_(("src1", GPR), ("src2", GPR)),
              DIV(("src1", GPR), ("src2", GPR)))

def_pat_riscv(udiv_(("src1", GPR), ("src2", GPR)),
              DIVU(("src1", GPR), ("src2", GPR)))

def_pat_riscv(srem_(("src1", GPR), ("src2", GPR)),
              REM(("src1", GPR), ("src2", GPR)))

def_pat_riscv(urem_(("src1", GPR), ("src2", GPR)),
              REMU(("src1", GPR), ("src2", GPR)))

# integer compare
def_pat_riscv(seteq_(("src1", GPR), ("src2", GPR)),
              SLTIU(XOR(("src1", GPR), ("src2", GPR)), 1))
def_pat_riscv(seteq_(("src1", GPR), ("imm", imm)),
              SLTIU(XORI(("src1", GPR), ("imm", I32Imm)), 1))

def_pat_riscv(setne_(("src1", GPR), ("src2", GPR)),
              SLTU(X0, XOR(("src1", GPR), ("src2", GPR))))

def_pat_riscv(setult_(("src1", GPR), ("src2", GPR)),
              SLTU(("src1", GPR), ("src2", GPR)))
def_pat_riscv(setlt_(("src1", GPR), ("src2", GPR)),
              SLT(("src1", GPR), ("src2", GPR)))

def_pat_riscv(setugt_(("src1", GPR), ("src2", GPR)),
              SLTU(("src2", GPR), ("src1", GPR)))
def_pat_riscv(setgt_(("src1", GPR), ("src2", GPR)),
              SLT(("src2", GPR), ("src1", GPR)))

# float compare
def_pat_riscv(seteq_(("src1", FPR32), ("src2", FPR32)),
              FEQ_S(("src1", FPR32), ("src2", FPR32)))
def_pat_riscv(setoeq_(("src1", FPR32), ("src2", FPR32)),
              FEQ_S(("src1", FPR32), ("src2", FPR32)))
              
def_pat_riscv(setolt_(("src1", FPR32), ("src2", FPR32)),
              FLT_S(("src1", FPR32), ("src2", FPR32)))
def_pat_riscv(setole_(("src1", FPR32), ("src2", FPR32)),
              FLE_S(("src1", FPR32), ("src2", FPR32)))

def_pat_riscv(setogt_(("src1", FPR32), ("src2", FPR32)),
              FLT_S(("src2", FPR32), ("src1", FPR32)))
def_pat_riscv(setoge_(("src1", FPR32), ("src2", FPR32)),
              FLE_S(("src2", FPR32), ("src1", FPR32)))

# double compare
def_pat_riscv(seteq_(("src1", FPR64), ("src2", FPR64)),
              FEQ_D(("src1", FPR64), ("src2", FPR64)))
def_pat_riscv(setoeq_(("src1", FPR64), ("src2", FPR64)),
              FEQ_D(("src1", FPR64), ("src2", FPR64)))

def_pat_riscv(setolt_(("src1", FPR64), ("src2", FPR64)),
              FLT_D(("src1", FPR64), ("src2", FPR64)))
def_pat_riscv(setole_(("src1", FPR64), ("src2", FPR64)),
              FLE_D(("src1", FPR64), ("src2", FPR64)))

def_pat_riscv(setogt_(("src1", FPR64), ("src2", FPR64)),
              FLT_D(("src2", FPR64), ("src1", FPR64)))
def_pat_riscv(setoge_(("src1", FPR64), ("src2", FPR64)),
              FLE_D(("src2", FPR64), ("src1", FPR64)))

# fcvt

def_pat_riscv(fp_to_sint_(("src", FPR32)),
              FCVT_W_S(("src", FPR32), 0b001),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(fp_to_uint_(("src", FPR32)),
              FCVT_WU_S(("src", FPR32), 0b001),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(sint_to_fp_(("src", GPR)),
              FCVT_S_W(("src", GPR), 0b111),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(uint_to_fp_(("src", GPR)),
              FCVT_S_WU(("src", GPR), 0b111),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(fp_to_sint_(("src", FPR64)),
              FCVT_W_D(("src", FPR64), 0b001),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(fp_to_uint_(("src", FPR64)),
              FCVT_WU_D(("src", FPR64), 0b001),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(sint_to_fp_(("src", GPR)),
              FCVT_D_W(("src", GPR)),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(uint_to_fp_(("src", GPR)),
              FCVT_D_WU(("src", GPR)),
              enabled=lambda target_info: target_info.hwmode == RV32)

def_pat_riscv(fp_to_sint_(("src", FPR32)),
              FCVT_L_S(("src", FPR32), 0b001))

def_pat_riscv(fp_to_uint_(("src", FPR32)),
              FCVT_LU_S(("src", FPR32), 0b001))

def_pat_riscv(fp_to_sint_(("src", FPR64)),
              FCVT_L_D(("src", FPR64), 0b001))

def_pat_riscv(fp_to_uint_(("src", FPR64)),
              FCVT_LU_D(("src", FPR64), 0b001))

def_pat_riscv(f64_(sint_to_fp_(("src", GPR))),
              FCVT_D_W(("src", GPR)))

def_pat_riscv(f64_(uint_to_fp_(("src", GPR))),
              FCVT_D_WU(("src", GPR)))

def_pat_riscv(fp_round_(("src", FPR64)),
              FCVT_S_D(("src", FPR64), 0b001))

def_pat_riscv(fp_extend_(("src", FPR32)),
              FCVT_D_S(("src", FPR32)))

def_pat_riscv(bitconvert_(("src", FPR64)),
              FMV_X_D(("src", GPR)))

def_pat_riscv(f64_(bitconvert_(("src", GPR))),
              FMV_D_X(("src", GPR)))

# branch, call
def_pat_riscv(brcond_(("cond", GPR), ("imm", bb)),
              BNE(("cond", GPR), X0, ("imm", BrTarget8)))

def_pat_riscv(riscv_call_(("func", tglobaladdr_)),
              PseudoCALL(("func", CallSymbol)))

def_pat_riscv(riscv_call_(("func", tglobaltlsaddr_)),
              PseudoCALL(("func", CallSymbol)))

def_pat_riscv(riscv_call_(("func", texternalsym_)),
              PseudoCALL(("func", CallSymbol)))
