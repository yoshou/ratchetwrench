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


GPR = def_riscv_regclass("GPR", [ValueType.I32], 32, [
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


class RISCVDagOp(DagOp):
    def __init__(self, name):
        super().__init__(name, "riscv_")


class RISCVDagOps(Enum):
    LUI = RISCVDagOp("lui")
    ADDI = RISCVDagOp("addi")

    SUB = RISCVDagOp("sub")
    CMP = RISCVDagOp("cmp")
    SETCC = RISCVDagOp("setcc")
    BRCOND = RISCVDagOp("brcond")

    VDUP = RISCVDagOp("vdup")

    SHUFP = RISCVDagOp("shufp")
    UNPCKL = RISCVDagOp("unpckl")
    UNPCKH = RISCVDagOp("unpckh")

    MOVSS = RISCVDagOp("movss")
    MOVSD = RISCVDagOp("movsd")

    CMPFP = RISCVDagOp("cmpfp")
    FMSTAT = RISCVDagOp("fmstat")

    CALL = RISCVDagOp("call")
    RETURN = RISCVDagOp("return")

    WRAPPER = RISCVDagOp("wrapper")
    WRAPPER_PIC = RISCVDagOp("wrapper_pic")
    WRAPPER_JT = RISCVDagOp("wrapper_jt")

    PIC_ADD = RISCVDagOp("pic_add")


riscv_brcond_ = NodePatternMatcherGen(RISCVDagOps.BRCOND)
riscv_sub_ = NodePatternMatcherGen(RISCVDagOps.SUB)
riscv_cmp_ = NodePatternMatcherGen(RISCVDagOps.CMP)
riscv_cmpfp_ = NodePatternMatcherGen(RISCVDagOps.CMPFP)
riscv_fmstat_ = NodePatternMatcherGen(RISCVDagOps.FMSTAT)
riscv_setcc_ = NodePatternMatcherGen(RISCVDagOps.SETCC)
riscv_movss_ = NodePatternMatcherGen(RISCVDagOps.MOVSS)
riscv_shufp_ = NodePatternMatcherGen(RISCVDagOps.SHUFP)
riscv_let_ = NodePatternMatcherGen(RISCVDagOps.RETURN)
riscv_wrapper_ = NodePatternMatcherGen(RISCVDagOps.WRAPPER)
riscv_pic_add_ = NodePatternMatcherGen(RISCVDagOps.PIC_ADD)


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

        return idx + 1, MatcherResult([base, offset])

    if value.node.opcode == VirtualDagOps.ADD:
        value1 = value.node.operands[0]
        value2 = value.node.operands[1]

        if value1.node.opcode == VirtualDagOps.CONSTANT:
            base = value2
            offset = DagValue(dag.add_target_constant_node(
                value1.ty, value1.node.value), 0)

            return idx + 1, MatcherResult([base, offset])
        elif value2.node.opcode == VirtualDagOps.CONSTANT:
            base = value1
            offset = DagValue(dag.add_target_constant_node(
                value2.ty, value2.node.value), 0)

            return idx + 1, MatcherResult([base, offset])

    if value.node.opcode == VirtualDagOps.SUB:
        if value2.node.opcode == VirtualDagOps.CONSTANT:
            raise NotImplementedError()

    # if value.node.opcode == RISCVDagOps.WRAPPER:
    #     base = value.node.operands[0]
    #     offset = DagValue(dag.add_target_constant_node(
    #         value.ty, 0), 0)

    #     return idx + 1, MatcherResult([base, offset])

    # only base.
    base = value

    assert(base.node.opcode != VirtualDagOps.TARGET_CONSTANT)

    offset = DagValue(dag.add_target_constant_node(
        value.ty, 0), 0)

    return idx + 1, MatcherResult([base, offset])


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

        return idx + 1, MatcherResult([base, offset])

    if value.node.opcode == VirtualDagOps.ADD:
        value1 = value.node.operands[0]
        value2 = value.node.operands[1]

        if value1.node.opcode == VirtualDagOps.CONSTANT:
            base = value2

            assert(value1.node.value.value % 4 == 0)
            offset = DagValue(dag.add_target_constant_node(
                value1.ty, ConstantInt(value1.node.value.value >> 2, value1.node.value.ty)), 0)

            return idx + 1, MatcherResult([base, offset])
        elif value2.node.opcode == VirtualDagOps.CONSTANT:
            base = value1

            assert(value2.node.value.value % 4 == 0)
            offset = DagValue(dag.add_target_constant_node(
                value2.ty, ConstantInt(value2.node.value.value >> 2, value2.node.value.ty)), 0)

            return idx + 1, MatcherResult([base, offset])

    if value.node.opcode == VirtualDagOps.SUB:
        if value2.node.opcode == VirtualDagOps.CONSTANT:
            raise NotImplementedError()

    # if value.node.opcode == RISCVDagOps.WRAPPER:
    #     base = value.node.operands[0]
    #     offset = DagValue(dag.add_target_constant_node(
    #         value.ty, 0), 0)

    #     return idx + 1, MatcherResult([base, offset])

    # only base.
    base = value

    assert(base.node.opcode != VirtualDagOps.TARGET_CONSTANT)

    offset = DagValue(dag.add_target_constant_node(
        value.ty, 0), 0)

    return idx + 1, MatcherResult([base, offset])


addrmode5 = ComplexOperandMatcher(match_addrmode5)


# addrmode 6
def match_addrmode6(node, values, idx, dag: Dag):
    from codegen.dag import VirtualDagOps, DagValue
    from codegen.types import ValueType

    value = values[idx]

    addr = value

    align = DagValue(dag.add_target_constant_node(
        value.ty, 1), 0)  # TODO: Need alignment information

    return idx + 1, MatcherResult([addr, align])


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

    return idx + 1, MatcherResult([target_value])


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


class RISCVMachineOps:

    @classmethod
    def insts(cls):
        for member, value in cls.__dict__.items():
            if isinstance(value, MachineInstructionDef):
                yield value

    LW = def_inst(
        "lw",
        outs=[("dst", GPR)],
        ins=[("src", AddrMode_Imm12)],
        patterns=[set_(("dst", GPR), load_(("src", addr)))]
    )

    LFW = def_inst(
        "lfw",
        outs=[("dst", FPR32)],
        ins=[("src", AddrMode_Imm12)],
        patterns=[set_(("dst", FPR32), load_(("src", addr)))]
    )

    LUI = def_inst(
        "lui",
        outs=[("dst", GPR)],
        ins=[("src", imm)],
        patterns=[]
    )

    ADDI = def_inst(
        "addi",
        outs=[("dst", GPR)],
        ins=[("src1", GPR), ("src2", imm)],
        patterns=[]
    )

    FADD_S = def_inst(
        "fadd_s",
        outs=[("dst", FPR32)],
        ins=[("src1", FPR32), ("src2", FPR32)],
        patterns=[set_(("dst", FPR32), fadd_(
            ("src1", FPR32), ("src2", FPR32)))]
    )

    BR = def_inst(
        "b",
        outs=[],
        ins=[("dst", BrTarget8)],
        patterns=[br_(("dst", bb))],
        is_terminator=True
    )

    # STRi12 = def_inst(
    #     "str_i12",
    #     outs=[],
    #     ins=[("src", GPR), ("dst", AddrMode_Imm12)],
    #     patterns=[store_(("src", GPR), ("dst", addrmode_imm12))]
    # )

    # MOVsi = def_inst(
    #     "mov_si",
    #     outs=[("dst", GPR)],
    #     ins=[("src", SORegImm)],
    #     patterns=[set_(("dst", GPR), ("src", so_reg_imm))]
    # )

    # MOVi = def_inst(
    #     "mov_i",
    #     outs=[("dst", GPR)],
    #     ins=[("src", I32Imm)],
    #     patterns=[set_(("dst", GPR), ("src", mod_imm))]
    # )

    # MOVi16 = def_inst(
    #     "mov_i16",
    #     outs=[("dst", GPR)],
    #     ins=[("src", I32Imm)],
    #     patterns=[set_(("dst", GPR), ("src", imm0_65535))]
    # )

    # MOVTi16 = def_inst(
    #     "movt_i16",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("imm", I32Imm)],
    #     constraints=[Constraint("dst", "src")]
    # )

    # MOVr = def_inst(
    #     "mov_r",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR)]
    # )

    # STR_PRE_IMM = def_inst(
    #     "str_pre_imm",
    #     outs=[("rn_web", GPR)],
    #     ins=[("rt", GPR), ("addr", AddrMode_Imm12)]
    # )

    # LDR_PRE_IMM = def_inst(
    #     "ldr_pre_imm",
    #     outs=[("rt", GPR), ("rn_web", GPR)],
    #     ins=[("addr", AddrMode_Imm12)]
    # )

    # STR_POST_IMM = def_inst(
    #     "str_post_imm",
    #     outs=[("rn_web", GPR)],
    #     ins=[("rt", GPR), ("addr", AddrMode_Imm12)]
    # )

    # LDR_POST_IMM = def_inst(
    #     "ldr_post_imm",
    #     outs=[("rt", GPR), ("rn_web", GPR)],
    #     ins=[("addr", AddrMode_Imm12)]
    # )

    # # load/store multiple and increment after
    # LDMIA = def_inst(
    #     "ldm_ia",
    #     outs=[],
    #     ins=[("src", GPR), ("regs", reglist)]
    # )

    # STMIA = def_inst(
    #     "stm_ia",
    #     outs=[],
    #     ins=[("src", GPR), ("regs", reglist)]
    # )

    # LDMIA_UPD = def_inst(
    #     "ldm_ia_upd",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("regs", reglist)],
    #     constraints=[Constraint("dst", "src")]
    # )

    # STMIA_UPD = def_inst(
    #     "stm_ia_upd",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("regs", reglist)],
    #     constraints=[Constraint("dst", "src")]
    # )

    # # load/store multiple and decrement before
    # LDMDB = def_inst(
    #     "ldm_db",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("regs", reglist)]
    # )

    # STMDB = def_inst(
    #     "stm_db",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("regs", reglist)]
    # )

    # LDMDB_UPD = def_inst(
    #     "ldm_db_upd",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("regs", reglist)],
    #     constraints=[Constraint("dst", "src")]
    # )

    # STMDB_UPD = def_inst(
    #     "stm_db_upd",
    #     outs=[("dst", GPR)],
    #     ins=[("src", GPR), ("regs", reglist)],
    #     constraints=[Constraint("dst", "src")]
    # )

    # ADDri = def_inst(
    #     "add_ri",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), add_(("src1", GPR), ("src2", mod_imm)))]
    # )

    # ADDrsi = def_inst(
    #     "add_rsi",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), add_(
    #         ("src1", GPR), ("src2", so_reg_imm)))]
    # )

    # ADDrr = def_inst(
    #     "add_rr",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", GPR)],
    #     patterns=[set_(("dst", GPR), add_(("src1", GPR), ("src2", GPR)))]
    # )

    # SUBri = def_inst(
    #     "sub_ri",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), riscv_sub_(
    #         ("src1", GPR), ("src2", mod_imm)))]
    # )

    # SUBrsi = def_inst(
    #     "sub_rsi",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), riscv_sub_(
    #         ("src1", GPR), ("src2", so_reg_imm)))]
    # )

    # SUBrr = def_inst(
    #     "sub_rr",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", GPR)],
    #     patterns=[set_(("dst", GPR), riscv_sub_(("src1", GPR), ("src2", GPR)))]
    # )

    # ANDri = def_inst(
    #     "and_ri",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), and_(("src1", GPR), ("src2", mod_imm)))]
    # )

    # ANDrsi = def_inst(
    #     "and_rsi",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), and_(
    #         ("src1", GPR), ("src2", so_reg_imm)))]
    # )

    # ANDrr = def_inst(
    #     "and_rr",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", GPR)],
    #     patterns=[set_(("dst", GPR), and_(("src1", GPR), ("src2", GPR)))]
    # )

    # ORri = def_inst(
    #     "or_ri",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), or_(("src1", GPR), ("src2", mod_imm)))]
    # )

    # ORrsi = def_inst(
    #     "or_rsi",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), or_(("src1", GPR), ("src2", so_reg_imm)))]
    # )

    # ORrr = def_inst(
    #     "or_rr",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", GPR)],
    #     patterns=[set_(("dst", GPR), or_(("src1", GPR), ("src2", GPR)))]
    # )

    # XORri = def_inst(
    #     "xor_ri",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), xor_(("src1", GPR), ("src2", mod_imm)))]
    # )

    # XORrsi = def_inst(
    #     "xor_rsi",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), xor_(
    #         ("src1", GPR), ("src2", so_reg_imm)))]
    # )

    # XORrr = def_inst(
    #     "xor_rr",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", GPR)],
    #     patterns=[set_(("dst", GPR), xor_(("src1", GPR), ("src2", GPR)))]
    # )

    # CMPri = def_inst(
    #     "cmp_ri",
    #     outs=[],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[riscv_cmp_(("src1", GPR), ("src2", mod_imm))],
    #     is_terminator=True
    # )

    # CMPrsi = def_inst(
    #     "cmp_rsi",
    #     outs=[],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[riscv_cmp_(("src1", GPR), ("src2", so_reg_imm))],
    #     is_terminator=True
    # )

    # CMPrr = def_inst(
    #     "cmp_rr",
    #     outs=[],
    #     ins=[("src1", GPR), ("src2", GPR)],
    #     patterns=[riscv_cmp_(("src1", GPR), ("src2", GPR))],
    #     is_terminator=True
    # )

    # # vfp instructions
    # VMOVS = def_inst(
    #     "vmovs",
    #     outs=[("dst", SPR)],
    #     ins=[("src", SPR)]
    # )

    # VMOVSR = def_inst(
    #     "vmovsr",
    #     outs=[("dst", SPR)],
    #     ins=[("src", GPR)],
    #     patterns=[set_(("dst", SPR), bitconvert_(("src", GPR)))]
    # )

    # VLDRS = def_inst(
    #     "vldrs",
    #     outs=[("dst", SPR)],
    #     ins=[("src", AddrMode5)],
    #     patterns=[set_(("dst", SPR), load_(("src", addrmode5)))]
    # )

    # VSTRS = def_inst(
    #     "vstrs",
    #     outs=[],
    #     ins=[("src", SPR), ("dst", AddrMode5)],
    #     patterns=[store_(("src", SPR), ("dst", addrmode5))]
    # )

    # VADDS = def_inst(
    #     "vadds",
    #     outs=[("dst", SPR)],
    #     ins=[("src1", SPR), ("src2", SPR)],
    #     patterns=[set_(("dst", SPR), fadd_(("src1", SPR), ("src2", SPR)))]
    # )

    # VSUBS = def_inst(
    #     "vsubs",
    #     outs=[("dst", SPR)],
    #     ins=[("src1", SPR), ("src2", SPR)],
    #     patterns=[set_(("dst", SPR), fsub_(("src1", SPR), ("src2", SPR)))]
    # )

    # VMULS = def_inst(
    #     "vmuls",
    #     outs=[("dst", SPR)],
    #     ins=[("src1", SPR), ("src2", SPR)],
    #     patterns=[set_(("dst", SPR), fmul_(("src1", SPR), ("src2", SPR)))]
    # )

    # VDIVS = def_inst(
    #     "vdivs",
    #     outs=[("dst", SPR)],
    #     ins=[("src1", SPR), ("src2", SPR)],
    #     patterns=[set_(("dst", SPR), fdiv_(("src1", SPR), ("src2", SPR)))]
    # )

    # VCMPS = def_inst(
    #     "vcmps",
    #     outs=[],
    #     ins=[("src1", SPR), ("src2", SPR)],
    #     patterns=[riscv_cmpfp_(("src1", SPR), ("src2", SPR))]
    # )

    # VLDRD = def_inst(
    #     "vldrd",
    #     outs=[("dst", DPR)],
    #     ins=[("src", AddrMode5)],
    #     patterns=[set_(("dst", DPR), load_(("src", addrmode5)))]
    # )

    # VSTRD = def_inst(
    #     "vstrd",
    #     outs=[],
    #     ins=[("src", DPR), ("dst", AddrMode5)],
    #     patterns=[store_(("src", DPR), ("dst", addrmode5))]
    # )

    # FMSTAT = def_inst(
    #     "fmstat",
    #     outs=[],
    #     ins=[],
    #     patterns=[riscv_fmstat_()]
    # )

    # # neon instructions
    # VLD1q64 = def_inst(
    #     "vld1_q64",
    #     outs=[("dst", QPR)],
    #     ins=[("src", AddrMode6)],
    #     patterns=[set_(("dst", QPR), load_(("src", addrmode6)))]
    # )
    # VST1q64 = def_inst(
    #     "vst1_q64",
    #     outs=[],
    #     ins=[("src", QPR), ("dst", AddrMode6)],
    #     patterns=[store_(("src", QPR), ("dst", addrmode6))]
    # )

    # VADDfq = def_inst(
    #     "vadd_fq",
    #     outs=[("dst", QPR)],
    #     ins=[("src1", QPR), ("src2", QPR)],
    #     patterns=[set_(("dst", QPR), fadd_(("src1", QPR), ("src2", QPR)))]
    # )

    # VSUBfq = def_inst(
    #     "vsub_fq",
    #     outs=[("dst", QPR)],
    #     ins=[("src1", QPR), ("src2", QPR)],
    #     patterns=[set_(("dst", QPR), fsub_(("src1", QPR), ("src2", QPR)))]
    # )

    # VMULfq = def_inst(
    #     "vmul_fq",
    #     outs=[("dst", QPR)],
    #     ins=[("src1", QPR), ("src2", QPR)],
    #     patterns=[set_(("dst", QPR), fmul_(("src1", QPR), ("src2", QPR)))]
    # )

    # VORRq = def_inst(
    #     "vorr_q",
    #     outs=[("dst", QPR)],
    #     ins=[("src1", QPR), ("src2", QPR)],
    #     patterns=[set_(("dst", QPR), or_(("src1", QPR), ("src2", QPR)))]
    # )

    # # # jcc
    # Bcc = def_inst(
    #     "bcc",
    #     outs=[],
    #     ins=[("dst", BrTarget8), ("cond", I32Imm)],
    #     patterns=[riscv_brcond_(("dst", bb), ("cond", timm))],
    #     is_terminator=True
    # )

    # B = def_inst(
    #     "b",
    #     outs=[],
    #     ins=[("dst", BrTarget8)],
    #     patterns=[br_(("dst", bb))],
    #     is_terminator=True
    # )

    # BL = def_inst(
    #     "bl",
    #     outs=[],
    #     ins=[("dst", riscv_bl_target)],
    #     is_call=True
    # )

    # MOVPCLR = def_inst(
    #     "movpclr",
    #     outs=[],
    #     ins=[],
    #     patterns=[(riscv_let_())],
    #     is_terminator=True
    # )

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

    # MOVCCr = def_inst(
    #     "movcc_r",
    #     outs=[("dst", GPR)],
    #     ins=[("false", GPR), ("true", GPR), ("p", cmovpred)],
    #     constraints=[Constraint("dst", "false")],
    #     is_terminator=True
    # )

    # MOVCCi16 = def_inst(
    #     "movcc_i16",
    #     outs=[("dst", GPR)],
    #     ins=[("false", GPR), ("true", I32Imm), ("p", cmovpred)],
    #     constraints=[Constraint("dst", "false")],
    #     is_terminator=True
    # )

    # MOVCCi = def_inst(
    #     "movcc_i",
    #     outs=[("dst", GPR)],
    #     ins=[("false", GPR), ("true", I32Imm), ("p", cmovpred)],
    #     constraints=[Constraint("dst", "false")],
    #     is_terminator=True
    # )

    # VDUPLN32q = def_inst(
    #     "udup_ln32q",
    #     outs=[("dst", QPR)],
    #     ins=[("src", SPR), ("lane", VectorIndex32)]
    # )

    # LDRLIT_ga_abs = def_inst(
    #     "LDRLIT_ga_abs",
    #     outs=[("dst", GPR)],
    #     ins=[("src", I32Imm)],
    #     # patterns=[set_(("dst", GPR), riscv_wrapper_(("src", tglobaladdr_)))]
    # )

    # PIC_ADD = def_inst(
    #     "pic_add",
    #     outs=[("dst", GPR)],
    #     ins=[("src1", GPR), ("src2", I32Imm)],
    #     patterns=[set_(("dst", GPR), riscv_pic_add_(
    #         ("src1", GPR), ("src2", timm)))]
    # )

    # MOVi32imm = def_inst(
    #     "mov_i32imm",
    #     outs=[("dst", GPR)],
    #     ins=[("src", I32Imm)],
    #     patterns=[set_(("dst", GPR), riscv_wrapper_(("src", tglobaladdr_)))]
    # )

    # LEApcrel = def_inst(
    #     "LEApcrel",
    #     outs=[("dst", GPR)],
    #     ins=[("label", I32Imm)],
    #     patterns=[set_(("dst", GPR), riscv_wrapper_(("label", tconstpool_)))]
    # )

    # CPEntry = def_inst(
    #     "CPEntry",
    #     outs=[],
    #     ins=[("instid", I32Imm), ("cpidx", I32Imm), ("size", I32Imm)]
    # )


ADDI = def_inst_node_(RISCVMachineOps.ADDI)
LUI = def_inst_node_(RISCVMachineOps.LUI)


from codegen.dag import DagValue


HI20 = def_node_xform_(I32Imm, lambda node, dag: DagValue(dag.add_target_constant_node(
    node.value_types[0], node.value), 0))


Lo12Sext = def_node_xform_(I32Imm, lambda node, dag: DagValue(dag.add_target_constant_node(
    node.value_types[0], node.value), 0))


riscv_patterns = []

def def_pat_riscv(pattern, result):
    def_pat(pattern, result, riscv_patterns)

def_pat_riscv(("imm", imm), ADDI(LUI(HI20(("imm", I32Imm))), Lo12Sext(("imm", I32Imm))))
