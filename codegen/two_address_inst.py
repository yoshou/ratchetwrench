#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.passes import *
from codegen.mir import *


class TwoAddressInst(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def get_tied_to_def_operand(self, inst: MachineInstruction, op_idx):
        operand = inst.operands[op_idx]
        if not operand.is_reg or operand.is_def or not operand.is_tied:
            return -1

        return operand.tied_to

    def collect_tied_pairs(self, inst, pairs):
        for idx, operand in enumerate(inst.operands):
            tied_to = self.get_tied_to_def_operand(inst, idx)
            if tied_to < 0:
                continue

            pairs.append((idx, tied_to))

    def process_machine_function(self, mfunc: MachineFunction):
        inst_info = mfunc.target_info.get_inst_info()
        for bb in mfunc.bbs:
            for inst in list(bb.insts):
                pairs = []
                self.collect_tied_pairs(inst, pairs)

                copied = set()
                for pair in pairs:
                    use_idx, def_idx = pair

                    src_reg = inst.operands[use_idx].reg
                    dst_reg = inst.operands[def_idx].reg

                    if mfunc.reg_info.has_one_use(src_reg):
                        inst.comment = "One use"

                    if use_idx not in copied:
                        minst = MachineInstruction(TargetDagOps.COPY)

                        minst.add_reg(dst_reg, RegState.Define)
                        minst.add_reg(src_reg, inst.operands[use_idx].flags)

                        minst.insert_before(inst)

                        copied.add(use_idx)

                        minst.comment = "Copy Two Addr"

                    inst.operands[use_idx].reg = inst.operands[def_idx].reg
                    inst.operands[use_idx].is_kill = False

                if inst.opcode == TargetDagOps.INSERT_SUBREG:
                    # From % reg = INSERT_SUBREG % reg, % subreg, subidx
                    # To % reg: subidx = COPY % subreg
                    subreg_idx = inst.operands[4]
                    inst.remove_operand(4)
                    inst.remove_operand(3)
                    inst.remove_operand(1)
                    inst.operands[0].subreg = subreg_idx.val
                    inst.opcode = TargetDagOps.COPY
