#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *


def dfs_bb(bb: MachineBasicBlock, action, visited):
    if bb in visited:
        return

    visited.add(bb)

    action(bb)

    for succ in bb.successors:
        dfs_bb(succ, action, visited)


class ExpandPseudosPostRA(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def lower_copy(self, inst: MachineInstruction):
        dst_op = inst.operands[0]
        src_op = inst.operands[1]

        if dst_op.reg == src_op.reg:
            inst.remove()
            return True

        src_op.subst_phys_reg(self.target_reg_info)
        dst_op.subst_phys_reg(self.target_reg_info)

        self.target_inst_info.copy_phys_reg(
            src_op.reg, dst_op.reg, src_op.is_kill, inst)
        inst.remove()
        return True

    def lower_subreg_to_reg(self, inst: MachineInstruction):
        dst_reg = inst.operands[0]
        src_op = inst.operands[2]
        subreg_idx = inst.operands[3]

        reg_info = self.target_reg_info

        def get_subclass_with_subreg(regclass, subreg):
            if isinstance(subreg, ComposedSubRegDescription):
                regclass = get_subclass_with_subreg(
                    regclass, subreg.subreg_a)
                subreg = subreg.subreg_b

            return regclass.subclass_and_subregs[subreg]

        subreg_idx = subregs[subreg_idx.val]

        dst_subreg_spec = None
        for subreg in dst_reg.reg.spec.subregs:
            if subreg_idx == subreg.idx:
                dst_subreg_spec = subreg.reg

        assert(dst_subreg_spec)

        dst_subreg = MachineRegister(dst_subreg_spec)

        self.target_inst_info.copy_phys_reg(
            src_op.reg, dst_subreg, src_op.is_kill, inst)
        inst.remove()

    def process_instruction(self, inst: MachineInstruction):
        if inst.opcode == TargetDagOps.COPY:
            self.lower_copy(inst)
        elif inst.opcode == TargetDagOps.SUBREG_TO_REG:
            self.lower_subreg_to_reg(inst)

        self.target_inst_info.expand_post_ra_pseudo(inst)

    def process_basicblock(self, bb: MachineBasicBlock):
        for inst in list(bb.insts):
            self.process_instruction(inst)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.target_reg_info = mfunc.target_info.get_register_info()

        for bb in mfunc.bbs:
            self.process_basicblock(bb)
