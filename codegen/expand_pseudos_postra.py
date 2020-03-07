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
            
        self.target_inst_info.copy_phys_reg(src_op.reg, dst_op.reg, src_op.is_kill, inst)
        inst.remove()
        return True

    def process_instruction(self, inst: MachineInstruction):
        if inst.opcode == TargetDagOps.COPY:
            self.lower_copy(inst)
            
        self.target_inst_info.expand_post_ra_pseudo(inst)

    def process_basicblock(self, bb: MachineBasicBlock):
        for inst in list(bb.insts):
            self.process_instruction(inst)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

        for bb in mfunc.bbs:
            self.process_basicblock(bb)
            
