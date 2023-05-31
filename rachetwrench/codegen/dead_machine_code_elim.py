#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.mir import *
from rachetwrench.codegen.spec import *
from rachetwrench.codegen.passes import *


class DeadMachineCodeElim(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def is_dead_inst(self, inst: MachineInstruction):
        reg_info = inst.mbb.func.reg_info

        # TODO: Need a flag to indicate side effects.
        has_def = False
        for operand in inst.operands:
            if not operand.is_reg or not operand.is_def:
                continue

            if operand.is_phys:
                return False

            has_def = True

            if not reg_info.is_use_empty(operand.reg):
                return False # The register is used by another operand.

        return has_def


    def process_instruction(self, inst: MachineInstruction):
        if self.is_dead_inst(inst):
            inst.remove()

    def process_basicblock(self, bb: MachineBasicBlock):
        for inst in bb.insts:
            self.process_instruction(inst)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

        for bb in mfunc.bbs:
            self.process_basicblock(bb)
