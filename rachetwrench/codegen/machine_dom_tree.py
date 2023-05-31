#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.mir import *
from rachetwrench.codegen.passes import *


class MachineDomTree(MachineFunctionPass):
    def __init__(self, mfunc):
        super().__init__()

    def process_instruction(self, inst: MachineInstruction):
        pass

    def process_basicblock(self, bb: MachineBasicBlock):
        for inst in bb.insts:
            self.process_instruction(inst)

    def process_function(self, func: MachineFunction):
        for bb in func.bbs:
            self.process_basicblock(bb)
