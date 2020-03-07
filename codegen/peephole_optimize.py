#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *


class PeepholeOptimize(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_instruction(self, inst: MachineInstruction):
        if hasattr(inst.opcode.value, "is_compare") and inst.opcode.value.is_compare:
            self.target_inst_info.optimize_compare_inst(self.mfunc, inst)

    def process_basicblock(self, bb: MachineBasicBlock):
        for inst in bb.insts:
            self.process_instruction(inst)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

        for bb in mfunc.bbs:
            self.process_basicblock(bb)
