#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *


class BranchFolderPass(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def optimize_block(self, bb: MachineBasicBlock):
        prev_bb = bb.func.bbs[bb.index - 1]
        analyzed, true_bb, false_bb, cond = self.target_inst_info.analyze_branch(
            prev_bb)

        if analyzed:
            if true_bb == bb and not false_bb:
                prev_bb.insts[-1].remove()

            if false_bb == bb:
                prev_bb.insts[-1].remove()

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

        for bb in mfunc.bbs:
            self.optimize_block(bb)
