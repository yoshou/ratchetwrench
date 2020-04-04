#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *


class ARMConstantIsland(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        data_layout = self.mfunc.func_info.func.module.data_layout

        cp = mfunc.constant_pool

        mbb = MachineBasicBlock()
        mfunc.append_bb(mbb)

        for i, entry in enumerate(cp.constants):
            from codegen.arm_def import ARMMachineOps

            size, align = data_layout.get_type_size_in_bits(entry.value.ty)

            minst = MachineInstruction(ARMMachineOps.CPEntry)
            minst.add_imm(i)
            minst.add_constant_pool_index(i)
            minst.add_imm(size)
            mbb.append_inst(minst)
