#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.passes import *
from rachetwrench.codegen.mir import *


class VirtualRegisterRewriter(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc: MachineFunction):
        reg_info = mfunc.target_info.get_register_info()
        for bb in mfunc.bbs:
            for inst in list(bb.insts):
                for operand in inst.operands:
                    if not operand.is_reg:
                        continue

                    subreg = operand.subreg

                    if subreg is not None:
                        operand.reg = MachineRegister(reg_info.get_subreg(
                            operand.reg.spec, subreg))
                        operand.subreg = None
