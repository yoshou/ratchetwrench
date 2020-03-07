#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, auto
from codegen.passes import *
from codegen.mir import *
from codegen.spec import *

class LinearScanRegisterAllocation(MachineFunctionPass):
    """
        A global register allocation.
    """

    def __init__(self):
        super().__init__()


    def expire_old_intervals(self, live_range):
        for rnge, regs in list(self.active_regs.items()):
            if rnge.end >= live_range.start:
                return

            self.active_regs.pop(rnge)
            for reg in regs:
                self.used_reg_units.remove(reg)
            

    def allocate(self, live_ranges):
        for live_range in live_ranges:
            self.expire_old_intervals(live_range)


    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc

        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.register_info = mfunc.target_info.get_register_info()

        self.used_reg_units = set()

        self.active_regs = {}

        live_ranges = list(sorted(self.mfunc.live_ranges.values(), key=lambda range: range.start))
        self.allocate(live_ranges)



