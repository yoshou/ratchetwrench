#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.passes import *
from rachetwrench.codegen.mir import *
from io import StringIO
import sys

class MIRPrinting(MachineFunctionPass):
    def __init__(self, output = sys.stdout):
        super().__init__()

        self.mir_funcs = []
        self.output = output

    def process_machine_function(self, mfunc: MachineFunction):
        with StringIO() as f:
            mfunc.print(f)
            self.mir_funcs.append(f.getvalue())

    def process_module(self, module):
        super().process_module(module)

        for s in self.mir_funcs:
            self.output.write(s)


