#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.passes import *
from rachetwrench.codegen.mir import *
from io import StringIO
import sys
from rachetwrench.ir.values import Module
from rachetwrench.ir.printer import print_module


class ModulePrinter(FunctionPass):
    def __init__(self, output=sys.stdout):
        super().__init__()

        self.output = output

    def process_function(self, func):
        pass

    def process_module(self, module: Module):
        super().process_module(module)

        self.output.write(print_module(module))
