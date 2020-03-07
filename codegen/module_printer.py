#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.passes import *
from codegen.mir import *
from io import StringIO
import sys
from ir.values import Module
from ir.printer import print_module


class ModulePrinter(FunctionPass):
    def __init__(self, output=sys.stdout):
        super().__init__()

        self.output = output

    def process_function(self, func):
        pass

    def process_module(self, module: Module):
        super().process_module(module)

        self.output.write(print_module(module))
