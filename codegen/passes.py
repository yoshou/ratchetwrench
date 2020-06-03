#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Pass:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass

    def process_function(self, func):
        pass

    def process_module(self, module):
        self.module = module

        self.initialize()

        funcs = module.funcs
        for func in funcs.values():
            self.process_function(func)

        self.finalize()


class FunctionPass(Pass):
    def __init__(self):
        super().__init__()

    def process_function(self, func):
        raise NotImplementedError()


class MachineFunctionPass(FunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc):
        raise NotImplementedError()

    def process_function(self, func):
        self.func = func

        if func.is_declaration:
            return

        mfunc = self.module.mfuncs[func]
        self.process_machine_function(mfunc)


class PassManager:
    def __init__(self):
        self.passes = []

    def run(self, ir):
        for p in self.passes:
            p.process_module(ir)
