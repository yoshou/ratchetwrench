#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.passes import *


def iter_inst_range(begin: MachineInstruction, end: MachineInstruction):
    bb = begin.mbb
    idx = bb.insts.index(begin)

    while idx < len(bb.insts) and bb.insts[idx] != end:
        yield bb.insts[idx]

        idx += 1


class MachineCopyProp(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def track_copy(self, inst):
        from codegen.spec import iter_reg_units

        dst_reg = inst.operands[0]
        src_reg = inst.operands[1]

        for reg_unit in iter_reg_units(dst_reg.reg.spec):
            self.copies[reg_unit] = inst

    def track_clear(self):
        self.copies = {}

    def track_has_any_copies(self):
        return len(self.copies) == 0

    def find_copy(self, inst, reg):
        from codegen.spec import iter_reg_units

        reg_unit = next(iter_reg_units(reg.spec))

        if reg_unit not in self.copies:
            return None

        copy = self.copies[reg_unit]

        for inter_inst in iter_inst_range(copy, inst):
            for operand in inter_inst.operands:
                pass

        return copy

    def find_backward_copy(self, inst, reg):
        from codegen.spec import iter_reg_units

        reg_unit = next(iter_reg_units(reg.spec))

        if reg_unit in self.copies:
            return self.copies[reg_unit]

        return None

    def forward_uses(self, inst):
        for operand in inst.operands:
            if not operand.is_reg or operand.is_def or operand.is_tied or operand.is_undef or operand.is_implicit:
                continue

            if not operand.is_renamable:
                continue

            copy = self.find_copy(inst, operand.reg)

            if not copy:
                continue

            copy_dst_reg = copy.operands[0]
            copy_src_reg = copy.operands[1]

            operand.reg = copy_src_reg.reg
            operand.is_renamable = copy_src_reg.is_renamable

            for inter_inst in iter_inst_range(copy, inst):
                for inter_operand in inter_inst.operands:
                    if not inter_operand.is_reg or not inter_operand.is_use or not inter_operand.is_kill:
                        continue

                    if inter_operand.reg == operand.reg:
                        operand.is_kill = False

    def forward_cp_bb(self, bb: MachineBasicBlock):
        from codegen.mir_emitter import TargetDagOps

        self.track_clear()

        for inst in bb.insts:
            if inst.opcode == TargetDagOps.COPY:
                src_reg = inst.operands[0].reg
                dst_reg = inst.operands[1].reg

                self.track_copy(inst)
                continue

            self.forward_uses(inst)

    def propagate_defs(self, inst: MachineInstruction):
        if not self.track_has_any_copies():
            return

        for operand in inst.operands:
            if not operand.is_reg or not operand.is_def or operand.is_tied or operand.is_undef or operand.is_implicit:
                continue

            if not operand.is_renamable:
                continue

            copy = self.find_backward_copy(inst, operand.reg)

            if not copy:
                continue

            raise NotImplementedError()

    def backward_cp_bb(self, bb: MachineBasicBlock):
        from codegen.mir_emitter import TargetDagOps

        self.track_clear()

        for inst in reversed(bb.insts):
            if inst.opcode == TargetDagOps.COPY:
                self.track_copy(inst)
                continue

            self.propagate_defs(inst)

    def process_machine_function(self, func: MachineFunction):
        for bb in func.bbs:
            self.backward_cp_bb(bb)
            self.forward_cp_bb(bb)
