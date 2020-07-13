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

        for reg_unit in iter_reg_units(src_reg.reg.spec):
            self.copies[reg_unit] = inst

    def track_invalidate_reg(self, reg):
        from codegen.spec import iter_reg_units

        regs_to_invalidate = set()

        for reg_unit in iter_reg_units(reg.spec):
            if reg_unit in self.copies:
                copy = self.copies[reg_unit]
                regs_to_invalidate.add(copy.operands[0].reg)
                regs_to_invalidate.add(copy.operands[1].reg)

        for reg_to_invalidate in regs_to_invalidate:
            for reg_unit in iter_reg_units(reg_to_invalidate.spec):
                self.copies.pop(reg_unit)

    def track_clear(self):
        self.copies = {}

    def track_has_any_copies(self):
        return len(self.copies) > 0

    def find_copy(self, inst, reg):
        from codegen.spec import iter_reg_units

        reg_unit = next(iter_reg_units(reg.spec))

        if reg_unit not in self.copies:
            return None

        copy = self.copies[reg_unit]

        dst_reg = copy.operands[0].reg
        src_reg = copy.operands[1].reg

        def clobbers_phys_reg(reg_mask, reg):
            return reg.spec in reg_mask

        for inter_inst in iter_inst_range(copy, inst):
            for operand in inter_inst.operands:
                if isinstance(operand, MORegisterMask):
                    if clobbers_phys_reg(operand.mask, src_reg) or clobbers_phys_reg(operand.mask, dst_reg):
                        return None

        return copy

    def find_backward_copy(self, inst, reg):
        from codegen.spec import iter_reg_units, iter_reg_aliases

        reg_unit = next(iter_reg_units(reg.spec))

        if reg_unit not in self.copies:
            return None

        copy = self.copies[reg_unit]

        if reg.spec != copy.operands[1].reg.spec:
            return None

        dst_reg = copy.operands[0].reg
        src_reg = copy.operands[1].reg

        def clobbers_phys_reg(reg_mask, reg):
            return reg.spec in reg_mask

        for inter_inst in iter_inst_range(inst.next_inst, copy):
            for operand in inter_inst.operands:
                if isinstance(operand, MORegisterMask):
                    if clobbers_phys_reg(operand.mask, src_reg) or clobbers_phys_reg(operand.mask, dst_reg):
                        return None

        return copy

    def is_forwardable_regclass_copy(self, copy_inst, inst, operand):
        copy_dst_reg = copy_inst.operands[0].reg
        regclass = self.target_reg_info.get_regclass_from_reg(
            copy_dst_reg.spec)

        return operand.reg.spec in regclass.regs

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

            if self.target_reg_info.is_reserved(copy_src_reg.reg.spec):
                continue

            if not self.is_forwardable_regclass_copy(copy, inst, operand):
                continue

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
                src_reg = inst.operands[1].reg
                dst_reg = inst.operands[0].reg

                self.track_copy(inst)
                continue

            self.forward_uses(inst)

    def is_backward_propagatable_copy(self, copy):
        copy_dst_reg = copy.operands[0]
        copy_src_reg = copy.operands[1]

        if self.target_reg_info.is_reserved(copy_dst_reg.reg.spec):
            return False

        if self.target_reg_info.is_reserved(copy_src_reg.reg.spec):
            return False

        return copy_src_reg.is_renamable and copy_src_reg.is_kill

    def is_backward_propagatable_regclass_copy(self, copy, inst, operand):
        copy_dst_reg = copy.operands[0].reg
        regclass = self.target_reg_info.get_regclass_from_reg(
            copy_dst_reg.spec)

        return operand.reg.spec in regclass.regs

    def propagate_defs(self, inst: MachineInstruction):
        from codegen.spec import iter_reg_units

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

            copy_dst_reg = copy.operands[0]
            copy_src_reg = copy.operands[1]

            if not self.is_backward_propagatable_regclass_copy(copy, inst, operand):
                continue

            if operand.reg == copy_dst_reg.reg:
                continue

            operand.reg = copy_dst_reg.reg
            operand.is_renamable = copy_dst_reg.is_renamable

            self.dead_copies.add(copy)

    def backward_cp_bb(self, bb: MachineBasicBlock):
        from codegen.mir_emitter import TargetDagOps

        self.track_clear()
        self.dead_copies = set()

        for inst in reversed(bb.insts):
            if inst.opcode == TargetDagOps.COPY:
                if self.is_backward_propagatable_copy(inst):
                    self.track_invalidate_reg(inst.operands[0].reg)
                    self.track_invalidate_reg(inst.operands[1].reg)
                    self.track_copy(inst)
                    continue

            self.propagate_defs(inst)

            for operand in inst.operands:
                if not operand.is_reg:
                    continue

                if operand.is_def:
                    self.track_invalidate_reg(operand.reg)

        for copy in self.dead_copies:
            copy.remove()

    def process_machine_function(self, mfunc: MachineFunction):
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.target_reg_info = mfunc.target_info.get_register_info()

        for bb in mfunc.bbs:
            self.backward_cp_bb(bb)
            # self.forward_cp_bb(bb)
