#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, auto
from rachetwrench.codegen.passes import *
from rachetwrench.codegen.mir import *
from rachetwrench.codegen.spec import *
from rachetwrench.codegen.live_intervals import cmp_interval_start_func, cmp_inst_func
from rachetwrench.codegen.live_intervals import LiveIntervals


class Spiller:
    def __init__(self):
        pass

    def spill(self, func, stack_slot, vreg, new_ranges=None):
        from rachetwrench.codegen.live_intervals import LiveRange, SlotSegment

        insts = set(
            [operand.inst for operand in func.reg_info.get_use_def_iter(vreg)])

        target_inst_info = func.target_info.get_inst_info()

        for inst in insts:
            has_use = False
            has_def = False
            for operand in inst.operands:
                if not operand.is_reg:
                    continue

                if operand.reg != vreg:
                    continue

                has_use |= (operand.is_use | (operand.subreg is not None))
                has_def |= operand.is_def

            assert(has_use | has_def)

            new_vreg = func.reg_info.create_virtual_register(vreg.regclass)

            if has_use:
                target_inst_info.copy_reg_from_stack(
                    new_vreg, stack_slot, vreg.regclass, inst).comment = "Reload"

            if has_def:
                target_inst_info.copy_reg_to_stack(
                    new_vreg, stack_slot, vreg.regclass, inst.next_inst).comment = "Spill"

            interval = LiveRange()
            interval.reg = new_vreg
            segment = SlotSegment()
            interval.segments.append(segment)

            start_inst = inst.prev_inst if has_use else inst
            end_inst = inst.next_inst if has_def else inst

            segment.start = start_inst
            segment.end = end_inst

            if new_ranges is not None:
                new_ranges.append(interval)

            for operand in inst.operands:
                if not operand.is_reg:
                    continue

                if operand.reg != vreg:
                    continue

                operand.reg = new_vreg


from functools import cmp_to_key


class LinearScanRegisterAllocation(MachineFunctionPass):
    """
        A global register allocation.
    """

    def __init__(self):
        super().__init__()

    def expire_old_intervals(self, live_range):
        for rnge in list(self.active_regs):
            if cmp_inst_func(rnge.end, live_range.start) > 0:
                continue

            self.active_regs.remove(rnge)
            self.inactive_regs.append(rnge)

            phys_reg = self.get_phys_reg(rnge)
            if not phys_reg:
                continue

            reg_units = set(iter_reg_units(phys_reg.spec))
            for reg_unit in reg_units:
                self.used_reg_units.remove(reg_unit)

    def is_reg_used(self, phys_reg: MachineRegister):
        assert(isinstance(phys_reg, MachineRegister))

        reg_units = set(iter_reg_units(phys_reg.spec))

        used_units = reg_units & self.used_reg_units
        return len(used_units) > 0

    def allocate_reg_or_stack(self, live_range):
        hwmode = self.mfunc.target_info.hwmode

        alloc_regs = self.register_info.get_ordered_regs(
            live_range.reg.regclass)

        if self.get_phys_reg(live_range):
            return

        for phys_reg_def in alloc_regs:
            phys_reg = MachineRegister(phys_reg_def)
            if self.is_reg_used(phys_reg):
                continue

            if set(iter_reg_units(phys_reg.spec)) & self.fixed:
                continue

            self.set_phys_reg(live_range, phys_reg)

            reg_units = set(iter_reg_units(phys_reg.spec))
            self.used_reg_units |= reg_units

            self.active_regs.append(live_range)
            return

        spill_reg = None
        phys_reg_to_alloc = None

        for active_reg in self.active_regs:
            if isinstance(active_reg.reg, MachineRegister):
                continue

            phys_reg = self.get_phys_reg(active_reg)
            if not phys_reg:
                continue

            for reg in live_range.reg.regclass.regs:
                if phys_reg.spec in iter_reg_aliases(reg):
                    spill_reg = active_reg
                    phys_reg_to_alloc = MachineRegister(reg)
                    break

                if spill_reg:
                    break

        assert(spill_reg)

        self.set_phys_reg(live_range, phys_reg_to_alloc)

        for active_reg in self.active_regs:
            if isinstance(active_reg.reg, MachineRegister):
                continue

            phys_reg = self.get_phys_reg(active_reg)
            if not phys_reg:
                continue

            if set(iter_reg_units(phys_reg.spec)) & set(iter_reg_units(phys_reg_to_alloc.spec)):
                regclass = active_reg.reg.regclass
                align = int(regclass.align / 8)

                tys = regclass.get_types(hwmode)
                size = tys[0].get_size_in_bits()
                size = int(int((size + 7) / 8))

                stack_slot = self.get_stack(active_reg)
                if stack_slot == -1:
                    stack_slot = self.mfunc.create_stack_object(size, align)
                    self.set_stack(active_reg, stack_slot)

                self.active_regs.remove(active_reg)
                reg_units = set(iter_reg_units(phys_reg.spec))
                self.used_reg_units -= reg_units
                self.set_phys_reg(active_reg, None)
                self.spills.append(active_reg)

        self.active_regs.append(live_range)
        self.used_reg_units |= set(iter_reg_units(phys_reg_to_alloc.spec))

    def allocate(self):
        alloc = True

        while alloc:
            alloc = False
            unhandled = list(
                sorted(self.mfunc.live_ranges.values(), key=cmp_to_key(cmp_interval_start_func)))

            self.used_reg_units = set()
            self.active_regs = []
            self.inactive_regs = []
            self.spills = []

            for cur_range in list(unhandled):
                if isinstance(cur_range.reg, MachineRegister):
                    continue

                self.expire_old_intervals(cur_range)

                phys_reg = self.get_phys_reg(cur_range)
                if phys_reg:
                    reg_units = set(iter_reg_units(phys_reg.spec))
                    self.used_reg_units |= reg_units
                    assert(cur_range not in self.active_regs)
                    self.active_regs.append(cur_range)
                    continue

                self.allocate_reg_or_stack(cur_range)

            for spill in self.spills:
                new_vregs = []
                self.spiller.spill(
                    self.mfunc, self.get_stack(spill), spill.reg, new_vregs)

                for new_vreg in new_vregs:
                    self.mfunc.live_ranges[new_vreg.reg] = new_vreg

                self.mfunc.live_ranges.pop(spill.reg)

                self.live_intervals.process_machine_function(self.mfunc)
                alloc = True

    def set_phys_reg(self, interval, reg):
        self.phys_reg_for_vreg[interval.reg] = reg

    def get_phys_reg(self, interval):
        if interval.reg not in self.phys_reg_for_vreg:
            return None

        return self.phys_reg_for_vreg[interval.reg]

    def set_stack(self, interval, reg):
        self.stack_for_vreg[interval.reg] = reg

    def get_stack(self, interval):
        if interval.reg not in self.stack_for_vreg:
            return -1

        return self.stack_for_vreg[interval.reg]

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc

        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.register_info = mfunc.target_info.get_register_info()

        self.allocatable_regs = self.register_info.allocatable_regs

        self.used_reg_units = set()

        self.active_regs = []
        self.fixed = set()

        self.spiller = Spiller()

        self.spills = []
        self.phys_reg_for_vreg = {}
        self.stack_for_vreg = {}

        self.live_intervals = LiveIntervals()

        for reg, live_range in self.mfunc.live_ranges.items():
            self.set_phys_reg(live_range, None)

            if isinstance(reg, MachineRegister):
                self.set_phys_reg(live_range, reg)
                if reg.spec in self.allocatable_regs:
                    regs = set(iter_reg_units(reg.spec))
                    self.fixed |= regs

        self.allocate()

        for mbb in self.mfunc.bbs:
            for inst in mbb.insts:
                for operand in inst.operands:
                    if not operand.is_reg or not operand.is_virtual:
                        continue

                    phys_reg = self.get_phys_reg(
                        self.mfunc.live_ranges[operand.reg])

                    if not phys_reg:
                        continue
                    operand.reg = phys_reg
                    operand.is_renamable = True

        self.mfunc.live_ranges = {}
