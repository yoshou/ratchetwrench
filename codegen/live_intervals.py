#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.passes import *


class SlotSegment:
    def __init__(self):
        self._start = None
        self._end = None

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        assert(isinstance(value, MachineInstruction))
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        assert(isinstance(value, MachineInstruction))
        self._end = value


from functools import cmp_to_key


def cmp_inst_func(a, b):
    if a.mbb.index < b.mbb.index:
        return -1
    elif a.mbb.index > b.mbb.index:
        return 1

    if a.index < b.index:
        return -1
    elif a.index > b.index:
        return 1

    return 0


def cmp_interval_start_func(a, b):
    return cmp_inst_func(a.start, b.start)


class LiveRange:
    def __init__(self):
        self.segments = []

    @property
    def start(self):
        if len(self.segments) == 0:
            return None

        return min([seg.start for seg in self.segments], key=cmp_to_key(cmp_inst_func))

    @property
    def end(self):
        if len(self.segments) == 0:
            return None

        return max([seg.end for seg in self.segments], key=cmp_to_key(cmp_inst_func))


class LiveIntervals(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.target_reg_info = mfunc.target_info.get_register_info()

        live_ins = {}
        live_outs = {}

        insts = [inst for mbb in mfunc.bbs for inst in mbb.insts]

        for inst in insts:
            live_ins[inst] = set()
            live_outs[inst] = set()

        succs = {}
        gens = {}
        kills = {}
        for inst in insts:
            succs[inst] = []
            gens[inst] = set()
            kills[inst] = set()

        phys_regs = set()

        for mbb in self.mfunc.bbs:
            for inst in mbb.insts:
                succ = succs[inst]

                for operand in inst.operands:
                    if operand.is_mbb and len(operand.mbb.insts) > 0:
                        succ.append(operand.mbb.insts[0])

                    if operand.is_reg:
                        if operand.is_def:
                            kills[inst].add(operand.reg)

                            if operand.is_phys:
                                phys_regs.add(operand.reg)

                        if operand.is_use or (operand.is_def and operand.subreg):
                            gens[inst].add(operand.reg)

                if inst.is_call:
                    non_csrs = self.target_reg_info.get_callee_clobbered_regs()

                    for non_csr in non_csrs:
                        reg = MachineRegister(non_csr)
                        kills[inst].add(reg)
                        gens[inst].add(reg)
                        phys_regs.add(reg)

                if inst != mbb.insts[-1]:
                    succ.append(inst.next_inst)

                if inst == mbb.insts[-1]:
                    for phys_reg in phys_regs:
                        kills[inst].add(phys_reg)
                        gens[inst].add(phys_reg)

        changed = True
        count = 0
        while changed:
            count += 1
            changed = False
            for inst in list(reversed(insts)):
                next_live_outs = set()
                for succ in succs[inst]:
                    next_live_outs |= live_ins[succ]

                changed |= (live_outs[inst] != next_live_outs)
                live_outs[inst] = next_live_outs

                gen = gens[inst]
                kill = kills[inst]

                next_live_ins = gen | (live_outs[inst] - kill)

                changed |= (live_ins[inst] != next_live_ins)
                live_ins[inst] = next_live_ins

        for inst in insts:
            live_in = live_ins[inst]
            live_out = live_outs[inst]

            killed = live_in - live_out

            for operand in inst.operands:
                if not operand.is_reg or operand.is_phys:
                    continue

                operand.is_dead = operand.is_def and operand.reg not in live_out

                is_kill = operand.is_use and operand.reg in killed

                operand.is_kill = operand.is_use and operand.reg in killed

        live_ranges = {}
        # live_regs = set()

        for live_in, _ in mfunc.reg_info.live_ins:
            # live_regs.add(live_in)

            live_ranges[live_in] = LiveRange()
            live_ranges[live_in].reg = live_in

            seg = SlotSegment()
            seg.start = inst
            seg.end = inst.mbb.func.bbs[0].insts[0]
            live_ranges[live_in].segments.append(seg)

        inst = insts[0]
        new_live_regs = live_ins[inst]
        for new_live_reg in new_live_regs:
            # live_regs.add(new_live_reg)
            if new_live_reg not in live_ranges:
                live_ranges[new_live_reg] = LiveRange()
                live_ranges[new_live_reg].reg = new_live_reg

            seg = SlotSegment()
            seg.start = inst
            seg.end = inst.mbb.func.bbs[-1].insts[-1]
            live_ranges[new_live_reg].segments.append(seg)

        for inst in insts:
            live_in = live_ins[inst]
            live_out = live_outs[inst]

            killed_regs = live_in - live_out
            for killed_reg in killed_regs:
                # live_regs.remove(killed_reg)
                if killed_reg not in live_ranges:
                    live_ranges[killed_reg] = LiveRange()
                    live_ranges[killed_reg].reg = killed_reg

                    seg = SlotSegment()
                    seg.start = inst
                    live_ranges[killed_reg].segments.append(seg)

                live_ranges[killed_reg].segments[-1].end = inst

            new_live_regs = live_out - live_in

            for kill in kills[inst]:
                if kill not in live_ranges:
                    new_live_regs.add(kill)

            for new_live_reg in new_live_regs:
                # live_regs.add(new_live_reg)
                if new_live_reg not in live_ranges:
                    live_ranges[new_live_reg] = LiveRange()
                    live_ranges[new_live_reg].reg = new_live_reg

                seg = SlotSegment()
                seg.start = inst
                seg.end = inst.mbb.func.bbs[-1].insts[-1]
                live_ranges[new_live_reg].segments.append(seg)

        self.mfunc.live_ranges = live_ranges
