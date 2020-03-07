#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.passes import *


class SlotIndex:
    def __init__(self, index):
        self.index = index


class SlotSegment:
    def __init__(self):
        self.start = -1
        self.end = -1


class LiveRange:
    def __init__(self):
        self.segments = []

    @property
    def start(self):
        if len(self.segments) == 0:
            return -1

        return min([seg.start for seg in self.segments])

    @property
    def end(self):
        if len(self.segments) == 0:
            return -1

        return min([seg.end for seg in self.segments])


class LiveIntervals(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

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

        for inst in insts:
            succ = succs[inst]

            for operand in inst.operands:
                if operand.is_mbb:
                    succ.append(operand.mbb.insts[0])

                if operand.is_reg and operand.is_virtual:
                    if operand.is_use:
                        gens[inst].add(operand.reg)
                    else:
                        kills[inst].add(operand.reg)

            if inst != inst.mbb.insts[-1]:
                idx = inst.mbb.insts.index(inst)
                succ.append(inst.mbb.insts[idx + 1])

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
        live_regs = set()

        for inst in insts:
            live_in = live_ins[inst]
            live_out = live_outs[inst]

            new_live_regs = live_in - live_regs

            for new_live_reg in new_live_regs:
                live_regs.add(new_live_reg)
                if new_live_reg not in live_ranges:
                    live_ranges[new_live_reg] = LiveRange()

                seg = SlotSegment()
                seg.start = insts.index(inst)
                live_ranges[new_live_reg].segments.append(seg)

            killed_regs = live_in - live_out
            for killed_reg in killed_regs:
                live_regs.remove(killed_reg)
                assert(live_ranges[killed_reg].segments[-1].end == -1)
                live_ranges[killed_reg].segments[-1].end = insts.index(inst)

        self.mfunc.live_ranges = live_ranges
