#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *


class CoalescerPair:
    def __init__(self, reg_info):
        self.dst_reg = None
        self.src_reg = None
        self.dst_subreg = None
        self.src_subreg = None
        self.is_partial = False
        self.reg_info = reg_info

    @property
    def is_phys(self):
        return isinstance(self.dst_reg, MachineRegister)

    def set_register(self, copy_inst):
        if not copy_inst.is_copy_like:
            return False

        dst_reg = copy_inst.operands[0].reg
        dst_subreg = copy_inst.operands[0].subreg

        src_reg = copy_inst.operands[1].reg
        src_subreg = copy_inst.operands[1].subreg

        if copy_inst.operands[1].is_phys:
            if copy_inst.operands[0].is_phys:
                return False

        if copy_inst.operands[0].is_phys:
            if dst_subreg is not None:
                dst_reg = self.target_reg_info.get_subreg(dst_reg, dst_subreg)
                dst_subreg = None

            if src_subreg is not None:
                return False

            self.src_reg = src_reg
            self.dst_reg = dst_reg

            return True

        return False


def overlap_live_range(live_range, reg_unit):
    inst = live_range.start

    while inst != live_range.end:
        for operand in inst.operands:
            if not operand.is_reg:
                continue

            if not operand.is_phys:
                continue

            if reg_unit in set(iter_reg_units(operand.reg.spec)):
                return True

        if inst.is_terminator:
            return True

        inst = inst.next_inst

    return False


class RegisterCoalescer(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def join_interval(self, copy_pair):
        live_range_info = self.mfunc.live_ranges
        
        src_live_range = live_range_info[copy_pair.src_reg]

        if copy_pair.is_phys:
            for unit in iter_reg_units(copy_pair.dst_reg.spec):
                if overlap_live_range(src_live_range, unit):
                    return False

            if not self.mfunc.reg_info.has_one_use(copy_pair.src_reg):
                return False

            for mbb in self.mfunc.bbs:
                for inst in mbb.insts:
                    for operand in inst.operands:
                        if not operand.is_reg:
                            continue
                            
                        if operand.reg != copy_pair.src_reg:
                            continue

                        operand.reg = copy_pair.dst_reg
                        operand.is_renamable = True

            live_range_info.pop(copy_pair.src_reg)

            return True
        else:
            return False


    def join_copy(self, copy_inst):
        pair = CoalescerPair(self.target_reg_info)
        if not pair.set_register(copy_inst):
            return False

        if not self.join_interval(pair):
            return False

        dst_live_range = self.mfunc.live_ranges[pair.dst_reg]

        for seg in list(dst_live_range.segments):
            if seg.start == copy_inst or seg.end == copy_inst:
                dst_live_range.segments.remove(seg)

        if len(dst_live_range.segments) == 0:
            self.mfunc.live_ranges.pop(pair.dst_reg)

        copy_inst.remove()

        return True

    def process_basicblock(self, bb: MachineBasicBlock):
        worklist = []

        for inst in bb.insts:
            if not inst.is_copy_like:
                continue

            if not inst.operands[0].is_reg or not inst.operands[0].is_phys:
                continue

            worklist.append(inst)

        for inst in worklist:
            self.join_copy(inst)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.target_reg_info = mfunc.target_info.get_register_info()

        for bb in mfunc.bbs:
            self.process_basicblock(bb)
