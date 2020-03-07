#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *


class PrologEpilogInsertion(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_instruction(self, inst: MachineInstruction):
        pass

    def process_basicblock(self, bb: MachineBasicBlock):
        for inst in bb.insts:
            self.process_instruction(inst)

    def insert_prolog_epilog(self, func: MachineFunction):
        self.target_lowering.lower_prolog(func, func.bbs[0])
        self.target_lowering.lower_epilog(func, func.bbs[-1])

    def calculate_stack_object_offsets(self, func: MachineFunction):
        frame_lowering = func.target_info.get_frame_lowering()
        stack_grows_down = frame_lowering.stack_grows_direction == StackGrowsDirection.Down

        frame = func.frame

        offset = 0
        align = 0
        for fixed_object in frame.stack_object[:frame.fixed_count]:
            if stack_grows_down:
                obj_offset = -fixed_object.offset
            else:
                obj_offset = fixed_object.offset + fixed_object.size
            if obj_offset > offset:
                offset = obj_offset

        max_align = frame.max_alignment

        for stack_object in frame.stack_object[frame.fixed_count:]:
            align = stack_object.align

            if stack_grows_down:
                offset += stack_object.size
                offset = int(int((offset + align - 1) / align) * align)

                stack_object.offset = -offset
            else:
                offset = int(int((offset + align - 1) / align) * align)

                stack_object.offset = offset
                offset += stack_object.size


    def process_machine_function(self, mfunc: MachineFunction):
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

        for bb in mfunc.bbs:
            self.process_basicblock(bb)

        self.insert_prolog_epilog(mfunc)

        self.calculate_stack_object_offsets(mfunc)

        for bb in mfunc.bbs:
            for inst in list(bb.insts):
                if self.target_lowering.is_frame_op(inst):
                    self.target_lowering.eliminate_call_frame_pseudo_inst(
                        mfunc, inst)

        for bb in mfunc.bbs:
            for inst in list(bb.insts):
                for idx in range(len(inst.operands)):
                    self.target_inst_info.eliminate_frame_index(
                        mfunc, inst, idx)
