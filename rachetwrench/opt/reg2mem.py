from rachetwrench.codegen.passes import FunctionPass
from rachetwrench.ir.values import *


class AllocaInfo:
    def __init__(self):
        self.defining_blocks = []
        self.using_blocks = []
        self.used_blocks = []

    def analyze(self, inst: AllocaInst):
        for use in inst.uses:
            if isinstance(use, StoreInst):
                self.defining_blocks.append(use.block)
            elif isinstance(use, LoadInst):
                self.using_blocks.append(use.block)

            self.used_blocks.append(use.block)

    @property
    def only_used_in_one_block(self):
        if len(self.used_blocks) < 2:
            return True
        for used_block in self.used_blocks:
            if used_block != self.used_blocks[-1]:
                return False

        return True


class Reg2Mem(FunctionPass):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass

    def demote_phi_to_stack(self, inst: PHINode):
        func = inst.block.func

        from rachetwrench.ir.types import i32

        slot = AllocaInst(func.blocks[0], ConstantInt(1, i32), inst.ty, 0)

        for value, block in zip(inst.incoming_values, inst.incoming_blocks):
            if len(block.insts) == 0:
                insert_pt = block
            else:
                insert_pt = block.insts[len(block.insts) - 2]
            StoreInst(insert_pt, value, slot)

        val = LoadInst(inst, slot)

        for use in inst.uses:
            for i in range(len(use.operands)):
                if use.operands[i] == inst:
                    use.set_operand(i, val)

        inst.remove()

    def process_function(self, func):

        work_list = []

        for bb in func.bbs:
            for inst in bb.insts:
                if isinstance(inst, PHINode):
                    work_list.append(inst)

        for inst in work_list:
            self.demote_phi_to_stack(inst)
