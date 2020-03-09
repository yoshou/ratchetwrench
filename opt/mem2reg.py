from codegen.passes import FunctionPass
from ir.values import *


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


class Mem2Reg(FunctionPass):
    def __init__(self):
        pass

    def initialize(self):
        self.removed_load_count = 0
        self.removed_store_count = 0

    def finalize(self):
        print("removed load", self.removed_load_count)
        print("removed store", self.removed_store_count)

    def rewrite_single_store_alloca(self, inst: AllocaInst, info):
        store_inst = None

        for use in inst.uses:
            if isinstance(use, StoreInst):
                store_inst = use
                break

        assert(store_inst)

        for use in inst.uses:
            if isinstance(use, StoreInst):
                continue

            assert(isinstance(use, LoadInst))

            for load_use in use.uses:
                for i, operand in enumerate(load_use.operands):
                    if operand is use:
                        load_use.set_operand(i, store_inst.rs)

        for use in inst.uses:
            if isinstance(use, LoadInst):
                self.removed_load_count += 1
            if isinstance(use, StoreInst):
                self.removed_store_count += 1
            use.remove()

        inst.remove()

    def promote_mem_to_reg(self, alloca_insts):
        for inst in alloca_insts:
            if len(inst.uses) == 0:
                inst.remove()
                continue

            info = AllocaInfo()
            info.analyze(inst)

            if len(info.defining_blocks) == 1:
                self.rewrite_single_store_alloca(inst, info)
                continue

    def is_promotable(self, inst):
        for use in inst.uses:
            if isinstance(use, LoadInst):
                pass
            elif isinstance(use, StoreInst):
                pass
            elif isinstance(use, BitCastInst):
                pass
            elif isinstance(use, GetElementPtrInst):
                if not use.has_all_zero_indices:
                    return False
            else:
                return False
        return True

    def process_function(self, func):
        alloca_insts = []

        for bb in func.bbs:
            for inst in bb.insts:
                if isinstance(inst, AllocaInst) and self.is_promotable(inst):
                    alloca_insts.append(inst)

        self.promote_mem_to_reg(alloca_insts)
