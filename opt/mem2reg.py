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
                self.only_store = use
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
        store_inst = info.only_store

        assert(store_inst)

        def replace_all_uses_with(used, value):
            for use in used.uses:
                for i, operand in enumerate(use.operands):
                    if operand is used:
                        assert(use.operands[i].ty == value.ty)
                        use.set_operand(i, value)

        for use in inst.uses:
            if use == store_inst:
                continue

            assert(isinstance(use, (LoadInst,)))

            replace_all_uses_with(use, store_inst.rs)

        for use in inst.uses:
            if isinstance(use, LoadInst):
                self.removed_load_count += 1
            if isinstance(use, StoreInst):
                self.removed_store_count += 1
            use.remove()

        inst.remove()

    def promote_mem_to_reg(self, alloca_insts):
        for i, inst in enumerate(alloca_insts):
            assert(self.is_promotable(inst))

            info = AllocaInfo()
            info.analyze(inst)

            if len(inst.uses) == 0:
                inst.remove()
                continue

            if len(info.defining_blocks) == 1:
                self.rewrite_single_store_alloca(inst, info)
                continue

    def is_promotable(self, inst):
        for use in inst.uses:
            if isinstance(use, LoadInst):
                if use.rs != inst:
                    return False

                for load_use in use.uses:
                    if isinstance(load_use, PHINode):
                        return False
            elif isinstance(use, StoreInst):
                if use.rd != inst:
                    return False
            elif isinstance(use, BitCastInst):
                return False
                pass
            elif isinstance(use, GetElementPtrInst):
                return False
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
