from codegen.passes import FunctionPass
from ir.values import *


class SimplifyCFG(FunctionPass):
    def __init__(self):
        pass

    def merge_block_into_predecessor(self, bb):
        if len(bb.insts) == 0:
            return False

        preds = list(bb.predecessors)

        if len(preds) != 1:
            return False

        pred = preds[0]

        if len(list(pred.successors)) != 1:
            return False

        for inst in bb.insts:
            if isinstance(inst, PHINode):
                return False

        pred.terminator.remove()

        for inst in list(bb.insts):
            inst.move_after(pred)

        bb.remove()

        return True

    def simplify(self, bb):
        changed = False

        if self.merge_block_into_predecessor(bb):
            return True

        return changed

    def process_function(self, func):
        while True:
            changed = False

            for bb in list(func.bbs):
                changed |= self.simplify(bb)

            if not changed:
                break
