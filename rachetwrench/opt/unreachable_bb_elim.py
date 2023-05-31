from rachetwrench.codegen.passes import FunctionPass
from rachetwrench.ir.values import *


def dfs_bb(bb: BasicBlock, action, visited):
    if bb in visited:
        return

    visited.add(bb)

    action(bb)

    for succ in bb.successors:
        dfs_bb(succ, action, visited)


class UnreachableBlockElim(FunctionPass):
    def __init__(self):
        pass

    def process_function(self, func):
        if len(func.blocks) == 0:
            return

        reachable = set()

        def collect_mbb(bb):
            reachable.add(bb)

        dfs_bb(func.blocks[0], collect_mbb, set())

        unreachable = set()
        for bb in func.blocks:
            if bb not in reachable:
                unreachable.add(bb)

        for bb in unreachable:
            bb.remove()
