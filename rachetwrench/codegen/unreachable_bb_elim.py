#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.mir import *
from rachetwrench.codegen.spec import *
from rachetwrench.codegen.passes import *


def dfs_bb(bb: MachineBasicBlock, action, visited):
    if bb in visited:
        return

    visited.add(bb)

    action(bb)

    for succ in bb.successors:
        dfs_bb(succ, action, visited)
        


class UnreachableBBElim(MachineFunctionPass):
    def __init__(self):
        super().__init__()

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc

        reachable = set()
        def collect_mbb(bb):
            reachable.add(bb)

        dfs_bb(mfunc.bbs[0], collect_mbb, set())

        unreachable = set()
        for bb in mfunc.bbs:
            if bb not in reachable:
                unreachable.add(bb)

                for succ in list(bb.successors):
                    bb.remove_successor(succ)

        for bb in unreachable:
            bb.remove_from_func()
            
