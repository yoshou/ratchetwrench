#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, auto
from codegen.passes import *
from codegen.mir import *
from codegen.spec import *


class PhysRegState(Enum):
    Disabled = auto()
    Free = auto()
    Reserved = auto()


class LiveRegInfo:
    def __init__(self, reg):
        assert(isinstance(reg, MachineVirtualRegister))
        self.reg = reg
        self.last_use = None
        self.phys_reg = None
        self.age = 0
        self.stack_slot = -1
        self.dirty = True


class FastRegisterAllocation(MachineFunctionPass):
    """
        A basic block based register allocation.
        All registers acrossing basic blocks are spilled to stack.
    """

    def __init__(self):
        super().__init__()

    def spill_virt_reg(self, reg, inst):
        live_reg = self.live_regs[reg]
        phys_reg = live_reg.phys_reg

        hwmode = inst.mbb.func.target_info.hwmode

        if not phys_reg:
            return

        if live_reg.dirty:
            regclass = reg.regclass
            align = int(regclass.align / 8)

            tys = regclass.get_types(hwmode)
            size = tys[0].get_size_in_bits()
            size = int(int((size + 7) / 8))

            stack_slot = live_reg.stack_slot
            if stack_slot == -1:
                stack_slot = self.mfunc.create_stack_object(size, align)
                live_reg.stack_slot = stack_slot

            self.target_inst_info.copy_reg_to_stack(
                phys_reg, stack_slot, regclass, inst).comment = "Spill"

            live_reg.dirty = False

        self.kill_virt_reg(live_reg)

    def get_phys_reg_state(self, phys_reg):
        if phys_reg not in self.phys_reg_states:
            return PhysRegState.Free

        return self.phys_reg_states[phys_reg]

    def set_phys_reg_state(self, phys_reg, state_or_virt_reg):
        assert(isinstance(state_or_virt_reg,
                          (PhysRegState, MachineVirtualRegister)))

        self.phys_reg_states[phys_reg] = state_or_virt_reg

    def define_phys_reg(self, phys_reg, new_state, inst):
        assert(isinstance(new_state, (PhysRegState, MachineVirtualRegister)))

        if new_state in [PhysRegState.Free, PhysRegState.Reserved]:
            old_state = self.get_phys_reg_state(phys_reg)
            if not isinstance(old_state, PhysRegState):
                self.spill_virt_reg(old_state, inst)

            regs = iter_reg_aliases(phys_reg.spec)
            for reg in regs:
                self.set_phys_reg_state(MachineRegister(reg), new_state)
        elif new_state == PhysRegState.Disabled:
            raise NotImplementedError()
        else:
            old_state = self.get_phys_reg_state(phys_reg)
            if not isinstance(old_state, PhysRegState):
                self.spill_virt_reg(old_state, inst)

            regs = iter_reg_aliases(phys_reg.spec)
            for reg in regs:
                self.set_phys_reg_state(MachineRegister(reg), new_state)

    def is_reg_used(self, phys_reg: MachineRegister):
        assert(isinstance(phys_reg, MachineRegister))

        regs = list(iter_reg_aliases(phys_reg.spec))
        for reg in regs:
            phys_reg_alias = MachineRegister(reg)
            if self.get_phys_reg_state(phys_reg_alias) != PhysRegState.Free:
                return True

        return False

    def alloc_virt_reg(self, reg: MachineVirtualRegister, inst: MachineInstruction):
        assert(isinstance(reg, MachineVirtualRegister))

        if reg in self.live_regs:
            live_reg = self.live_regs[reg]
        else:
            live_reg = LiveRegInfo(reg)
            self.live_regs[reg] = live_reg

        alloc_regs = self.register_info.get_ordered_regs(reg.regclass)

        for phys_reg_def in alloc_regs:
            phys_reg = MachineRegister(phys_reg_def)
            virt_reg_or_state = self.get_phys_reg_state(phys_reg)
            if not self.is_reg_used(phys_reg):
                self.define_phys_reg(phys_reg, reg, inst)
                live_reg.phys_reg = phys_reg

                return live_reg

        import random

        phys_reg = MachineRegister(random.choice(alloc_regs))
        self.define_phys_reg(phys_reg, reg, inst)
        live_reg.phys_reg = phys_reg

        return live_reg

    def kill_virt_reg(self, live_reg: LiveRegInfo):
        phys_reg = live_reg.phys_reg

        regs = list(iter_reg_aliases(phys_reg.spec))
        for reg in regs:
            self.set_phys_reg_state(MachineRegister(reg), PhysRegState.Free)

        live_reg.phys_reg = None

    def spill_all(self, inst: MachineInstruction, only_live_out=False):
        for reg, live_reg in self.live_regs.items():
            if live_reg.phys_reg is None:
                continue

            may_live_out = False
            for use in self.mfunc.reg_info.get_use_iter(reg):
                if use.inst.mbb != inst.mbb:
                    may_live_out = True

            if not only_live_out or may_live_out:
                self.spill_virt_reg(reg, inst)

    def define_virt_reg(self, reg: MachineVirtualRegister, inst: MachineInstruction):
        if reg not in self.live_regs or self.live_regs[reg].phys_reg is None:
            live_reg = self.alloc_virt_reg(reg, inst)
        else:
            live_reg = self.live_regs[reg]
            live_reg.age = 0

        live_reg.dirty = True

        return live_reg.phys_reg

    def reload_virt_reg(self, reg: MachineVirtualRegister, inst: MachineInstruction):
        live_reg = self.live_regs[reg]

        live_reg.age = 0
        phys_reg = live_reg.phys_reg
        stack_slot = live_reg.stack_slot

        if not phys_reg:
            live_reg = self.alloc_virt_reg(reg, inst)
            # if live_reg.stack_slot != -1:
            #     live_reg.stack_slot = stack_slot
            phys_reg = live_reg.phys_reg

            assert(live_reg.stack_slot != -1)

            self.target_inst_info.copy_reg_from_stack(
                phys_reg, live_reg.stack_slot, reg.regclass, inst).comment = "Reload"

        assert(phys_reg is not None)
        return phys_reg

    def set_phys_reg(self, operand, phys_reg):
        operand.reg = phys_reg
        operand.is_renamable = True

    def allocate_instruction(self, inst: MachineInstruction):
        # Replace using vregs to phys registers.
        kill_list = []
        for operand in inst.operands:
            if not operand.is_reg or operand.is_def:
                continue

            if not operand.is_virtual:
                continue

            reg = operand.reg

            phys_reg = self.reload_virt_reg(reg, inst)
            assert(isinstance(phys_reg, MachineRegister))

            self.set_phys_reg(operand, phys_reg)
            if operand.is_kill:
                kill_list.append(reg)

        for kill_reg in kill_list:
            live_reg = self.live_regs[kill_reg]
            if live_reg.phys_reg is not None:
                self.kill_virt_reg(live_reg)

        # Function calling needs to save some registers.
        if inst.is_call:
            self.spill_all(inst)

        # Define phys registers.
        for operand in inst.operands:
            if not operand.is_reg or operand.is_implicit:
                continue

            if not operand.is_phys:
                continue

            reg = operand.reg

            if operand.is_use:
                continue
                state = PhysRegState.Free if operand.is_kill else PhysRegState.Reserved
                regs = iter_reg_aliases(reg.spec)
                for reg in regs:
                    self.set_phys_reg_state(
                        MachineRegister(reg), state)
            else:
                state = PhysRegState.Free if operand.is_dead else PhysRegState.Reserved
                self.define_phys_reg(reg, PhysRegState.Reserved, inst)

        # Allocate virtual registers.
        for operand in inst.operands:
            if not operand.is_reg:
                continue

            if not operand.is_virtual:
                continue

            if not operand.is_def:
                continue

            reg = operand.reg

            phys_reg = self.define_virt_reg(reg, inst)
            self.set_phys_reg(operand, phys_reg)

        for live_reg in self.live_regs.values():
            if live_reg.phys_reg:
                live_reg.age += 1

    def allocate_basicblock(self, mbb: MachineBasicBlock):
        for inst in mbb.insts:
            self.allocate_instruction(inst)

        self.spill_all(mbb.first_terminator, True)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc

        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()
        self.register_info = mfunc.target_info.get_register_info()

        self.live_regs = {}
        self.phys_reg_states = {}

        for reg, vreg in mfunc.reg_info.live_ins:
            continue
            alias_regs = iter_reg_aliases(reg.spec)
            for alias_reg in alias_regs:
                self.set_phys_reg_state(MachineRegister(alias_reg), vreg)
            self.live_regs[vreg] = LiveRegInfo(vreg)
            self.live_regs[vreg].phys_reg = reg

        # mfunc.reg_info.live_ins.clear()

        for bb in mfunc.bbs:
            self.allocate_basicblock(bb)
