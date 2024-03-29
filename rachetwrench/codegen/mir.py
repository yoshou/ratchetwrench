from enum import Enum, IntFlag
from rachetwrench.codegen.dag import DagOp
from rachetwrench.codegen.mir_emitter import MachineRegister, MachineVirtualRegister
from rachetwrench.codegen.spec import StackGrowsDirection
from rachetwrench.ir.data_layout import DataLayout
from rachetwrench.ir.values import Function


class MachineConstantPoolValue:
    def __init__(self, ty):
        self.ty = ty


class MachineConstantPoolEntry:
    def __init__(self, value, alignment):
        from rachetwrench.ir.values import Constant
        assert(isinstance(value, (Constant, MachineConstantPoolValue)))
        self.value = value
        self.alignment = alignment

    @property
    def is_machine_cp_entry(self):
        return isinstance(self.value, MachineConstantPoolValue)


class MachineConstantPool:
    def __init__(self, data_layout: DataLayout):
        self.data_layout = data_layout
        self.alignment = 1
        self.constants = []

    @property
    def pool_alignment(self):
        return self.alignment

    def can_share_constant_pool_index(self, value1, value2):
        if value1 == value2:
            return True

        return False

    def get_or_create_index(self, value, alignment):
        self.alignment = max(self.alignment, alignment)

        for idx, constant in enumerate(self.constants):
            if self.can_share_constant_pool_index(constant.value, value):
                if constant.alignment < alignment:
                    constant.alignment = alignment
                return idx

        self.constants.append(MachineConstantPoolEntry(value, alignment))
        return len(self.constants) - 1


class StackObject:
    def __init__(self, size, align, offset=0):
        self.size = size
        self.align = align
        self.offset = offset


class CalleeSavedInfo:
    def __init__(self, reg, frame_idx):
        self.reg = reg
        self.frame_idx = frame_idx


class MachineFrame:
    def __init__(self, func, stack_alignment):
        self.stack_object = []
        self.fixed_count = 0
        self.func = func
        self.max_alignment = 1
        self.stack_alignment = stack_alignment
        self.calee_save_info = []

    def update_max_alignment(self, align):
        if self.max_alignment < align:
            self.max_alignment = align

    def create_stack_object(self, size, align):
        assert(size > 0)
        assert(align > 0)
        idx = len(self.stack_object) - self.fixed_count

        self.stack_object.append(StackObject(size, align))
        self.update_max_alignment(align)

        return idx

    def calculate_alignment_from_offset(self, offset, max_align):
        power_of_2 = 0
        if offset == 0:
            return max_align

        while offset & 1 == 0:
            power_of_2 += 1
            offset = offset >> 1

        return 2 ** power_of_2

    def create_fixed_stack_object(self, size, offset):
        stack_align = 4
        align = self.calculate_alignment_from_offset(offset, stack_align)
        align = min(align, stack_align)
        self.stack_object.insert(0, StackObject(size, align, offset))
        self.fixed_count += 1
        idx = -self.fixed_count

        return idx

    def get_stack_object(self, idx):
        return self.stack_object[self.fixed_count + idx]

    def get_stack_object_offset(self, idx):
        return self.stack_object[self.fixed_count + idx].offset

    def compute_max_call_frame_size(self, setup_opcode, recovery_op):
        max_size = 0
        for bb in self.func.bbs:
            for inst in bb.insts:
                if inst.opcode == setup_opcode or inst.opcode == recovery_op:
                    size = inst.operands[0].val
                    max_size = max([max_size, size])

        return max_size

    def estimate_stack_size(self, setup_opcode, recovery_op):
        frame_lowering = self.func.target_info.get_frame_lowering()
        stack_grows_down = frame_lowering.stack_grows_direction == StackGrowsDirection.Down

        offset = 0
        for fixed_object in self.stack_object[:self.fixed_count]:
            if stack_grows_down:
                obj_offset = -fixed_object.offset
            else:
                obj_offset = fixed_object.offset + fixed_object.size
            if obj_offset > offset:
                offset = obj_offset

        max_align = self.max_alignment

        for stack_object in self.stack_object[self.fixed_count:]:
            align = stack_object.align
            offset += stack_object.size
            offset = int(int((offset + align - 1) / align) * align)

            max_align = max([max_align, align])

        offset += self.compute_max_call_frame_size(setup_opcode, recovery_op)

        stack_size = int(int((offset + max_align - 1) / max_align) * max_align)

        return stack_size

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


class MachineRegisterInfo:
    def __init__(self, mfunc):
        self.mfunc = mfunc
        self.vregs = []
        self.live_ins = []
        self.reg_use_def_chain_head = {}

    def create_virtual_register(self, regclass):
        vreg = MachineVirtualRegister(regclass, len(self.vregs))
        self.vregs.append(vreg)
        return vreg

    def add_live_in(self, phys_reg, vreg=None):
        self.live_ins.append((phys_reg, vreg))

    def remove_reg_operand_to_use(self, operand):
        assert(operand.next is not None or operand.prev is not None)
        head = self.get_reg_use_def_chain(operand.reg)

        if head == operand:
            self.reg_use_def_chain_head[operand.reg] = operand.next
        else:
            operand.prev.next = operand.next

        if operand.next is not None:
            operand.next.prev = operand.prev
        else:
            head.prev = operand.prev

        operand.next = operand.prev = None

    def add_reg_operand_to_use(self, operand):
        assert(operand.prev is None and operand.next is None)
        head = self.get_reg_use_def_chain(operand.reg)
        if head is None:
            self.set_reg_use_def_chain(operand)

            # Previous of the head is the last.
            operand.prev = operand
            operand.next = None
            return

        last = head.prev
        head.prev = operand
        operand.prev = last

        if operand.is_def:
            operand.next = head
            # Mark as the latest definition.
            self.set_reg_use_def_chain(operand)
        else:
            operand.next = None
            last.next = operand

    def get_use_iter(self, reg):
        current = self.get_reg_use_def_chain(reg)

        while current is not None:
            if current.is_use:
                yield current

            current = current.next

    def get_use_def_iter(self, reg):
        current = self.get_reg_use_def_chain(reg)

        while current is not None:
            yield current
            current = current.next

    def has_one_use(self, reg):
        it = self.get_use_iter(reg)
        if next(it, None) == None:
            return False
        return next(it, None) == None

    def is_use_empty(self, reg):
        it = self.get_use_iter(reg)
        return next(it, None) == None

    def get_reg_use_def_chain(self, reg):
        if reg in self.reg_use_def_chain_head:
            return self.reg_use_def_chain_head[reg]

        return None

    def set_reg_use_def_chain(self, operand):
        self.reg_use_def_chain_head[operand.reg] = operand


class FunctionInfo:
    def __init__(self, func, calling_conv):
        self.func = func
        self.pic_label_id = 0

    def create_pic_label_id(self):
        label_id = self.pic_label_id
        self.pic_label_id += 1
        return label_id


class MachineFunction:
    def __init__(self, target_info, func: Function):
        self.target_info = target_info
        self._bbs = []

        stack_align = target_info.get_frame_lowering().stack_alignment

        self.frame = MachineFrame(self, stack_align)
        self.reg_info = MachineRegisterInfo(self)
        self.func_info = FunctionInfo(func, target_info.get_calling_conv())
        self.constant_pool = MachineConstantPool(func.module.data_layout)

    @property
    def bbs(self):
        return tuple(self._bbs)

    def append_bb(self, bb):
        self.insert_bb(bb, len(self._bbs))

    def insert_bb(self, bb, index):
        if bb.func is not None:
            raise ValueError("The basic block exists in another.")

        for bb_after in self._bbs[index:]:
            bb_after.index += 1
        bb.index = index
        bb.func = self

        self._bbs.insert(index, bb)

    def remove_bb(self, bb):
        if bb.func is not self:
            raise ValueError("The basic block exists in another.")

        index = bb.index
        for bb_after in self._bbs[index:]:
            bb_after.index -= 1
        bb.index = -1
        bb.func = None

        self._bbs.pop(index)

    def create_stack_object(self, size, align):
        return self.frame.create_stack_object(size, align)

    def print_inst(self, inst, f, slot_id_map):
        start_op = 0
        first = True
        for operand in inst.operands:
            if not operand.is_reg or not operand.is_def or operand.is_implicit:
                break

            if first:
                first = False
            else:
                f.write(', ')

            operand.print(f, slot_id_map)

            start_op += 1

        if start_op > 0:
            f.write(' = ')

        from rachetwrench.codegen.spec import TargetDagOps

        if isinstance(inst.opcode, TargetDagOps):
            if inst.opcode == TargetDagOps.COPY:
                f.write('copy')
            elif inst.opcode == TargetDagOps.COPY_TO_REGCLASS:
                f.write('copy_to_subreg')
            elif inst.opcode == TargetDagOps.EXTRACT_SUBREG:
                f.write('extract_subreg')
            elif inst.opcode == TargetDagOps.INSERT_SUBREG:
                f.write('insert_subreg')
            elif inst.opcode == TargetDagOps.SUBREG_TO_REG:
                f.write('subreg_to_reg')
            elif inst.opcode == TargetDagOps.INLINEASM:
                f.write('asm')
            elif inst.opcode == TargetDagOps.IMPLICIT_DEF:
                f.write('implicit_def')
            else:
                raise NotImplementedError()
        else:
            f.write('{}'.format(inst.opcode.mnemonic))

        first = True
        for operand in inst.operands[start_op:]:
            if first:
                f.write(' ')
                first = False
            else:
                f.write(', ')
            operand.print(f, slot_id_map)

        if inst.comment != "":
            f.write(f" # {inst.comment}")

        f.write('\n')

    def print_frame_info(self, f):
        f.write("  stack:\n")
        for idx, stack_object in enumerate(self.frame.stack_object[self.frame.fixed_count:]):
            f.write("    ")
            f.write(
                f"id : {idx}, offset = {stack_object.offset}, size = {stack_object.size}, align = {stack_object.align}")
            f.write("\n")

    def print(self, f):
        slot_id_map = {}

        f.write('\n')
        f.write('function({}): # {}\n'.format(
            id(self), self.func_info.func.name))
        self.print_frame_info(f)
        for bb in self.bbs:
            f.write('\n')
            f.write('  bb({}):\n'.format(id(bb)))
            f.write('\n')
            for inst in bb.insts:
                f.write('    ')
                self.print_inst(inst, f, slot_id_map)


class MachineBasicBlock:
    def __init__(self):
        self.func = None
        self._insts = []
        self.index = -1

        self.successors = []
        self.predecessors = []

    def split_basic_block(self, inst):
        new_mbb = MachineBasicBlock()

        split_idx = self._insts.index(inst)
        for inst in self._insts[split_idx:]:
            new_mbb._insts.append(inst)
            inst.mbb = new_mbb

        for idx in reversed(range(split_idx, len(self._insts))):
            self._insts.pop(idx)

        self.func.insert_bb(new_mbb, self.number + 1)

        return new_mbb

    def append_inst(self, inst):
        self.insert_inst(inst, len(self._insts))

    def insert_inst(self, inst, index):
        if inst.mbb is not None:
            raise ValueError("The instruction exists in another.")

        for inst_after in self._insts[index:]:
            inst_after.index += 1
        inst.mbb = self
        inst.index = index
        self._insts.insert(index, inst)

        reginfo = self.func.reg_info
        for operand in inst.operands:
            if operand.is_reg:
                reginfo.add_reg_operand_to_use(operand)

    def remove_inst(self, inst):
        if inst.mbb is not self:
            raise ValueError("The instruction exists in another.")

        reginfo = self.func.reg_info
        for operand in inst.operands:
            if operand.is_reg:
                reginfo.remove_reg_operand_to_use(operand)

        index = self._insts.index(inst)
        for inst_after in self._insts[index:]:
            inst_after.index -= 1
        inst.index = -1
        self._insts.pop(index)
        inst.mbb = None

    @property
    def first_terminator(self):
        term = None
        for inst in reversed(self._insts):
            if not inst.is_terminator:
                break

            term = inst

        if term is None:
            raise Exception(
                "The last instruction of th basic block must be terminator.")

        return term

    @property
    def insts(self):
        return tuple(self._insts)

    def remove_from_func(self):
        self.func.remove_bb(self)

    def add_successor(self, bb):
        self.successors.append(bb)
        bb.add_predecessor(self)

    def add_predecessor(self, bb):
        self.predecessors.append(bb)

    def remove_successor(self, bb):
        self.successors.remove(bb)
        bb.remove_predecessor(self)

    def remove_predecessor(self, bb):
        self.predecessors.remove(bb)

    @property
    def number(self):
        if self not in self.func.bbs:
            return -1
        return self.func.bbs.index(self)


class MachineOperand:
    def __init__(self, target_flags=0):
        self.target_flags = target_flags
        self.tied_to = -1
        self._inst = None

    @property
    def inst(self):
        return self._inst

    @inst.setter
    def inst(self, value):
        self._inst = value

    @property
    def is_def(self):
        return False

    @property
    def is_reg(self):
        return False

    @property
    def is_jti(self):
        return False

    @property
    def is_mbb(self):
        return False

    @property
    def is_tied(self):
        return self.tied_to >= 0

    @property
    def is_implicit(self):
        return False

    def print(self, f, slot_id_map):
        raise NotImplementedError()


class RegState(IntFlag):
    Non = 0x0
    Define = 0x2
    Implicit = 0x4
    Kill = 0x8
    Dead = 0x10
    Undef = 0x20
    EarlyClobber = 0x40
    Debug = 0x80
    InternalRead = 0x100
    Renamable = 0x200
    DefineNoRead = Define | Undef,
    ImplicitDefine = Implicit | Define,
    ImplicitKill = Implicit | Kill


class MOReg(MachineOperand):
    def __init__(self, reg, flags, subreg=None):
        super().__init__()
        assert(reg is not None)
        assert(isinstance(reg, (MachineRegister, MachineVirtualRegister)))
        self._reg = reg
        self.flags = flags
        self._subreg = subreg
        self.prev = None
        self.next = None

    @property
    def subreg(self):
        return self._subreg

    @subreg.setter
    def subreg(self, value):
        self._subreg = value

    def subst_phys_reg(self, reg_info):
        reg = self._reg.spec
        if self.subreg is not None:
            reg = reg_info.get_subreg(reg, self._subreg)
            self._subreg = None

        self._reg = MachineRegister(reg)

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, reg):
        assert(reg is not None)

        if reg == self._reg:
            return

        self.inst.remove_reg_operand_to_use(self)
        self._reg = reg
        self.inst.add_reg_operand_to_use(self)

    @property
    def is_def(self):
        return (self.flags & RegState.Define) == RegState.Define

    @property
    def is_use(self):
        return not self.is_def

    @property
    def is_implicit(self):
        return (self.flags & RegState.Implicit) == RegState.Implicit

    @property
    def is_kill(self):
        return (self.flags & RegState.Kill) == RegState.Kill

    @is_kill.setter
    def is_kill(self, value):
        if value:
            self.flags = self.flags | RegState.Kill
        else:
            self.flags = self.flags & ~RegState.Kill

    @property
    def is_implicit_define(self):
        return (self.flags & RegState.ImplicitDefine) == RegState.ImplicitDefine

    @property
    def is_dead(self):
        return (self.flags & RegState.Dead) == RegState.Dead

    @is_dead.setter
    def is_dead(self, value):
        if value:
            self.flags = self.flags | RegState.Dead
        else:
            self.flags = self.flags & ~RegState.Dead

    @property
    def is_renamable(self):
        return (self.flags & RegState.Renamable) == RegState.Renamable

    @is_renamable.setter
    def is_renamable(self, value):
        if value:
            self.flags = self.flags | RegState.Renamable
        else:
            self.flags = self.flags & ~RegState.Renamable

    @property
    def is_early_clobber(self):
        return (self.flags & RegState.EarlyClobber) == RegState.EarlyClobber

    @property
    def is_undef(self):
        return (self.flags & RegState.Undef) == RegState.Undef

    @property
    def is_reg(self):
        return True

    @property
    def is_phys(self):
        return isinstance(self.reg, MachineRegister)

    @property
    def is_virtual(self):
        return isinstance(self.reg, MachineVirtualRegister)

    def print(self, f, slot_id_map):
        from rachetwrench.codegen.spec import subregs

        if self.is_implicit_define:
            f.write("implicit-def ")
        elif self.is_implicit:
            f.write("implicit ")
        elif self.is_kill:
            f.write("killed ")
        elif self.is_dead:
            f.write("dead ")

        if self.is_def and not self.is_implicit:
            slot_id_map[self.reg] = len(slot_id_map)

        if isinstance(self.reg, MachineRegister):
            f.write('${}'.format(self.reg.spec.name))
            if self.subreg is not None:
                subreg = subregs[self.subreg]
                f.write(f'.{subreg.name}')
        else:
            f.write('%{}'.format(self.reg.vid))
            if self.subreg is not None:
                subreg = subregs[self.subreg]
                f.write(f'.{subreg.name}')

        if self.is_def and isinstance(self.reg, MachineVirtualRegister):
            f.write(f':{self.reg.regclass.name}')


class MOImm(MachineOperand):
    def __init__(self, value):
        super().__init__()
        from rachetwrench.ir.values import ConstantInt, ConstantFP

        assert(isinstance(value, int))
        self.val = value

    def print(self, f, slot_id_map):
        f.write(f'{self.val}')


class MOBasicBlock(MachineOperand):
    def __init__(self, mbb, target_flags):
        super().__init__(target_flags)
        self.mbb = mbb

    def print(self, f, slot_id_map):
        f.write(f'%bb.{id(self.mbb)}')

    @property
    def is_mbb(self):
        return True


class MOFrameIndex(MachineOperand):
    def __init__(self, index):
        super().__init__()
        self._index = index

    @property
    def index(self):
        return self._index

    def print(self, f, slot_id_map):
        f.write(f'%stack.{self.index}')


class MOTargetIndex(MachineOperand):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.offset = 0

    def print(self, f, slot_id_map):
        raise NotImplementedError()


class MOExternalSymbol(MachineOperand):
    def __init__(self, symbol, target_flags=0):
        super().__init__(target_flags)
        self.symbol = symbol
        self.offset = 0

    def print(self, f, slot_id_map):
        f.write(f'@{self.symbol}')


class MORegisterMask(MachineOperand):
    def __init__(self, mask, target_flags=0):
        assert(isinstance(mask, list))
        super().__init__(target_flags)
        self.mask = mask

    def print(self, f, slot_id_map):
        f.write(f'mask')


class MOConstantPoolIndex(MachineOperand):
    def __init__(self, index, target_flags=0):
        super().__init__(target_flags)
        self.index = index
        self.offset = 0

    def print(self, f, slot_id_map):
        f.write(f'%const.{self.index}')


class MOGlobalAddress(MachineOperand):
    def __init__(self, value, offset=0, target_flags=0):
        super().__init__(target_flags)
        self.value = value
        self.offset = 0

    def print(self, f, slot_id_map):
        f.write(f'@{self.value.name}')


class MachineInstruction:
    def __init__(self, opcode):
        self.mbb = None
        self.index = -1
        self.opcode = opcode
        self._operands = []
        self.comment = ""

    @property
    def is_copy(self):
        from rachetwrench.codegen.spec import TargetDagOps
        return self.opcode == TargetDagOps.COPY

    @property
    def is_subreg_to_reg(self):
        from rachetwrench.codegen.spec import TargetDagOps
        return self.opcode == TargetDagOps.SUBREG_TO_REG

    @property
    def is_insert_subreg(self):
        from rachetwrench.codegen.spec import TargetDagOps
        return self.opcode == TargetDagOps.INSERT_SUBREG

    @property
    def is_extract_subreg(self):
        from rachetwrench.codegen.spec import TargetDagOps
        return self.opcode == TargetDagOps.EXTRACT_SUBREG

    @property
    def is_full_copy(self):
        return self.is_copy and not self.operands[0].subreg and not self.operands[1].subreg

    @property
    def is_copy_like(self):
        return self.is_copy or self.is_subreg_to_reg

    @property
    def is_identity_copy(self):
        return self.is_copy and self.operands[0].reg == self.operands[1].reg and self.operands[0].subreg == self.operands[1].subreg

    @property
    def is_call(self):
        from rachetwrench.codegen.spec import MachineInstructionDef
        if isinstance(self.opcode, MachineInstructionDef):
            return self.opcode.is_call
        return False

    @property
    def is_terminator(self):
        from rachetwrench.codegen.spec import MachineInstructionDef
        if isinstance(self.opcode, MachineInstructionDef):
            return self.opcode.is_terminator
        return False

    @property
    def is_branch(self):
        from rachetwrench.codegen.spec import MachineInstructionDef
        if isinstance(self.opcode, MachineInstructionDef):
            return self.opcode.is_branch
        return False

    @property
    def is_unconditional_branch(self):
        from rachetwrench.codegen.spec import MachineInstructionDef
        if isinstance(self.opcode, MachineInstructionDef):
            return self.opcode.is_branch and self.opcode.is_barrier
        return False

    @property
    def is_conditional_branch(self):
        from rachetwrench.codegen.spec import MachineInstructionDef
        if isinstance(self.opcode, MachineInstructionDef):
            return self.opcode.is_branch and not self.opcode.is_barrier
        return False

    def add_reg_operand_to_use(self, operand):
        if self.mbb is not None:
            reginfo = self.mbb.func.reg_info
            reginfo.add_reg_operand_to_use(operand)

    def remove_reg_operand_to_use(self, operand):
        if self.mbb is not None:
            reginfo = self.mbb.func.reg_info
            reginfo.remove_reg_operand_to_use(operand)

    def add_operand(self, operand: MachineOperand):
        assert(operand not in self._operands)
        assert(operand.inst is None)
        operand.inst = self
        self._operands.append(operand)
        if operand.is_reg:
            self.add_reg_operand_to_use(operand)

        return operand

    def remove_operand(self, idx):
        operand = self._operands[idx]
        if operand.is_reg:
            self.remove_reg_operand_to_use(operand)
        operand.inst = None
        self._operands.pop(idx)

    def replace_operand(self, idx, operand):
        to_remove = self._operands[idx]
        if to_remove is not None and to_remove.is_reg:
            self.remove_reg_operand_to_use(to_remove)

        self._operands[idx] = operand
        to_remove.inst = None
        operand.inst = self

        if operand is not None and operand.is_reg:
            self.add_reg_operand_to_use(operand)

    @property
    def operands(self):
        class Indexer:
            def __init__(self, inst):
                self._inst = inst

            def __setitem__(self, idx, operand):
                self._inst.replace_operand(idx, operand)

            def __getitem__(self, idx):
                return self._inst._operands[idx]

            def __len__(self):
                return len(self._inst._operands)
        return Indexer(self)

    def add_imm(self, value):
        return self.add_operand(MOImm(value))

    def add_def_reg(self, value):
        return self.add_reg(value, RegState.Define)

    def add_reg(self, value, flags):
        return self.add_operand(MOReg(value, flags))

    def add_global_address(self, value, target_flags=0):
        return self.add_operand(MOGlobalAddress(value, 0, target_flags))

    def add_mbb(self, value, target_flags=0):
        return self.add_operand(MOBasicBlock(value, target_flags))

    def add_frame_index(self, value):
        return self.add_operand(MOFrameIndex(value))

    def add_target_index(self, value):
        return self.add_operand(MOTargetIndex(value))

    def add_constant_pool_index(self, value, target_flags=0):
        return self.add_operand(MOConstantPoolIndex(value, target_flags))

    def add_external_symbol(self, value, target_flags=0):
        return self.add_operand(MOExternalSymbol(value, target_flags))

    def add_reg_mask(self, value):
        return self.add_operand(MORegisterMask(value))

    def insert_before(self, inst):
        if inst.mbb is None:
            raise ValueError("The inst is not inserted.")
        assert(self not in inst.mbb.insts)
        idx = inst.mbb.insts.index(inst)
        inst.mbb.insert_inst(self, idx)

    def insert_after(self, inst):
        if inst.mbb is None:
            raise ValueError("The inst is not inserted.")
        assert(self not in inst.mbb.insts)
        idx = inst.mbb.insts.index(inst) + 1
        inst.mbb.insert_inst(self, idx)

    def remove(self):
        self.mbb.remove_inst(self)

    @property
    def prev_inst(self):
        mbb = self.mbb
        index = mbb.insts.index(self)
        if index == 0:
            return None

        return mbb.insts[index - 1]

    @property
    def next_inst(self):
        mbb = self.mbb
        index = mbb.insts.index(self)
        if index == len(mbb.insts):
            return None

        return mbb.insts[index + 1]
