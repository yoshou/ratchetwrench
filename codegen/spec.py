#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import OrderedDict
from codegen.types import MachineValueType, ValueType


class SubRegDescription:
    def __init__(self, name, size_in_bit, offset_in_bit=0):
        self.name = name
        self.size_in_bit = 0
        self.offset_in_bit = 0


class ComposedSubRegDescription:
    def __init__(self, name, subreg_a, subreg_b):
        self.name = name
        self.subreg_a = subreg_a
        self.subreg_b = subreg_b


class MachineRegisterDef:

    regs = []

    def __init__(self, name, subregs=[], subreg_descs=[], aliases=[], encoding=0):
        assert(len(subregs) == len(subreg_descs))

        self.name = name
        self.explicit_subregs = list(subregs)
        self.explicit_subreg_descs = list(subreg_descs)
        self.explicit_aliases = list(aliases)
        self.encoding = encoding

        self.subregs = []
        self.aliases = list(aliases)

        self.reg_units = set()
        self.superregs = set()

        MachineRegisterDef.regs.append(self)


def dfs(graph, v, visited, action):
    if v in visited:
        return

    visited.add(v)
    action(v)

    adjs = list(graph[v].values())

    for node, adj in graph[v].items():
        if adj == 1 or adj == -1:
            dfs(graph, node, visited, action)


def post_dfs(graph, v, visited, action):
    if v in visited:
        return

    visited.add(v)

    adjs = list(graph[v].values())

    for node, adj in graph[v].items():
        if adj == 1 or adj == -1:
            dfs(graph, node, visited, action)

    action(v)


def compute_connected(graph):
    regs = list(graph.keys())

    visited = set()
    for reg in regs:
        if reg not in visited:
            nodes = []
            dfs(graph, reg, visited, nodes.append)
            yield nodes


def compute_reg_graph():
    regs = MachineRegisterDef.regs

    reg_graph = {}
    for reg1 in regs:
        reg_graph[reg1] = {}
        for reg2 in regs:
            reg_graph[reg1][reg2] = 0

    for a_reg in regs:
        for subreg in a_reg.subregs:
            reg_graph[a_reg][subreg.reg] = 1
            reg_graph[subreg.reg][a_reg] = -1

    return reg_graph


def compute_reg_groups(reg_graph):
    reg_groups = list(compute_connected(reg_graph))

    for reg_group in reg_groups:
        for reg in reg_group:
            reg.reg_group = reg_group

    return reg_groups

# Register overlapping information.
# The leaf register is overlapping itself.


class RegUnit:
    def __init__(self, reg_a, reg_b=None):
        self.reg_a = reg_a
        self.reg_b = reg_b


class SubRegAndIndex:
    def __init__(self, reg, idx):
        self.reg = reg
        self.idx = idx


def compute_reg_subregs_all(reg_graph):
    regs = list(reg_graph.keys())

    def compute_reg_subregs(reg, computed):
        if reg in computed:
            return

        for subreg, subreg_index in zip(reg.explicit_subregs, reg.explicit_subreg_descs):
            reg.subregs.append(SubRegAndIndex(subreg, subreg_index))

        for subreg in reg.explicit_subregs:
            compute_reg_subregs(subreg, computed)

            for child_reg in subreg.subregs:
                reg.subregs.append(child_reg)

        for subreg in reg.explicit_subregs:
            reg.reg_units |= subreg.reg_units

        for alias in reg.explicit_aliases:
            regunit = RegUnit(reg, alias)

            reg.reg_units |= regunit
            alias.reg_units |= regunit

        if len(reg.reg_units) == 0:
            reg.reg_units |= {RegUnit(reg)}

    computed = set()
    for reg in regs:
        compute_reg_subregs(reg, computed)


def compute_reg_superregs_all(reg_graph):
    regs = list(reg_graph.keys())

    def compute_reg_superregs(reg, computed):
        if reg in computed:
            return

        for subreg in reg.subregs:
            compute_reg_superregs(subreg.reg, computed)

        for subreg in reg.subregs:
            subreg.reg.superregs.add(reg)

    computed = set()
    for reg in regs:
        compute_reg_superregs(reg, computed)


def iter_reg_units(reg):
    return iter(reg.reg_units)


def iter_reg_unit_roots(unit):
    assert(unit.reg_a is not None)
    yield unit.reg_a
    if unit.reg_b is not None:
        yield unit.compute_reg_subregs_all


def iter_super_regs(reg):
    def iter_super_regs_dfs(reg, visited):
        if reg in visited:
            return

        yield reg

        for super_reg in reg.superregs:
            yield from iter_super_regs_dfs(super_reg, visited)

    yield from iter_super_regs_dfs(reg, set())


def iter_sub_regs(reg):
    def iter_sub_regs_dfs(reg, visited):
        if reg in visited:
            return

        for subreg, subreg_idx in zip(reg.subregs, reg.subreg_idx):
            yield subreg, subreg_idx

        for subreg in reg.subregs:
            yield from iter_sub_regs_dfs(subreg, visited)

    yield from iter_sub_regs_dfs(reg, set())


def iter_reg_aliases(reg):
    for reg_unit in iter_reg_units(reg):
        for root in iter_reg_unit_roots(reg_unit):
            for reg in iter_super_regs(root):
                yield reg


subregs = []


def def_subreg(name, size, offset=0):
    subreg = SubRegDescription(name, size, offset)
    subregs.append(subreg)
    return subreg


def def_composed_subreg(name, subreg_a, subreg_b):
    subreg = ComposedSubRegDescription(name, subreg_a, subreg_b)
    subregs.append(subreg)
    return subreg


def def_reg(*args, **kwargs):
    return MachineRegisterDef(*args, **kwargs)


regclasses = []


def def_regclass(*args, **kwargs):
    regclass = MachineRegisterClassDef(*args, **kwargs)
    regclasses.append(regclass)
    return regclass


def infer_subregclass_and_subreg(regclass):
    subreg_map = {}

    # Make pairs of subreg_idx and registers for all registers in the regclass.
    for reg in regclass.regs:
        for subreg_subregidx in reg.subregs:
            subreg, subreg_idx = subreg_subregidx.reg, subreg_subregidx.idx
            if subreg_idx not in subreg_map:
                subreg_map[subreg_idx] = []
            subreg_map[subreg_idx].append(reg)

    for subreg_idx in subregs:
        if subreg_idx not in subreg_map:
            continue

        regs = subreg_map[subreg_idx]

        # If registers of subreg_idx consist of a defined regclass, the registers corresponding subreg_idx is the class.
        if len(regs) == len(regclass.regs):
            regclass.subclass_and_subregs[subreg_idx] = regclass
            continue

        # Create a new regclass for the registers of the subreg_idx.
        # Types can't be infered.
        subregclass = def_regclass(
            regclass.name + "_with_" + subreg_idx.name, tys=regclass.tys, align=regclass.align, regs=regs)
        regclass.subclass_and_subregs[subreg_idx] = subregclass


NOREG = def_reg("noreg")


class MachineOperandDef:
    pass


class MachineRegisterClassDef(MachineOperandDef):
    def __init__(self, name, tys, align, regs):
        self.name = name
        self.tys = [ty if isinstance(
            ty, MachineValueType) else MachineValueType(ty) for ty in tys]
        self.align = align
        self.regs = regs
        self.subclass_and_subregs = {}


class ValueOperandDef(MachineOperandDef):
    def __init__(self, ty):
        self.ty = ty

    @property
    def operand_info(self):
        return [self]


class Constraint:
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2


class MachineInstructionDef:
    def __init__(self, mnemonic, outs, ins, patterns=[], uses=[], defs=[], constraints=[], size=0, is_compare=False, is_terminator=False, is_call=False):
        self.mnemonic = mnemonic
        self.outs = OrderedDict(outs)
        self.ins = OrderedDict(ins)
        self.patterns = patterns
        self.uses = uses
        self.defs = defs
        self.constraints = constraints
        self.size = size
        self.is_compare = is_compare
        self.is_terminator = is_terminator
        self.is_call = is_call

    @property
    def operands(self):
        operands = []
        for name, ty in self.outs.items():
            if isinstance(ty, MachineRegisterClassDef):
                operands.append((name, ty))
            elif isinstance(ty, ValueOperandDef):
                for i, info in enumerate(ty.operand_info):
                    operands.append((f"{name}.{i}", info))
            else:
                raise NotImplementedError()
        for name, ty in self.ins.items():
            if isinstance(ty, MachineRegisterClassDef):
                operands.append((name, ty))
            elif isinstance(ty, ValueOperandDef):
                for i, info in enumerate(ty.operand_info):
                    operands.append((f"{name}.{i}", info))
            else:
                raise NotImplementedError()

        return operands

    @property
    def num_operands(self):
        return len(self.operands)

    def __repr__(self):
        return self.mnemonic

    def __str__(self):
        return self.mnemonic


def def_inst(*args, **kwargs):
    return MachineInstructionDef(*args, **kwargs)


class CCArgFlags:
    def __init__(self):
        self.val_align = 0
        self.val_size = 0
        self.addr_space = 0


class CallingConvArg:
    def __init__(self, arg_ty, mvt, arg_idx, offset, flags):
        self.arg_ty = arg_ty
        self.mvt = mvt
        self.arg_idx = arg_idx
        self.offset = offset
        self.flags = flags


class CallingConvReturn:
    def __init__(self, arg_ty, mvt, arg_idx, offset, flags):
        self.arg_ty = arg_ty
        self.mvt = mvt
        self.arg_idx = arg_idx
        self.offset = offset
        self.flags = flags


class CCArgReg:
    def __init__(self, arg_idx, vt, loc_vt, loc_info, reg, flags):
        self.arg_idx = arg_idx
        self.vt = vt
        self.loc_vt = loc_vt
        self.loc_info = loc_info
        self.reg = reg
        self.flags = flags


class CCArgMem:
    def __init__(self, arg_idx, vt, loc_vt, loc_info, offset, flags):
        self.arg_idx = arg_idx
        self.vt = vt
        self.loc_vt = loc_vt
        self.loc_info = loc_info
        self.offset = offset
        self.flags = flags


from enum import Enum, auto


class CCArgLocInfo(Enum):
    Full = auto()

    SExt = auto()  # sign extend
    ZExt = auto()  # zero extend

    BCvt = auto()  # bit convert

    Indirect = auto()


from enum import Enum, auto


class CallingConvID(Enum):
    C = auto()
    Fast = auto()
    X86_StdCall = auto()
    X86_FastCall = auto()
    X86_VectorCall = auto()
    X86_INTR = auto()
    X86_RegCall = auto()
    X86_64_SystemV = auto()
    Win64 = auto()


class CallingConvState:
    def __init__(self, calling_conv, mfunc):
        self.calling_conv = calling_conv
        self.mfunc = mfunc
        self.stack_offset = 0
        self.stack_maxalign = 1
        self.values = []
        self.allocated_regs = set()
        self.calling_conv_id = calling_conv.id

    def compute_arguments_layout(self, args):
        for i, arg in enumerate(args):
            vt = args[i].mvt
            flags = args[i].flags

            self.calling_conv.allocate_argument(
                i, vt, vt, CCArgLocInfo.Full, flags, self)

    def compute_returns_layout(self, returns):
        for i, ret in enumerate(returns):
            vt = returns[i].mvt
            flags = returns[i].flags

            self.calling_conv.allocate_return(
                i, vt, vt, CCArgLocInfo.Full, flags, self)

    def assign_reg_value(self, arg_idx, vt, loc_vt, loc_info: CCArgLocInfo, reg, flags: CCArgFlags):
        self.values.append(CCArgReg(arg_idx, vt, loc_vt, loc_info, reg, flags))

    def assign_stack_value(self, arg_idx, vt, loc_vt, loc_info: CCArgLocInfo, offset, flags: CCArgFlags):
        self.values.append(
            CCArgMem(arg_idx, vt, loc_vt, loc_info, offset, flags))

    def alloc_reg_from_list(self, regs, shadow_regs=None):
        for i in range(len(regs)):
            reg = regs[i]
            if reg not in self.allocated_regs:
                reg_aliases = iter_reg_aliases(reg)
                for reg_alias in reg_aliases:
                    self.allocated_regs.add(reg_alias)

                if shadow_regs is not None:
                    shadow_reg = shadow_regs[i]
                    shadow_reg_aliases = iter_reg_aliases(shadow_reg)
                    for shadow_reg_alias in shadow_reg_aliases:
                        self.allocated_regs.add(shadow_reg_alias)
                return reg

        return None

    def alloc_stack(self, size, alignment):
        self.stack_offset = int(
            int((self.stack_offset + alignment - 1) / alignment) * alignment)
        offset = self.stack_offset
        self.stack_offset += size

        self.stack_maxalign = max([self.stack_maxalign, alignment])

        return offset


class CallingConv:
    def __init__(self):
        pass

    def can_lower_return(self, func):
        raise NotImplementedError()


class StackGrowsDirection(Enum):
    Up = auto()
    Down = auto()


class TargetFrameLowering:
    def __init__(self, alignment):
        self.stack_alignment = alignment

    @property
    def stack_grows_direction(self):
        raise NotImplementedError()


class TargetLowering:
    def __init__(self):
        pass

    def lower(self, node, dag):
        raise NotImplementedError()

    def lower_arguments(self, func, builder):
        raise NotImplementedError()

    def lower_prolog(self, func):
        raise NotImplementedError()

    def lower_epilog(self, func):
        raise NotImplementedError()

    def lower_optimal_memory_op(self, size, src_op, dst_op, src_align, dst_align, builder):
        raise NotImplementedError()

    def get_machine_vreg(self, ty):
        raise NotImplementedError()

    def eliminate_call_frame_pseudo_inst(self, func, inst):
        raise NotImplementedError()


class TargetInstInfo:
    def __init__(self):
        pass

    def copy_reg_to_stack(self, reg, stack_slot, inst):
        raise NotImplementedError()

    def copy_reg_from_stack(self, reg, stack_slot, inst):
        raise NotImplementedError()

    def eliminate_frame_index(self, func, inst, idx):
        raise NotImplementedError()

    def optimize_compare_inst(self, func, inst):
        raise NotImplementedError()

    def expand_post_ra_pseudo(self, inst):
        raise NotImplementedError()


class TargetRegisterInfo:
    def __init__(self):
        pass

    def get_ordered_regs(self, regclass):
        raise NotImplementedError()

    def get_regclass_from_reg(self, reg):
        raise NotImplementedError()

    def get_register_type(self, vt):
        if vt in [MachineValueType(e) for e in ValueType]:
            return vt

        raise NotImplementedError()

    def get_register_count(self, vt):
        if vt in [MachineValueType(e) for e in ValueType]:
            return 1

        raise NotImplementedError()


class InstructionSelector:
    def __init__(self):
        pass

    def select(self, node, dag):
        raise NotImplementedError()


class Legalizer:
    def __init__(self):
        pass

    def legalize_node_type(self, node, dag, legalized):
        raise NotImplementedError()


from enum import Enum, auto


class ArchType(Enum):
    Unknown = auto()
    ARM = "arm"
    ARM64 = "arm64"
    Thumb = "thumb"
    X86 = "x86"
    X86_64 = "x86_64"


class OS(Enum):
    Unknown = auto()
    Linux = auto()
    Windows = auto()


class Environment(Enum):
    Unknown = auto()
    GNU = auto()
    MSVC = auto()
    EABI = auto()


class ObjectFormatType(Enum):
    Unknown = auto()
    COFF = auto()
    ELF = auto()


class Triple:
    def __init__(self, arch=ArchType.Unknown, os=OS.Unknown, env=Environment.Unknown, objformat=ObjectFormatType.Unknown):
        self.arch = arch
        self.os = os
        self.env = env
        self.objformat = objformat


class TargetInfo:
    def __init__(self, triple):
        self.triple = triple

    def get_inst_info(self) -> TargetInstInfo:
        raise NotImplementedError()

    def get_lowering(self) -> TargetLowering:
        raise NotImplementedError()

    def get_register_info(self) -> TargetRegisterInfo:
        raise NotImplementedError()

    def get_calling_conv(self) -> CallingConv:
        raise NotImplementedError()

    def get_instruction_selector(self) -> InstructionSelector:
        raise NotImplementedError()

    def get_frame_lowering(self) -> TargetFrameLowering:
        raise NotImplementedError()
