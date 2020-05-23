#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import html
from enum import Enum
from collections import namedtuple
import pygraphviz as pgv

from ir.values import *
from ir.types import PrimitiveType, StructType, PointerType, LabelType, VoidType, VectorType, ArrayType

from ir.types import f32, f64


class SlotTracker:
    def __init__(self):
        self.func_inst_id = {}

    def track(self, func):
        label = 0

        for arg in func.args:
            self.func_inst_id[arg] = f"%{label}"
            label += 1

        for block in func.blocks:
            self.func_inst_id[block] = f"%{label}"
            label += 1

            for inst in block.insts:
                if isinstance(inst.ty, (VoidType)):
                    continue

                self.func_inst_id[inst] = f"%{label}"
                label += 1

    @property
    def local_id(self):
        return self.func_inst_id

    def __getitem__(self, value):
        return self.func_inst_id[value]


def get_type_name(ty):
    if isinstance(ty, PrimitiveType):
        if ty.name == "f16":
            return "half"
        elif ty.name == "f32":
            return "float"
        elif ty.name == "f64":
            return "double"
        elif ty.name == "f128":
            return "fp128"
        return ty.name
    if isinstance(ty, VectorType):
        return f"<{ty.size} x {get_type_name(ty.elem_ty)}>"
    if isinstance(ty, ArrayType):
        return f"[{ty.size} x {get_type_name(ty.elem_ty)}]"
    if isinstance(ty, StructType):
        return f"%{ty.name}"
    if isinstance(ty, PointerType):
        return f"{get_type_name(ty.elem_ty)}*"
    if isinstance(ty, LabelType):
        return ty.name
    if isinstance(ty, VoidType):
        return ty.name
    if isinstance(ty, FunctionType):
        param_ty_list = ", ".join(
            [f"{get_type_name(param)}" for param in ty.params if param and not isinstance(param, VoidType)])
        return_ty_name = get_type_name(ty.return_ty)

        if ty.is_variadic:
            param_ty_list += ", ..."

        return f"{return_ty_name} ({param_ty_list})"

    raise ValueError("Invalid type.")


def get_type_name2(ty, module):
    if isinstance(ty, PrimitiveType):
        if ty.name == "f16":
            return "half"
        elif ty.name == "f32":
            return "float"
        elif ty.name == "f64":
            return "double"
        elif ty.name == "f128":
            return "fp128"
        return ty.name
    if isinstance(ty, VectorType):
        return f"<{ty.size} x {get_type_name(ty.elem_ty)}>"
    if isinstance(ty, ArrayType):
        return f"[{ty.size} x {get_type_name(ty.elem_ty)}]"
    if isinstance(ty, StructType):
        if ty.name != "":
            return f"%{ty.name}"

        field_ty_names = ", ".join([get_type_name2(field_ty, module)
                                    for field_ty in ty.fields])
        if ty.is_packed:
            return f"<{{{field_ty_names}}}>"
        return f"{{{field_ty_names}}}"
    if isinstance(ty, PointerType):
        return f"{get_type_name(ty.elem_ty)}*"
    if isinstance(ty, LabelType):
        return ty.name
    if isinstance(ty, VoidType):
        return ty.name
    if isinstance(ty, FunctionType):
        param_ty_list = ", ".join(
            [f"{get_type_name(param)}" for param in ty.params if param and not isinstance(param, VoidType)])
        return_ty_name = get_type_name(ty.return_ty)

        if ty.is_variadic:
            param_ty_list += ", ..."

        return f"{return_ty_name} ({param_ty_list})"

    raise ValueError("Invalid type.")


def get_value_type(value):
    return get_type_name(value.ty)


def get_value_type2(value, module):
    return get_type_name2(value.ty, module)


def float_to_hex(f):
    import struct
    f = struct.unpack('f', struct.pack('f', f))[0]
    return "0x{:X}".format(struct.unpack('<Q', struct.pack('<d', f))[0])


def double_to_hex(f):
    import struct
    return "0x{:X}".format(struct.unpack('<Q', struct.pack('<d', f))[0])


def get_value_name(value):
    from ir.types import i8

    if value is None:
        return ""

    if isinstance(value, ConstantInt):
        return str(value)
    elif isinstance(value, ConstantFP):
        if value.ty == f32:
            return float_to_hex(value.value)
        elif value.ty == f64:
            return double_to_hex(value.value)
        raise NotImplementedError()
    elif isinstance(value, ConstantPointerNull):
        return "null"
    elif isinstance(value, ConstantVector):
        elem_ty_name = get_type_name(value.ty.elem_ty)
        lst = ', '.join(
            [f"{elem_ty_name} {get_value_name(value)}" for value in value.values])
        return f"<{lst}>"
    elif isinstance(value, ConstantArray):
        # if value.ty.elem_ty == i8:
        #     elems = [get_value_name(elem) for elem in value.values]
        #     return f'c"{"".join(elems)}"'

        elem_ty_name = get_type_name(value.ty.elem_ty)
        lst = ', '.join(
            [f"{elem_ty_name} {get_value_name(value)}" for value in value.values])
        return f"[{lst}]"
    elif isinstance(value, ConstantStruct):
        lst = ', '.join(
            [f"{get_type_name(field_value.ty)} {get_value_name(field_value)}" for field_value in value.values])
        return f"{{{lst}}}"
    elif isinstance(value, (Function, GlobalVariable)):
        return f"{value.value_name}"
    elif isinstance(value, IntToPtrInst):
        return f"inttoptr ({get_value_type(value.rs)} {get_value_name(value.rs)} to {get_value_type(value)})"
    elif isinstance(value, BitCastInst):
        return f"bitcast ({get_value_type(value.rs)} {get_value_name(value.rs)} to {get_value_type(value)})"
    elif isinstance(value, TruncInst):
        return f"trunc ({get_value_type(value.rs)} {get_value_name(value.rs)} to {get_value_type(value)})"
    else:
        raise ValueError()


def get_ordering_name(value):
    assert(value != AtomicOrdering.NotAtomic)

    TABLE = {
        AtomicOrdering.Unordered: "unordered",
        AtomicOrdering.Monotonic: "motononic",
        AtomicOrdering.Acquire: "acquire",
        AtomicOrdering.Release: "release",
        AtomicOrdering.AcquireRelease: "acq_rel",
        AtomicOrdering.SequentiallyConsistent: "seq_cst"
    }

    return TABLE[value]


def print_inst(inst, slot_id_map):
    def get_value_name(value):
        if value is None:
            return ""

        if isinstance(value, ConstantInt):
            return value
        elif isinstance(value, ConstantFP):
            if value.ty == f32:
                return float_to_hex(value.value)
            elif value.ty == f64:
                return double_to_hex(value.value)
            raise NotImplementedError()
        elif isinstance(value, ConstantPointerNull):
            return "null"
        elif isinstance(value, ConstantArray):
            elem_ty_name = get_type_name(value.ty.elem_ty)
            lst = ', '.join(
                [f"{elem_ty_name} {str(value)}" for value in value.values])
            return f"[{lst}]"
        elif isinstance(value, ConstantVector):
            elem_ty_name = get_type_name(value.ty.elem_ty)
            lst = ', '.join(
                [f"{elem_ty_name} {str(value)}" for value in value.values])
            return f"<{lst}>"
        elif isinstance(value, (Function, GlobalVariable)):
            return f"{value.value_name}"
        elif isinstance(value, InlineAsm):
            strs = ["asm"]
            if value.has_side_effect:
                strs.append("sideeffect")

            strs.append(f'"{value.asm_string}"')
            strs.append(f', "{value.constraints}"')

            return " ".join(strs)
        else:
            # if value.has_name:
            #     return f"{value.value_name}"
            # else:
            t = type(value)
            return slot_id_map[value]

    module = inst.block.func.module

    if isinstance(inst, ReturnInst):
        if len(inst.operands) > 0:
            return f"ret {get_value_type(inst.rs)} {get_value_name(inst.rs)}"
        else:
            return f"ret void"

    if isinstance(inst, JumpInst):
        return f"br {get_value_type(inst.goto_target)} {get_value_name(inst.goto_target)}"

    if isinstance(inst, BranchInst):
        return f"br {get_value_type(inst.cond)} {get_value_name(inst.cond)}, {get_value_type(inst.then_target)} {get_value_name(inst.then_target)}, {get_value_type(inst.else_target)} {get_value_name(inst.else_target)}"

    if isinstance(inst, SwitchInst):
        case_lines = [
            f"\t\t{get_value_type(value)} {get_value_name(value)}, label {get_value_name(block)}" for value, block in inst.cases]

        cases = "\n".join(case_lines)

        assert(inst.default)

        return f"switch {get_value_type(inst.value)} {get_value_name(inst.value)}, label {get_value_name(inst.default)} [\n{cases}\n\t]"

    if isinstance(inst, LoadInst):
        return f"{get_value_name(inst)} = load {get_value_type(inst)}, {get_value_type(inst.rs)} {get_value_name(inst.rs)}"

    if isinstance(inst, StoreInst):
        return f"store {get_value_type(inst.rs)} {get_value_name(inst.rs)}, {get_value_type(inst.rd)} {get_value_name(inst.rd)}"

    if isinstance(inst, FenceInst):
        return f'fence syncscope("{str(inst.syncscope.value.name)}") {get_ordering_name(inst.ordering)}'

    if isinstance(inst, GetElementPtrInst):
        idx_list = ", ".join(
            [f"{get_value_type(idx)} {get_value_name(idx)}" for idx in inst.idx])
        inbounds = "inbounds " if inst.inbounds else ""
        return f"{get_value_name(inst)} = getelementptr {inbounds}{get_type_name(inst.pointee_ty.elem_ty)}, {get_value_type(inst.rs)} {get_value_name(inst.rs)}, {idx_list}"

    if isinstance(inst, ExtractValueInst):
        idx_list = ", ".join(
            [f"{idx}" for idx in inst.idx])

        return f"{get_value_name(inst)} = extractvalue {get_value_type2(inst.value, module)} {get_value_name(inst.value)}, {idx_list}"

    if isinstance(inst, BitCastInst):
        return f"{get_value_name(inst)} = bitcast {get_value_type(inst.rs)} {get_value_name(inst.rs)} to {get_value_type(inst)}"

    if isinstance(inst, BinaryInst):
        return f"{get_value_name(inst)} = {inst.op} {get_value_type(inst.rs)} {get_value_name(inst.rs)}, {get_value_name(inst.rt)}"

    if isinstance(inst, CmpInst):
        return f"{get_value_name(inst)} = icmp {inst.op} {get_value_type(inst.rs)} {get_value_name(inst.rs)}, {get_value_name(inst.rt)}"

    if isinstance(inst, FCmpInst):
        return f"{get_value_name(inst)} = fcmp {inst.op} {get_value_type(inst.rs)} {get_value_name(inst.rs)}, {get_value_name(inst.rt)}"

    if isinstance(inst, CallInst):
        arg_list = ", ".join(
            [f"{get_value_type(arg)} {get_value_name(arg)}" for arg in inst.args])
        return_ty_name = get_type_name2(
            inst.func_ty.return_ty, module)

        if isinstance(inst.callee, Function):
            func_ty_name = get_type_name(inst.func_ty)

            ty_or_fnty = func_ty_name if inst.callee.is_variadic else return_ty_name
        else:
            ty_or_fnty = return_ty_name

        if isinstance(inst.ty, VoidType):
            return f"call {ty_or_fnty} {get_value_name(inst.callee)}({arg_list})"
        else:
            return f"{get_value_name(inst)} = call {ty_or_fnty} {get_value_name(inst.callee)}({arg_list})"

    if isinstance(inst, AllocaInst):
        size_part = ""
        if inst.count.value != 1:
            size_part = f", {get_value_type(inst.count)} {get_value_name(inst.count)}"
        return f"{get_value_name(inst)} = alloca {get_type_name(inst.alloca_ty)}{size_part}, align {inst.align}"

    if isinstance(inst, CastInst):
        return f"{get_value_name(inst)} = {inst.op} {get_value_type(inst.rs)} {get_value_name(inst.rs)} to {get_value_type(inst)}"

    if isinstance(inst, InsertElementInst):
        return f"{get_value_name(inst)} = insertelement {get_value_type(inst.vec)} {get_value_name(inst.vec)}, {get_value_type(inst.elem)} {get_value_name(inst.elem)}, {get_value_type(inst.idx)} {get_value_name(inst.idx)}"

    if isinstance(inst, ExtractElementInst):
        return f"{get_value_name(inst)} = extractelement {get_value_type(inst)} {get_value_name(inst.vec)}, {get_value_type(inst.idx)} {get_value_name(inst.idx)}"

    if isinstance(inst, PHINode):
        values = [
            f"[{get_value_name(value)}, {get_value_name(block)}]" for block, value in inst.values.items()]
        return f"{get_value_name(inst)} = phi {get_value_type(inst)} {', '.join(values)}"

    raise Exception


def print_block(block, slot_id_map, indent=4):
    lines = []
    lines.append(f"; <label>:{slot_id_map[block][1:]}")
    lines.extend([print_inst(inst, slot_id_map) for inst in block.insts])

    lines = list(map(lambda l: (' '*indent) + l, lines))
    return "\n".join(lines)


def get_arg_string(arg):
    attrs = " ".join([str(attr.kind.value) for attr in arg.attrs])
    return f"{get_type_name(arg.ty)} {attrs}"


def print_function(func, indent=0):
    lines = []

    slot_id_map = SlotTracker()
    slot_id_map.track(func)

    arg_list = ", ".join(
        [get_arg_string(arg) for arg in func.args])
    if func.is_variadic:
        arg_list = arg_list + ",..."

    if func.comdat:
        if func.comdat.name == func.name:
            comdat = "comdat"
        else:
            comdat = f"comdat(${func.comdat.name})"
    else:
        comdat = ""

    if func.is_declaration:
        lines.append(
            f"declare {get_type_name(func.return_ty)} {func.value_name}({arg_list});")
    else:
        lines.append(
            f"define {get_type_name(func.return_ty)} {func.value_name}({arg_list}) {comdat} {{")
        lines.extend([print_block(block, slot_id_map) +
                      "\n" for block in func.blocks])
        lines.append("}")

    lines = list(map(lambda l: (' '*indent) + l, lines))
    return "\n".join(lines)


def get_thread_local(value):
    from ir.values import ThreadLocalMode

    if value.thread_local == ThreadLocalMode.NotThreadLocal:
        return ""

    if value.thread_local == ThreadLocalMode.GeneralDynamicTLSModel:
        return "thread_local"
    elif value.thread_local == ThreadLocalMode.LocalDynamicTLSModel:
        return "thread_local(localdynamic)"
    elif value.thread_local == ThreadLocalMode.InitialExecTLSModel:
        return "thread_local(initialexec)"
    elif value.thread_local == ThreadLocalMode.LocalExecTLSModel:
        return "thread_local(localexec)"

    return ""


def get_linkage(value):
    from ir.values import GlobalLinkage

    if value.linkage == GlobalLinkage.External:
        return "external"
    elif value.linkage == GlobalLinkage.Internal:
        return "internal"

    return ""


def get_comdat_kind(value):
    if value.kind == ComdatKind.Any:
        return "any"

    return ""


def print_module(module):
    lines = []
    for comdat_name, comdat in module.comdats.items():
        lines.append(f"${comdat_name} = comdat {get_comdat_kind(comdat)}")

    for name, struct_ty in module.structs.items():
        field_ty_list = ", ".join(
            [get_type_name(ty) for ty in struct_ty.fields])

        if struct_ty.is_packed:
            lines.append(f"%{name} = type <{{ {field_ty_list} }}>")
        else:
            lines.append(f"%{name} = type {{ {field_ty_list} }}")

    for name, global_var in module.globals.items():
        init = "" if global_var.initializer is None else get_value_name(
            global_var.initializer)

        decorations = ""

        thread_local = get_thread_local(global_var)
        if thread_local != "":
            decorations += thread_local + " "

        linkage = get_linkage(global_var)
        if global_var.linkage == GlobalLinkage.External:
            init = ""

        obj = "global"
        if global_var.is_constant:
            obj = "constant"

        lines.append(
            f"@{name} = {linkage} {decorations} {obj} {get_type_name(global_var.ty.elem_ty)} {init}, align 4")

    lines.extend([print_function(func) for func in module.funcs.values()])

    return "\n".join(lines)
