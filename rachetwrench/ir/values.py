#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import os
import html
from enum import Enum, auto
from collections import namedtuple
from rachetwrench.ir.types import Type, LabelType, VoidType, PointerType, PrimitiveType, CompositeType, FunctionType, VectorType, StructType, ArrayType
from rachetwrench.ir.data_layout import DataLayout


def get_mangled_type_name(ty: Type):
    if isinstance(ty, PointerType):
        return "p0" + get_mangled_type_name(ty.elem_ty)
    if isinstance(ty, (PrimitiveType, CompositeType)):
        return ty.name

    raise Exception("Unreachable")


def encode_intrinsic_func_name(name, tys):
    return ".".join([name] + [get_mangled_type_name(ty) for ty in tys])


def get_return_ty(name):
    if name == "llvm.memcpy":
        return VoidType()
    if name == "llvm.va_start":
        return VoidType()
    if name == "llvm.va_end":
        return VoidType()
    if name == "llvm.returnaddress":
        return PointerType(PrimitiveType("i8"), 0)

    raise Exception("Unreachable")


class AttributeKind(Enum):
    Non = auto()
    Alignment = "align"
    AllocSize = "allocsize"
    AlwaysInline = "alwaysinline"
    ArgMemOnly = "argmemonly"
    Builtin = "builtin"
    ByVal = "byval"
    Cold = "cold"
    Convergent = "convergent"
    Dereferenceable = "dereferenceable"
    DereferenceableOrNull = "dereferenceable_or_null"
    InAlloca = "inalloca"
    InReg = "inreg"
    InaccessibleMemOnly = "inaccessiblememonly"
    InaccessibleMemOrArgMemOnly = "inaccessiblemem_or_argmemonly"
    InlineHint = "inlinehint"
    JumpTable = "jumptable"
    MinSize = "minsize"
    Naked = "naked"
    Nest = "nest"
    NoAlias = "noalias"
    NoBuiltin = "nobuiltin"
    NoCapture = "nocapture"
    NoCfCheck = "nocf_check"
    NoDuplicate = "noduplicate"
    NoImplicitFloat = "noimplicitfloat"
    NoInline = "noinline"
    NoRecurse = "norecurse"
    NoRedZone = "noredzone"
    NoReturn = "noreturn"
    NoUnwind = "nounwind"
    NonLazyBind = "nonlazybind"
    NonNull = "nonnull"
    OptForFuzzing = "optforfuzzing"
    OptimizeForSize = "optsize"
    OptimizeNone = "optnone"
    ReadNone = "readnone"
    ReadOnly = "readonly"
    Returned = "returned"
    ReturnsTwice = "returns_twice"
    SExt = "signext"
    SafeStack = "safestack"
    SanitizeAddress = "sanitize_address"
    SanitizeHWAddress = "sanitize_hwaddress"
    SanitizeMemory = "sanitize_memory"
    SanitizeThread = "sanitize_thread"
    ShadowCallStack = "shadowcallstack"
    Speculatable = "speculatable"
    StackAlignment = "alignstack"
    StackProtect = "ssp"
    StackProtectReq = "sspreq"
    StackProtectStrong = "sspstrong"
    StrictFP = "strictfp"
    StructRet = "sret"
    SwiftError = "swifterror"
    SwiftSelf = "swiftself"
    UWTable = "uwtable"
    WriteOnly = "writeonly"
    ZExt = "zeroext"


class Attribute:
    def __init__(self, kind, value=None):
        self.kind = kind
        self.value = value

    def __hash__(self):
        return hash((self.kind, self.value))

    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False

        return self.kind == other.kind and self.value == other.value


class Module:
    def __init__(self):
        self._funcs = {}
        self._globals = {}
        self.structs = {}
        self.data_layout = DataLayout()
        self.comdats = {}

    def add_func(self, name, func):
        if name not in self._funcs or not func.is_declaration:
            self._funcs[name] = func
        return self._funcs[name]

    def add_global(self, name, variable):
        if name not in self._globals or variable.linkage != GlobalLinkage.External:
            self._globals[name] = variable
        return self._globals[name]

    @property
    def funcs(self):
        return dict(self._funcs)

    @property
    def globals(self):
        return dict(self._globals)

    def add_struct_type(self, name, ty):
        self.structs[name] = ty

    def add_comdat(self, name, kind):
        comdat = Comdat(name, kind)
        self.comdats[name] = comdat
        return comdat

    def contains_struct_type(self, name):
        return name in self.structs

    def get_or_declare_intrinsic_func(self, name, arg_tys):
        if name == "llvm.memcpy":
            enc_name = encode_intrinsic_func_name(name, arg_tys[:3])
        elif name in ["llvm.va_start", "llvm.va_end"]:
            enc_name = encode_intrinsic_func_name(name, [])
        elif name == "llvm.returnaddress":
            enc_name = encode_intrinsic_func_name(name, [])
        else:
            raise ValueError("The intrinsic function name is not supported.")

        if enc_name in self._funcs:
            return self._funcs[enc_name]
        else:
            func = Function(
                self, FunctionType(get_return_ty(name), arg_tys), GlobalLinkage.Global, enc_name)

            for arg_ty in arg_tys:
                func.add_arg(Argument(arg_ty))

            self._funcs[enc_name] = func

            return func

    def get_named_value(self, name):
        if name in self._funcs:
            return self._funcs[name]

        return KeyError()


class Value:
    def __init__(self, ty: Type, name=""):
        self.name = name
        self.ty = ty
        self._uses = []

    @property
    def uses(self):
        return tuple(self._uses)

    def add_use(self, user):
        self._uses.append(user)

    def remove_use(self, user):
        idx = self._uses.index(user)
        self._uses.pop(idx)

    @property
    def has_name(self):
        return self.name != ""

    @property
    def value_name(self):
        if not self.has_name:
            return ""

        if isinstance(self, GlobalValue):
            return f"@{self.name}"

        return f"%{self.name}"

    @property
    def value_type(self):
        return self.ty


class Constant(Value):
    def __init__(self, ty, name=""):
        super().__init__(ty, name)


class ConstantExpr(Constant):
    def __init__(self, ty, name=""):
        super().__init__(ty, name)


def get_constant_null_value(ty):
    if isinstance(ty, PrimitiveType):
        if ty.name.startswith("i"):
            return ConstantInt(0, ty)
        if ty.name.startswith("f"):
            return ConstantFP(0.0, ty)

    if isinstance(ty, VectorType):
        return ConstantVector([get_constant_null_value(ty.elem_ty)] * ty.size, ty)

    if isinstance(ty, ArrayType):
        return ConstantArray([get_constant_null_value(ty.elem_ty)] * ty.size, ty)

    if isinstance(ty, StructType):
        values = []
        for field in ty.fields:
            values.append(get_constant_null_value(field))

        return ConstantStruct(values, ty)

    if isinstance(ty, PointerType):
        return ConstantPointerNull(ty)

    raise ValueError("Invalid type.")


class GlobalLinkage(Enum):
    Local = auto()
    Global = auto()
    Weak = auto()
    External = auto()
    Internal = auto()
    Private = auto()


class GlobalVisibility(Enum):
    Default = auto()
    Internal = auto()
    Hidden = auto()
    Protected = auto()
    Exported = auto()


class ThreadLocalMode(Enum):
    NotThreadLocal = auto()
    GeneralDynamicTLSModel = auto()
    LocalDynamicTLSModel = auto()
    InitialExecTLSModel = auto()
    LocalExecTLSModel = auto()


class GlobalValue(Constant):
    def __init__(self, ty, linkage, name="", thread_local=ThreadLocalMode.NotThreadLocal, addr_space=0):
        super().__init__(PointerType(ty, addr_space), name)
        self.linkage = linkage
        self.visibility = GlobalVisibility.Default
        self.vty = ty
        self.thread_local = thread_local

    @property
    def is_thread_local(self):
        return self.thread_local != ThreadLocalMode.NotThreadLocal


class GlobalObject(GlobalValue):
    def __init__(self, ty, linkage, name, thread_local=ThreadLocalMode.NotThreadLocal):
        super().__init__(ty, linkage, name, thread_local)
        self.comdat = None

    @property
    def has_comdat(self):
        return self.comdat is not None


class GlobalVariable(GlobalObject):
    def __init__(self, ty, is_constant, linkage, name, thread_local=ThreadLocalMode.NotThreadLocal, initializer=None):
        assert(isinstance(linkage, GlobalLinkage))
        if initializer:
            assert(ty == initializer.ty)
        super().__init__(ty, linkage, name, thread_local)
        self.is_constant = is_constant
        self.initializer = initializer

    @property
    def is_declaration(self):
        return not self.initializer


class Argument(Value):
    def __init__(self, ty, name=""):
        assert(ty)
        super().__init__(ty, name)

        self.attrs = []

    def add_attribute(self, attr):
        self.attrs.append(attr)

    def has_attribute(self, attr_kind):
        for attr in self.attrs:
            if attr.kind == attr_kind:
                return True

        return False


class ConstantInt(Constant):
    def __init__(self, value: int, ty):
        super().__init__(ty)
        assert(isinstance(value, int))
        self.value = value

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if not isinstance(other, ConstantInt):
            return False

        return self.ty == other.ty and self.value == other.value

    def __hash__(self):
        return hash(tuple([self.ty, self.value]))


class ConstantFP(Constant):
    def __init__(self, value: float, ty):
        super().__init__(ty)
        assert(isinstance(value, float))
        self.value = value

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return "{:e}".format(self.value)

    def __eq__(self, other):
        if not isinstance(other, ConstantFP):
            return False

        return self.ty == other.ty and self.value == other.value

    def __hash__(self):
        return hash((self.ty, self.value))


class ConstantVector(Constant):
    def __init__(self, values: int, ty):
        super().__init__(ty)
        assert(len(values) == ty.size)
        assert(isinstance(ty, VectorType))
        self.values = values

    def __repr__(self):
        return str(self.values)

    def __str__(self):
        return str(self.values)

    def __eq__(self, other):
        if not isinstance(other, ConstantVector):
            return False

        return self.ty == other.ty and self.values == other.values

    def __hash__(self):
        return hash((self.ty, *self.values))


class ConstantStruct(Constant):
    def __init__(self, values: int, ty):
        super().__init__(ty)
        assert(isinstance(ty, StructType))
        self.values = values

    def __repr__(self):
        return str(self.values)

    def __str__(self):
        return str(self.values)

    def __eq__(self, other):
        if not isinstance(other, ConstantStruct):
            return False

        return self.ty == other.ty and self.values == other.values

    def __hash__(self):
        return hash((self.ty, *self.values))


class ConstantArray(Constant):
    def __init__(self, values, ty):
        super().__init__(ty)
        assert(isinstance(ty, ArrayType))
        if values:
            assert(ty.elem_ty == values[0].ty)
        self.values = values

    def __repr__(self):
        return str(self.values)

    def __str__(self):
        return str(self.values)

    def __eq__(self, other):
        if not isinstance(other, ConstantArray):
            return False

        return self.ty == other.ty and self.values == other.values

    def __hash__(self):
        return hash((self.ty, *self.values))


class ConstantPointerNull(Constant):
    def __init__(self, ty):
        super().__init__(ty)

    def __eq__(self, other):
        if not isinstance(other, ConstantPointerNull):
            return False

        return self.ty == other.ty

    def __hash__(self):
        return hash((self.ty,))


class Function(GlobalObject):
    def __init__(self, module, ty, linkage, name):
        assert(isinstance(ty, FunctionType))
        super().__init__(ty, linkage, name)
        self.module = module
        self.return_ty = ty.return_ty
        self.bbs = []
        self.args = []
        self._attributes = set()

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, value: set):
        self._attributes = value

    @property
    def func_ty(self):
        return self.ty.elem_ty

    @property
    def is_variadic(self):
        return self.ty.elem_ty.is_variadic

    @property
    def is_declaration(self):
        return len(self.bbs) == 0

    def add_block(self, block, block_before):
        if block_before:
            idx = self.bbs.index(block_before)
            self.bbs.insert(idx + 1, block)
        else:
            self.bbs.append(block)

    def remove_block(self, block):
        self.bbs.remove(block)

    def after_block(self, block):
        idx = self.bbs.index(block)
        if idx + 1 < len(self.bbs):
            return self.bbs[idx + 1]
        return None

    def add_arg(self, arg):
        self.args.append(arg)

    @property
    def blocks(self):
        return self.bbs


class BasicBlock(Value):
    def __init__(self, func, block_before=None):
        super().__init__(LabelType())
        self.func = func
        self.insts = []

        func.add_block(self, block_before)

    def move(self, block_before):
        if self == block_before:
            return

        self.func.blocks.remove(self)
        idx = self.func.blocks.index(block_before)
        self.func.blocks.insert(idx + 1, self)

    @property
    def next_block(self):
        return self.func.after_block(self)

    @property
    def terminator(self):
        if len(self.insts) == 0 or not self.insts[-1].is_terminator:
            return None

        return self.insts[-1]

    @property
    def phis(self):
        return [inst for inst in self.insts if isinstance(inst, PHINode)]

    def add_inst(self, inst, inst_before):
        assert(isinstance(inst, Instruction))

        if inst_before:
            idx = self.insts.index(inst_before)
            self.insts.insert(idx + 1, inst)
        else:
            self.insts.append(inst)

    def remove(self):
        idx = self.func.blocks.index(self)
        self.func.blocks.pop(idx)
        self.func = None

    @property
    def predecessors(self):
        for bb in self.func.blocks:
            succs = list(bb.successors)
            if self in bb.successors:
                yield bb

    @property
    def successors(self):
        for inst in reversed(self.insts):
            if not inst.is_terminator:
                break

            for succ in inst.successors:
                yield succ


class User(Value):
    def __init__(self, ty, ops, num_ops, name=""):
        super().__init__(ty, name)

        self._operands = [None] * num_ops
        for idx, op in enumerate(ops):
            self.set_operand(idx, op)

    def set_operand(self, idx: int, value: Value):
        if self._operands[idx] is not None:
            self._operands[idx].remove_use(self)

        if value is not None:
            value.add_use(self)

        self._operands[idx] = value

    def get_operand(self, idx: int):
        return self._operands[idx]

    @property
    def operands(self):
        return tuple(self._operands)


class Instruction(User):
    def __init__(self, block_or_inst: BasicBlock, ty, ops, num_ops, name=""):
        super().__init__(ty, ops, num_ops, name)

        if not block_or_inst:
            return

        if isinstance(block_or_inst, BasicBlock):
            assert(
                not block_or_inst.insts or not block_or_inst.insts[-1].is_terminator)
            self.block = block_or_inst
            self.block.add_inst(self, None)
        else:
            assert(isinstance(block_or_inst, Instruction))
            self.block = block_or_inst.block
            self.block.add_inst(self, block_or_inst)

    @property
    def successors(self):
        return []

    def remove(self):
        if len(self.uses) > 0:
            raise ValueError("This instruction is used by others.")

        if not self.block:
            return

        for i, operand in enumerate(self.operands):
            self.set_operand(i, None)

        self.block.insts.remove(self)
        self.block = None

    def move_after(self, block_or_inst):
        self.block.insts.remove(self)

        if isinstance(block_or_inst, BasicBlock):
            block_or_inst.insts.append(self)
            self.block = block_or_inst
        else:
            assert(isinstance(block_or_inst, Instruction))
            insert_index = block_or_inst.insts.index(block_or_inst)

            self.block.insts.insert(insert_index, self)
            self.block = block_or_inst.block

    @property
    def is_terminator(self):
        raise NotImplementedError()


class UnaryInst(Instruction):
    def __init__(self, block, ty, rs: Value, name=""):
        super().__init__(block, ty, [rs], 1)

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def is_terminator(self):
        return False


class BinaryInst(Instruction):
    def __init__(self, block, op, rs, rt):
        assert(rs.ty == rt.ty)
        super().__init__(block, rs.ty, [rs, rt], 2)
        self.op = op

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def rt(self):
        return self.get_operand(1)

    @property
    def is_terminator(self):
        return False


class CmpInst(Instruction):
    def __init__(self, block, op, rs, rt):
        assert(rs.ty == rt.ty)
        super().__init__(block, PrimitiveType("i1"), [rs, rt], 2)
        self.op = op

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def rt(self):
        return self.get_operand(1)

    @property
    def is_terminator(self):
        return False


class FCmpInst(Instruction):
    def __init__(self, block, op, rs, rt):
        assert(rs.ty == rt.ty)
        assert(rs.ty.name.startswith("f"))
        assert(rt.ty.name.startswith("f"))
        super().__init__(block, PrimitiveType("i1"), [rs, rt], 2)
        self.op = op

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def rt(self):
        return self.get_operand(1)

    @property
    def is_terminator(self):
        return False


class LoadInst(UnaryInst):
    def __init__(self, block: BasicBlock, rs: Value, is_volatile=False):
        assert(isinstance(rs.ty, PointerType))
        super().__init__(block, rs.ty.elem_ty, rs)

        self.is_volatile = is_volatile

    @property
    def is_terminator(self):
        return False


class StoreInst(Instruction):
    def __init__(self, block, rs: Value, rd: Value, is_volatile=False):
        assert(isinstance(rd.ty, PointerType))
        assert(rd.ty.elem_ty == rs.ty)
        super().__init__(block, VoidType(), [rs, rd], 2)

        self.is_volatile = is_volatile

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def rd(self):
        return self.get_operand(1)

    @property
    def is_terminator(self):
        return False


def get_indexed_type(ty, idx_list):
    if len(idx_list) == 0:
        return ty

    idx = idx_list[0]

    if isinstance(ty, PointerType):
        field_ty = ty.elem_ty
    elif isinstance(ty, VectorType):
        field_ty = ty.elem_ty
    elif isinstance(ty, ArrayType):
        field_ty = ty.elem_ty
    elif isinstance(ty, StructType):
        field_ty = ty.fields[idx.value]
    else:
        raise ValueError("Can't to access the field of the type.")

    if len(idx_list) == 1:
        return field_ty
    else:
        return get_indexed_type(field_ty, idx_list[1:])


def check_inbounds(ty, idx_list):
    for idx in idx_list:
        if not isinstance(idx, Constant):
            return False

    return True


class GetElementPtrInst(Instruction):
    def __init__(self, block, ptr, pointee_ty, *idx):
        assert(isinstance(ptr.ty, PointerType))
        assert(pointee_ty == ptr.ty)
        elem_ty = get_indexed_type(pointee_ty, list(idx))

        super().__init__(block, PointerType(
            elem_ty, 0), [ptr, *idx], 1 + len(idx))
        self.pointee_ty = pointee_ty
        self.inbounds = check_inbounds(pointee_ty, list(idx))

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def idx(self):
        return self.operands[1:]

    @property
    def is_terminator(self):
        return False

    @property
    def has_all_zero_indices(self):
        for i in self.idx:
            if not isinstance(i, ConstantInt):
                return False
            if i.value != 0:
                return False

        return True


class AllocaInst(UnaryInst):
    def __init__(self, block, count, ty, addr_space, align=8, name=""):
        assert(not isinstance(ty, VoidType))
        super().__init__(block, PointerType(ty, addr_space), count, name)
        self.alloca_ty = ty
        self.align = align

    @property
    def count(self):
        return self.get_operand(0)

    def is_static_alloca(self):
        count = self.count
        if not isinstance(count, ConstantInt):
            return False

        return True

    @property
    def is_terminator(self):
        return False


class PHINode(Instruction):
    def __init__(self, block, ty, values):
        super().__init__(block, ty, values, len(values))
        assert(len(values) > 0)
        self.check_value_types()

    @property
    def values(self):
        return {k: v for k, v in zip(self.incoming_blocks, self.incoming_values)}

    def check_value_types(self):
        ty = self.ty

        for value in self.operands[0::2]:
            if ty != value.ty:
                raise ValueError("Values must be the same types.")
            ty = value.ty

    @property
    def incoming_values(self):
        return [value for value in self.operands[::2]]

    @property
    def incoming_blocks(self):
        return [value for value in self.operands[1::2]]

    @property
    def is_terminator(self):
        return False


class CallInst(Instruction):
    def __init__(self, block, func_ty, callee, args):
        assert(isinstance(func_ty, FunctionType))
        super().__init__(block, func_ty.return_ty,
                         [callee, *args], 1 + len(args))
        self.func_ty = func_ty

    @property
    def callee(self):
        return self.get_operand(0)

    @property
    def args(self):
        return self.operands[1:]

    @property
    def is_terminator(self):
        return False


class CastInst(UnaryInst):
    def __init__(self, block, op, rs, ty):
        super().__init__(block, ty, rs)
        self.op = op

    @property
    def is_terminator(self):
        return False


class TruncInst(CastInst):
    def __init__(self, block, rs, ty):
        assert(rs.ty.name != "i1")
        super().__init__(block, "trunc", rs, ty)

    @property
    def is_terminator(self):
        return False


class ZExtInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "zext", rs, ty)

    @property
    def is_terminator(self):
        return False


class SExtInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "sext", rs, ty)

    @property
    def is_terminator(self):
        return False


class FPTruncInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "fptrunc", rs, ty)

    @property
    def is_terminator(self):
        return False


class FPExtInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "fpext", rs, ty)

    @property
    def is_terminator(self):
        return False


class FPToUIInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "fptoui", rs, ty)

    @property
    def is_terminator(self):
        return False


class UIToFPInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "uitofp", rs, ty)

    @property
    def is_terminator(self):
        return False


class FPToSIInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "fptosi", rs, ty)

    @property
    def is_terminator(self):
        return False


class SIToFPInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "sitofp", rs, ty)

    @property
    def is_terminator(self):
        return False


class PtrToIntInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "ptrtoint", rs, ty)

    @property
    def is_terminator(self):
        return False


class IntToPtrInst(CastInst):
    def __init__(self, block, rs, ty):
        assert(isinstance(rs.ty, PrimitiveType))
        super().__init__(block, "inttoptr", rs, ty)

    @property
    def is_terminator(self):
        return False


class BitCastInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "bitcast", rs, ty)

    @property
    def is_terminator(self):
        return False


class AddrSpaceCastInst(CastInst):
    def __init__(self, block, rs, ty):
        super().__init__(block, "addrspacecast", rs, ty)

    @property
    def is_terminator(self):
        return False


class AtomicOrdering(Enum):
    NotAtomic = 0
    Unordered = 1
    Monotonic = 2
    Acquire = 3
    Release = 4
    AcquireRelease = 5
    SequentiallyConsistent = 6


class SyncScopeValue:
    def __init__(self, name, id):
        self.name = name
        self.id = id


class SyncScope(Enum):
    # Synchronized with respect to signal handlers executing in the same thread.
    SingleThread = SyncScopeValue("singlethread", 0)

    # Synchronized with respect to all concurrently executing threads.
    System = SyncScopeValue("system", 1)


class FenceInst(Instruction):
    def __init__(self, block, ordering: AtomicOrdering, syncscope: SyncScope):
        super().__init__(block, VoidType(), [], 0)

        self.ordering = ordering
        self.syncscope = syncscope


class AtomicCmpXchgInst(Instruction):
    def __init__(self, block, ptr, cmp, new_value, success_ordering, failure_ordering, syncscope: SyncScope):
        super().__init__(block, VoidType(), [ptr, cmp, new_value], 3)

        self.success_ordering = success_ordering
        self.failure_ordering = failure_ordering
        self.syncscope = syncscope


class AtomicRMW(Instruction):
    def __init__(self, block, ptr, value, ordering, syncscope: SyncScope):
        super().__init__(block, VoidType(), [ptr, value], 2)

        self.ordering = ordering
        self.syncscope = syncscope


class IntrinsicInst(CallInst):
    pass


class MemcpyInst(IntrinsicInst):
    pass


class VAArgInst(Instruction):
    def __init__(self, block, va_list, ty):
        super().__init__(block, ty, [va_list], 1)

    @property
    def va_list(self):
        return self.operands[0]

    @property
    def is_terminator(self):
        return False


# terminator


class SwitchInst(Instruction):
    def __init__(self, block, value, default, cases):
        super().__init__(block, VoidType(), [
            value, default, *cases], 2 + len(cases))

    @property
    def value(self):
        return self.get_operand(0)

    @property
    def default(self):
        return self.get_operand(1)

    @property
    def cases(self):
        return [(val, dest) for val, dest in zip(self.case_vals, self.case_dests)]

    @property
    def case_vals(self):
        return self.operands[2::2]

    @property
    def case_dests(self):
        return self.operands[3::2]

    @property
    def is_terminator(self):
        return True

    @property
    def successors(self):
        return [self.default, *self.case_dests]


class BranchInst(Instruction):
    def __init__(self, block, cond, then_target, else_target):
        assert(cond.ty.name == "i1")
        super().__init__(block, VoidType(), [
            cond, then_target, else_target], 3)

    @property
    def cond(self):
        return self.get_operand(0)

    @property
    def then_target(self):
        return self.get_operand(1)

    @property
    def else_target(self):
        return self.get_operand(2)

    @property
    def is_terminator(self):
        return True

    @property
    def successors(self):
        return [self.then_target, self.else_target]


class JumpInst(Instruction):
    def __init__(self, block, goto_target):
        super().__init__(block, VoidType(), [goto_target], 1)

    @property
    def goto_target(self):
        return self.get_operand(0)

    @property
    def is_terminator(self):
        return True

    @property
    def successors(self):
        return [self.goto_target]


class ReturnInst(Instruction):
    def __init__(self, block, rs):
        super().__init__(block, VoidType(), [] if rs is None else [rs], 0 if rs is None else 1)

    @property
    def rs(self):
        return self.get_operand(0)

    @property
    def is_terminator(self):
        return True


class InsertElementInst(Instruction):
    def __init__(self, block, vec, elem, idx):
        assert(isinstance(vec.ty, VectorType))
        super().__init__(block, vec.ty, [vec, elem, idx], 3)

    @property
    def vec(self):
        return self.get_operand(0)

    @property
    def elem(self):
        return self.get_operand(1)

    @property
    def idx(self):
        return self.get_operand(2)

    @property
    def is_terminator(self):
        return False


class ExtractElementInst(Instruction):
    def __init__(self, block, vec, idx):
        assert(isinstance(vec.ty, VectorType))
        super().__init__(block, vec.elem_ty, [vec, idx], 2)

    @property
    def vec(self):
        return self.get_operand(0)

    @property
    def idx(self):
        return self.get_operand(1)

    @property
    def is_terminator(self):
        return False


def get_indexed_type_fixed(ty, idx_list):
    if len(idx_list) == 0:
        return ty

    idx = idx_list[0]

    if isinstance(ty, PointerType):
        field_ty = ty.elem_ty
    elif isinstance(ty, VectorType):
        field_ty = ty.elem_ty
    elif isinstance(ty, ArrayType):
        field_ty = ty.elem_ty
    elif isinstance(ty, StructType):
        field_ty = ty.fields[idx]
    else:
        raise ValueError("Can't to access the field of the type.")

    if len(idx_list) == 1:
        return field_ty
    else:
        return get_indexed_type_fixed(field_ty, idx_list[1:])


class ExtractValueInst(Instruction):
    def __init__(self, block, value, *idx):
        assert(isinstance(value.ty, (ArrayType, StructType)))
        elem_ty = get_indexed_type_fixed(value.ty, list(idx))
        super().__init__(block, elem_ty, [value], 1)
        self.idx = list(idx)

    @property
    def value(self):
        return self.get_operand(0)

    @property
    def is_terminator(self):
        return False


class InsertValueInst(Instruction):
    def __init__(self, block, value, elem, *idx):
        assert(isinstance(value.ty, (ArrayType, StructType)))
        elem_ty = get_indexed_type_fixed(value.ty, list(idx))
        assert(elem_ty == elem.ty)
        super().__init__(block, value.ty, [value, elem], 2)
        self.idx = list(idx)

    @property
    def value(self):
        return self.get_operand(0)

    @property
    def elem(self):
        return self.get_operand(1)

    @property
    def is_terminator(self):
        return False


class ComdatKind(Enum):
    # The linker may choose any COMDAT.
    Any = auto()
    # The data referenced by the COMDAT must be the same.
    ExactMatch = auto()
    # The linker will choose the largest COMDAT.
    Largest = auto()
    # No other Module may specify this COMDAT.
    NoDuplicates = auto()
    # The data referenced by the COMDAT must be the same size.
    SameSize = auto()


class Comdat:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind


class ConstraintPrefix(Enum):
    Input = auto()
    Output = auto()
    Clobber = auto()


class ConstraintInfo:
    def __init__(self):
        self.ty = None
        self.codes = None

    def parse(self, s):
        i = 0

        self.ty = ConstraintPrefix.Input
        self.codes = []

        if s.startswith("~", i):
            self.ty = ConstraintPrefix.Clobber
            i += 1
        elif s.startswith("=", i):
            self.ty = ConstraintPrefix.Output
            i += 1

        while i < len(s):
            if s.startswith("{", i):
                close_brace = s.find("}", i + 1)
                if close_brace >= len(s):
                    break

                self.codes.append(s[i:close_brace + 1])
                i = close_brace + 1
            else:
                self.codes.append(s[i:i+1])
                i += 1


class InlineAsm(Value):
    def __init__(self, ty: FunctionType, asm_string: str, constraints: str, has_side_effect):
        super().__init__(PointerType(ty, 0))

        self.asm_string = asm_string
        self.constraints = constraints
        self.has_side_effect = has_side_effect
        self.vty = ty
        self.func_ty = ty

    def parse_constraints(self):
        infos = []
        for s in self.constraints.split(","):
            info = ConstraintInfo()
            info.parse(s)
            infos.append(info)

        return infos


# scheduling


class PropPredInst(Instruction):
    def __init__(self, block, cond, pred=None):
        super().__init__(block)
        self.pred = pred
        self.cond = cond


class InvPredInst(Instruction):
    def __init__(self, block, pred):
        super().__init__(block)
        self.pred = pred


class RestorePredInst(Instruction):
    def __init__(self, block, pred):
        super().__init__(block)
        self.pred = pred


class ReturnPredInst(Instruction):
    def __init__(self, block, expr, pred=None):
        super().__init__(block)
        self.expr = expr
        self.pred = pred


class MaskPredInst(Instruction):
    def __init__(self, block, pred):
        super().__init__(block)
        self.pred = pred


class PushPredInst(Instruction):
    def __init__(self, block, pred=None):
        super().__init__(block)
        self.pred = pred


class BranchPredInst(Instruction):
    def __init__(self, block, cond, then_target, else_target, pred):
        super().__init__(block)
        self.pred = pred
        self.cond = cond
        self.then_target = then_target
        self.else_target = else_target
