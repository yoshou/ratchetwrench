#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, auto
from ir.types import *
from ir.values import *
from codegen.types import *


class DagOp:
    def __init__(self, name, ns):
        self.name = name
        self.namespace = ns

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class VirtualDagOp(DagOp):
    def __init__(self, name):
        super().__init__(name, "default")


class VirtualDagOps(Enum):
    ENTRY = VirtualDagOp("EntryToken")
    TOKEN_FACTOR = VirtualDagOp("TokenFactor")
    UNDEF = VirtualDagOp("Undef")
    MERGE_VALUES = VirtualDagOp("merge_values")
    BUILD_VECTOR = VirtualDagOp("build_vector")
    INSERT_VECTOR_ELT = VirtualDagOp("insert_vector_elt")
    EXTRACT_VECTOR_ELT = VirtualDagOp("extract_vector_elt")
    SCALAR_TO_VECTOR = VirtualDagOp("scalar_to_vector")
    SHUFFLE_VECTOR = VirtualDagOp("shuffle_vector")

    CONSTANT = VirtualDagOp("Constant")
    CONSTANT_FP = VirtualDagOp("ConstantFP")
    CONSTANT_POOL = VirtualDagOp("ConstantPool")
    GLOBAL_ADDRESS = VirtualDagOp("GlobalAddress")
    GLOBAL_TLS_ADDRESS = VirtualDagOp("GlobalTLSAddress")
    FRAME_INDEX = VirtualDagOp("FrameIndex")
    EXTERNAL_SYMBOL = VirtualDagOp("ExternalSymbol")

    TARGET_CONSTANT = VirtualDagOp("TargetConstant")
    TARGET_CONSTANT_FP = VirtualDagOp("TargetConstantFP")
    TARGET_CONSTANT_POOL = VirtualDagOp("TargetConstantPool")
    TARGET_GLOBAL_ADDRESS = VirtualDagOp("TargetGlobalAddress")
    TARGET_GLOBAL_TLS_ADDRESS = VirtualDagOp("TargetGlobalTLSAddress")
    TARGET_FRAME_INDEX = VirtualDagOp("TargetFrameIndex")
    TARGET_EXTERNAL_SYMBOL = VirtualDagOp("TargetExternalSymbol")

    # Integer binary arithmetic operators.
    ADD = VirtualDagOp("add")
    SUB = VirtualDagOp("sub")
    MUL = VirtualDagOp("mul")
    SDIV = VirtualDagOp("sdiv")
    UDIV = VirtualDagOp("udiv")
    SREM = VirtualDagOp("srem")
    UREM = VirtualDagOp("urem")
    SDIVREM = VirtualDagOp("sdivrem")
    UDIVREM = VirtualDagOp("udivrem")

    ADDC = VirtualDagOp("addc")
    SUBC = VirtualDagOp("subc")

    # Floating point binary arithmetic operators.
    FADD = VirtualDagOp("fadd")
    FSUB = VirtualDagOp("fsub")
    FMUL = VirtualDagOp("fmul")
    FDIV = VirtualDagOp("fdiv")
    FREM = VirtualDagOp("frem")

    FMA = VirtualDagOp("fma")
    FMAD = VirtualDagOp("fmad")

    # Bitwise operations
    AND = VirtualDagOp("and")
    OR = VirtualDagOp("or")
    XOR = VirtualDagOp("xor")

    # Shift operations
    SHL = VirtualDagOp("shl")
    SRA = VirtualDagOp("sra")
    SRL = VirtualDagOp("srl")
    ROTL = VirtualDagOp("rotl")
    ROTR = VirtualDagOp("rotr")
    FSHL = VirtualDagOp("fshl")
    FSHR = VirtualDagOp("fshr")

    BITCAST = VirtualDagOp("bitcast")
    SIGN_EXTEND = VirtualDagOp("sign_extend")
    ZERO_EXTEND = VirtualDagOp("zero_extend")

    LOAD = VirtualDagOp("load")
    STORE = VirtualDagOp("store")

    ATOMIC_FENCE = VirtualDagOp("atomic_fence")

    SETCC = VirtualDagOp("setcc")
    BR = VirtualDagOp("br")
    BRCOND = VirtualDagOp("brcond")

    CONDCODE = VirtualDagOp("CondCode")
    REGISTER = VirtualDagOp("Register")
    COPY_FROM_REG = VirtualDagOp("CopyFromReg")
    COPY_TO_REG = VirtualDagOp("CopyToReg")

    BASIC_BLOCK = VirtualDagOp("BasicBlock")

    DYNAMIC_STACKALLOC = VirtualDagOp("DynamicStackAlloc")

    CALLSEQ_START = VirtualDagOp("callseq_start")
    CALLSEQ_END = VirtualDagOp("callseq_end")

    TARGET_REGISTER = VirtualDagOp("TargetRegister")


class DagNode:
    def __init__(self, opcode, value_types, operands):
        self._opc = opcode
        self._vts = value_types
        self._ops = []
        self._uses = set()

        for operand in operands:
            self._add_operand(operand)

        self._freeze()
        self._compute_hash()

    def _freeze(self):
        self._vts = tuple(self._vts)
        self._ops = tuple(self._ops)

    def _add_operand(self, operand):
        if isinstance(operand, DagValue):
            operand = DagEdge(self, operand)
            self._ops.append(operand)
        elif isinstance(operand, DagEdge):
            assert(operand.source == self)
            self._ops.append(operand)
        else:
            raise ValueError(
                "Argument operand must be of type DagEdge or DagValue.")

    @property
    def value_types(self):
        return self._vts

    @property
    def opcode(self):
        return self._opc

    @opcode.setter
    def opcode(self, value):
        raise RuntimeError()

    @property
    def operands(self):
        return [op.ref for op in self._ops]

    @property
    def ops(self):
        return self._ops

    def get_label(self):
        if isinstance(self.opcode, Enum):
            return str(self.opcode.value)

        return str(self.opcode)

    def __eq__(self, other):
        if not isinstance(other, DagNode):
            return False
        for opnd1, opnd2 in zip(self.operands, other.operands):
            if opnd1.node is not opnd2.node:
                return False
            if opnd1.index != opnd2.index:
                return False
        return self.opcode == other.opcode and self.value_types == other.value_types

    def _compute_hash(self):
        operands = tuple((id(opnd.node), opnd.index) for opnd in self.operands)
        self._hash_val = hash((self.opcode, tuple(self.value_types), operands))

    def __hash__(self):
        return self._hash_val

    @property
    def uses(self):
        return self._uses


class DagValue:
    def __init__(self, node: DagNode, index):
        assert(isinstance(node, DagNode))
        self._node = node
        self.index = index

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        assert(isinstance(node, DagNode))
        self._node = node

    @property
    def ty(self):
        return self.node.value_types[self.index]

    def get_value(self, idx):
        return DagValue(self._node, idx)

    @property
    def valid(self):
        return self.index < len(self._node.value_types)


class DagEdge:
    def __init__(self, source: DagNode, ref: DagValue):
        assert(isinstance(source, DagNode))
        assert(isinstance(ref, DagValue))
        self.source = source
        self.ref = ref


class MemDagNode(DagNode):
    def __init__(self, opcode, value_types, operands, mem_operand):
        super().__init__(opcode, value_types, operands)
        self.mem_operand = mem_operand

    def __eq__(self, other):
        if not isinstance(other, MemDagNode):
            return False
        return self.mem_operand == other.mem_operand and super().__eq__(other)

    def __hash__(self):
        return hash((super().__hash__(), self.mem_operand))


class MachinePointerInfo:
    def __init__(self, value):
        self.value = value


class MachineMemOperand:
    def __init__(self, ptr_info, size):
        self.ptr_info = ptr_info
        self.size = size


class LoadDagNode(MemDagNode):
    def __init__(self, value_types, operands, mem_operand):
        super().__init__(VirtualDagOps.LOAD, value_types, operands, mem_operand)

    def get_label(self):
        return f"load<(load {self.mem_operand.size})>"

    def __eq__(self, other):
        if not isinstance(other, LoadDagNode):
            return False
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class StoreDagNode(MemDagNode):
    def __init__(self, value_types, operands, mem_operand):
        super().__init__(VirtualDagOps.STORE, [
            MachineValueType(ValueType.OTHER)], operands, mem_operand)

    def get_label(self):
        return f"store<(store {self.mem_operand.size})>"

    def __eq__(self, other):
        if not isinstance(other, StoreDagNode):
            return False
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class RegisterDagNode(DagNode):
    def __init__(self, value_types, reg):
        super().__init__(VirtualDagOps.REGISTER, value_types, [])
        assert(isinstance(reg, (MachineRegister, MachineVirtualRegister)))
        self.reg = reg

    def get_label(self):
        prefix = '$' if isinstance(self.reg, MachineRegister) else ''
        return f"Register {prefix}{self.reg}"

    def __eq__(self, other):
        if not isinstance(other, RegisterDagNode):
            return False
        return self.reg == other.reg and self.value_types[0] == other.value_types[0]

    def __hash__(self):
        return hash((super().__hash__(), self.reg))


class MachineRegister:
    def __init__(self, spec):
        from codegen.spec import MachineRegisterDef

        assert(isinstance(spec, MachineRegisterDef))
        self.spec = spec

    def __eq__(self, other):
        if not isinstance(other, MachineRegister):
            return False
        return self.spec.name == other.spec.name

    def __hash__(self):
        return hash(self.spec.name)

    def __str__(self):
        return self.spec.name


class MachineVirtualRegister:
    def __init__(self, regclass, vid):
        self.regclass = regclass
        self.vid = vid

    def __str__(self):
        return f"%{self.vid}"


class BasicBlockDagNode(DagNode):
    def __init__(self, bb: BasicBlock):
        super().__init__(VirtualDagOps.BASIC_BLOCK, [
            MachineValueType(ValueType.OTHER)], [])
        self.bb = bb

    def get_label(self):
        return f"BasicBlock<{hex(id(self.bb))}>"

    def __eq__(self, other):
        if not isinstance(other, BasicBlockDagNode):
            return False
        return self.bb == other.bb and self.value_types[0] == other.value_types[0]

    def __hash__(self):
        return hash((super().__hash__(), self.bb))


class GlobalAddressDagNode(DagNode):
    def __init__(self, opcode, value_types, value: GlobalValue, target_flags):
        super().__init__(opcode, value_types, [])
        self.value = value
        self.target_flags = target_flags

    def get_label(self):
        return f"{self.opcode.value}<{hex(id(self.value))}>"

    def __eq__(self, other):
        if not isinstance(other, GlobalAddressDagNode):
            return False
        return self.value == other.value and self.value_types[0] == other.value_types[0] and self.target_flags == other.target_flags

    def __hash__(self):
        return hash((super().__hash__(), self.value, self.target_flags))


class ExternalSymbolDagNode(DagNode):
    def __init__(self, is_target, symbol: str, value_type, target_flags):
        assert(isinstance(value_type, MachineValueType))
        opcode = VirtualDagOps.TARGET_EXTERNAL_SYMBOL if is_target else VirtualDagOps.EXTERNAL_SYMBOL
        super().__init__(opcode, [value_type], [])
        self.symbol = symbol
        self.target_flags = target_flags

    def get_label(self):
        return f"{self.opcode.value}<{self.symbol}>"

    def __eq__(self, other):
        if not isinstance(other, ExternalSymbolDagNode):
            return False
        return self.symbol == other.symbol and self.value_types[0] == other.value_types[0]

    def __hash__(self):
        return hash((super().__hash__(), self.symbol))


class FrameIndexDagNode(DagNode):
    def __init__(self, is_target, index, ptr_ty):
        assert(isinstance(ptr_ty, MachineValueType))
        opcode = VirtualDagOps.TARGET_FRAME_INDEX if is_target else VirtualDagOps.FRAME_INDEX
        super().__init__(opcode, [ptr_ty], [])
        self.index = index

    def get_label(self):
        return f"{self.opcode.value}<{self.index}>"

    def __eq__(self, other):
        if not isinstance(other, FrameIndexDagNode):
            return False
        return self.opcode == other.opcode and self.value_types[0] == other.value_types[0] and self.index == other.index

    def __hash__(self):
        return hash((super().__hash__(), self.index))


class ConstantPoolDagNode(DagNode):
    def __init__(self, is_target: bool, value_types, constant, align: int = 0, target_flags=0):
        from codegen.mir import MachineConstantPoolValue
        assert(isinstance(constant, (Constant, MachineConstantPoolValue)))
        assert(len(value_types) == 1)
        opcode = VirtualDagOps.TARGET_CONSTANT_POOL if is_target else VirtualDagOps.CONSTANT_POOL
        super().__init__(opcode, value_types, [])
        self.constant = constant
        self.align = align
        self.target_flags = target_flags

    def get_label(self):
        return f"{self.opcode.value}<{self.constant}>"

    @property
    def value(self):
        return self.constant

    def __hash__(self):
        return hash((super().__hash__(), self.constant, self.target_flags))

    def __eq__(self, other):
        if not isinstance(other, ConstantPoolDagNode):
            return False
        return self.opcode == other.opcode and self.value_types[0] == other.value_types[0] and self.constant == other.constant and self.target_flags == other.target_flags


class ConstantDagNode(DagNode):
    def __init__(self, is_target: bool, value_types, constant: ConstantInt):
        assert(len(value_types) == 1)
        assert(isinstance(constant, ConstantInt))
        opcode = VirtualDagOps.TARGET_CONSTANT if is_target else VirtualDagOps.CONSTANT
        super().__init__(opcode, value_types, [])
        self._constant = constant

    def get_label(self):
        return f"{self.opcode.value}<{self._constant}>"

    @property
    def is_one(self):
        return self._constant.value == 1

    @property
    def is_zero(self):
        return self._constant.value == 0

    @property
    def value(self):
        return self._constant

    def __hash__(self):
        return hash((super().__hash__(), self._constant))

    def __eq__(self, other):
        if not isinstance(other, ConstantDagNode):
            return False
        return self.opcode == other.opcode and self.value_types[0] == other.value_types[0] and self._constant == other._constant


class ConstantFPDagNode(DagNode):
    def __init__(self, is_target: bool, value_types, constant: ConstantFP):
        assert(isinstance(constant, ConstantFP))
        assert(len(value_types) == 1)
        opcode = VirtualDagOps.TARGET_CONSTANT_FP if is_target else VirtualDagOps.CONSTANT_FP
        super().__init__(opcode, value_types, [])
        self._constant = constant

    def get_label(self):
        return f"{self.opcode.value}<{self._constant}>"

    @property
    def is_one(self):
        return self._constant.value == 1.0

    @property
    def is_zero(self):
        return self._constant.value == 0.0

    @property
    def value(self):
        return self._constant

    def __hash__(self):
        return hash((super().__hash__(), self._constant))

    def __eq__(self, other):
        if not isinstance(other, ConstantFPDagNode):
            return False
        return self.opcode == other.opcode and self.value_types[0] == other.value_types[0] and self._constant == other._constant


class CondCode(Enum):
    # Ordered (True if not nan)
    SETO = "seto"
    SETOEQ = "setoeq"
    SETOGT = "setogt"
    SETOGE = "setoge"
    SETOLT = "setolt"
    SETOLE = "setole"
    SETONE = "setone"

    # Unordered (True if nan) or Unsigned for integer
    SETUO = "setuo"
    SETUEQ = "setueq"
    SETUGT = "setugt"
    SETUGE = "setuge"
    SETULT = "setult"
    SETULE = "setule"
    SETUNE = "setune"

    # Signed
    SETEQ = "seteq"
    SETGT = "setgt"
    SETGE = "setge"
    SETLT = "setlt"
    SETLE = "setle"
    SETNE = "setne"


class ConditionCodeDagNode(DagNode):
    def __init__(self, value_types, cond: CondCode):
        super().__init__(VirtualDagOps.CONDCODE, value_types, [])
        self.cond = cond

    def get_label(self):
        return f"{self.cond.value}"

    def __hash__(self):
        return hash((super().__hash__(), self.cond))

    def __eq__(self, other):
        if not isinstance(other, ConditionCodeDagNode):
            return False
        return self.opcode == other.opcode and self.cond == other.cond


class ShuffleVectorDagNode(DagNode):
    def __init__(self, value_types, values, mask):
        assert(len(value_types) == 1)
        super().__init__(VirtualDagOps.SHUFFLE_VECTOR, value_types, values)
        self.mask = mask

    def __hash__(self):
        return hash((super().__hash__(), *self.mask))

    def __eq__(self, other):
        if not isinstance(other, ShuffleVectorDagNode):
            return False
        return super().__eq__(other) and self.mask == other.mask


class MachineDagNode(DagNode):
    def __init__(self, opcode, value_types, operands):
        super().__init__(opcode, value_types, operands)


class Dag:
    def __init__(self, mfunc):
        self.mfunc = mfunc
        self.nodes = {}
        entry_node = self.add_node(VirtualDagOps.ENTRY, [
                                   MachineValueType(ValueType.OTHER)])
        self._root = self.entry = DagValue(entry_node, 0)

    @property
    def data_layout(self):
        return self.mfunc.func_info.func.module.data_layout

    def add_node(self, opcode, value_types, *operands):
        node = DagNode(opcode, value_types, list(operands))
        node = self.append_node(node)

        return node

    def add_undef(self, ty):
        return self.add_node(VirtualDagOps.UNDEF, [ty])

    def add_merge_values(self, values):
        if len(values) == 1:
            return values[0]

        vts = []
        for value in values:
            assert(isinstance(value, DagValue))
            vts.append(value.ty)

        return DagValue(self.add_node(VirtualDagOps.MERGE_VALUES, vts, *values), 0)

    def add_shuffle_vector(self, value_ty: MachineValueType, vec1, vec2, mask):
        assert(isinstance(value_ty, MachineValueType))
        assert(isinstance(vec1, DagValue))
        assert(isinstance(vec2, DagValue))
        assert(isinstance(mask, list))
        node = ShuffleVectorDagNode([value_ty], [vec1, vec2], mask)
        node = self.append_node(node)
        return node

    def add_load_node(self, value_ty: MachineValueType, chain, ptr, is_volatile, ptr_info=None, mem_value_ty=None, mem_operand=None):
        assert(isinstance(value_ty, MachineValueType))

        if not mem_operand:
            if not mem_value_ty:
                mem_value_ty = value_ty

            mem_operand = MachineMemOperand(
                ptr_info, mem_value_ty.get_size_in_byte())

        offset = DagValue(self.add_undef(ptr.ty), 0)
        node = LoadDagNode(
            [value_ty, MachineValueType(ValueType.OTHER)], [chain, ptr, offset], mem_operand)
        node = self.append_node(node)

        if is_volatile:
            self.root = DagValue(node, 1)
        return node

    def add_store_node(self, chain: DagValue, ptr: DagValue, value: DagValue, is_volatile=True, ptr_info=None, mem_value_ty=None, mem_operand=None):
        assert(chain.ty == MachineValueType(ValueType.OTHER))
        assert(ptr.ty != MachineValueType(ValueType.OTHER))
        assert(value.ty != MachineValueType(ValueType.OTHER))

        value_ty = value.ty

        if not mem_value_ty:
            mem_value_ty = value_ty

        if not mem_operand:
            mem_operand = MachineMemOperand(
                ptr_info, mem_value_ty.get_size_in_byte())

        offset = DagValue(self.add_undef(ptr.ty), 0)
        node = StoreDagNode([MachineValueType(ValueType.OTHER)], [
                            chain, value, ptr, offset], mem_operand)
        node = self.append_node(node)

        if is_volatile:
            self.root = DagValue(node, 0)
        return node

    def add_frame_index_node(self, ptr_ty, index, is_target=False):
        assert(isinstance(ptr_ty, MachineValueType))
        node = FrameIndexDagNode(is_target, index, ptr_ty)
        node = self.append_node(node)
        return node

    def add_constant_pool_node(self, value_ty: MachineValueType, value, is_target=False, align=0, target_flags=0):
        assert(isinstance(value_ty, MachineValueType))
        node = ConstantPoolDagNode(
            is_target, [value_ty], value, align, target_flags)
        node = self.append_node(node)
        return node

    def add_constant_node(self, value_ty: MachineValueType, value, is_target=False):
        assert(isinstance(value_ty, MachineValueType))
        if isinstance(value, int):
            value = ConstantInt(value, value_ty.get_ir_type())
        assert(isinstance(value, ConstantInt))
        node = ConstantDagNode(is_target, [value_ty], value)
        node = self.append_node(node)
        return node

    def add_constant_fp_node(self, value_ty: MachineValueType, value, is_target=False):
        assert(isinstance(value_ty, MachineValueType))
        node = ConstantFPDagNode(is_target, [value_ty], value)
        node = self.append_node(node)
        return node

    def add_target_constant_node(self, value_ty: MachineValueType, value):
        return self.add_constant_node(value_ty, value, True)

    def add_target_constant_fp_node(self, value_ty: MachineValueType, value):
        return self.add_constant_fp_node(value_ty, value, True)

    def add_condition_code_node(self, cond: CondCode):
        node = ConditionCodeDagNode([MachineValueType(ValueType.OTHER)], cond)
        node = self.append_node(node)
        return node

    def add_basic_block_node(self, bb):
        node = BasicBlockDagNode(bb)
        node = self.append_node(node)
        return node

    def add_register_node(self, value_ty, reg):
        assert(isinstance(value_ty, MachineValueType))
        node = RegisterDagNode([value_ty], reg)
        node = self.append_node(node)
        return node

    def add_target_register_node(self, value_ty, reg):
        from codegen.spec import MachineRegisterDef

        assert(isinstance(value_ty, MachineValueType))
        assert(isinstance(reg, MachineRegisterDef))
        node = RegisterDagNode([value_ty], MachineRegister(reg))
        node = self.append_node(node)
        return node

    def add_global_address_node(self, value_ty, value, is_target, target_flags=0):
        assert(isinstance(value_ty, MachineValueType))

        if value.is_thread_local:
            opcode = VirtualDagOps.TARGET_GLOBAL_TLS_ADDRESS if is_target else VirtualDagOps.GLOBAL_TLS_ADDRESS
        else:
            opcode = VirtualDagOps.TARGET_GLOBAL_ADDRESS if is_target else VirtualDagOps.GLOBAL_ADDRESS

        node = GlobalAddressDagNode(opcode, [value_ty], value, target_flags)
        node = self.append_node(node)
        return node

    def add_external_symbol_node(self, value_ty, symbol: str, is_target=False, target_flags=0):
        assert(isinstance(value_ty, MachineValueType))

        node = ExternalSymbolDagNode(is_target, symbol, value_ty, target_flags)
        node = self.append_node(node)
        return node

    def add_copy_from_reg_node(self, ty, *operands):
        assert(len(operands) == 1)
        node = DagNode(VirtualDagOps.COPY_FROM_REG, [ty, MachineValueType(
            ValueType.OTHER)], [self.root] + list(operands))
        node = self.append_node(node)

        return node

    def add_copy_to_reg_node(self, *operands):
        node = DagNode(VirtualDagOps.COPY_TO_REG, [MachineValueType(
            ValueType.OTHER)], [self.root] + list(operands))
        node = self.append_node(node)

        self.root = DagValue(node, 0)
        return node

    def add_machine_dag_node(self, opcode, value_types, *operands):
        node = MachineDagNode(opcode, value_types, list(operands))
        return self.append_node(node)

    @property
    def control_root(self):
        return self._root

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, edge: DagValue):
        assert(isinstance(edge, DagValue))
        assert(edge.ty.value_type == ValueType.OTHER)
        self._root = edge

    def dfs_pre(self, func, *args):
        visited = set()

        def dfs_pre_rec(node: DagNode):
            if node is None:
                return

            assert(isinstance(node, DagNode))

            if node in visited:
                return

            visited.add(node)

            func(node, *args)

            for operand in node.operands:
                assert(isinstance(operand, DagValue))
                dfs_pre_rec(operand.node)

        dfs_pre_rec(self.root.node)

    def dfs_post(self, func, *args):
        visited = set()

        def dfs_post_rec(node: DagNode):
            if node is None:
                return

            if node in visited:
                return

            visited.add(node)

            for operand in node.operands:
                assert(isinstance(operand, DagNodeValue))
                dfs_post_rec(operand.node)

            func(node, *args)

        dfs_post_rec(self.root.node)

    def append_node(self, node: DagNode):
        assert(isinstance(node, DagNode))
        if node in self.nodes:
            node = self.nodes[node]
        else:
            self.nodes[node] = node

            for op in node.ops:
                op.ref.node.uses.add(op)

        return node

    def remove_node(self, node: DagNode):
        for op in set(node.ops):
            op.ref.node.uses.remove(op)

        self.nodes.pop(node)

    def remove_unreachable_nodes(self):
        reachable = set()

        def collect_node(node):
            reachable.add(node)

        self.dfs_pre(collect_node)

        remove_nodes = set()
        for node in self.nodes.keys():
            if node not in reachable:
                remove_nodes.add(node)

        for node in remove_nodes:
            self.remove_node(node)
