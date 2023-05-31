from rachetwrench.ir.values import *
from rachetwrench.ir.types import *
import ctypes


class ExecutionContext:
    def __init__(self, func):
        self.func = func
        self.bb_ptr = 0
        self.inst_ptr = 0
        self.values = {}
        self.alloc_mems = []

    def next_inst(self):
        if self.bb_ptr >= len(self.func.bbs):
            return False

        current_bb = self.func.bbs[self.bb_ptr]

        if self.inst_ptr >= len(current_bb.insts) - 1:
            self.inst_ptr = 0
            self.bb_ptr += 1
            return self.next_inst()

        self.inst_ptr += 1

        return True

    @property
    def inst(self):
        if self.bb_ptr >= len(self.func.bbs):
            return None

        current_bb = self.func.bbs[self.bb_ptr]

        if self.inst_ptr >= len(current_bb.insts):
            self.inst_ptr = 0
            self.bb_ptr += 1
            return self.inst

        return current_bb.insts[self.inst_ptr]


class GenericValue:
    def __init__(self, value):
        self.value = value


class Pointer:
    def __init__(self, memory, address):
        self.memory = memory
        self.address = address


class Interpreter:
    def __init__(self, module):
        self.module = module
        self.exec_ctx_stack = []
        self.memory = []

    def run_function(self, func: Function, *params):
        self.exec_ctx_stack.append(ExecutionContext(func))

        exec_ctx = self.exec_ctx_stack[-1]

        assert(len(params) >= len(func.args))

        for arg, param in zip(func.args, params):
            self.set_value(arg, param, exec_ctx)

        self.run()

        return self.exit_value

    def create_ty_value(self, ty, value):
        if isinstance(ty, StructType):
            assert(isinstance(value, list))

            field_vals = []
            for field_ty, field_val in zip(ty.fields, value):
                field_vals.append(self.create_ty_value(field_ty, field_val))

            return (ty, field_vals)

        if isinstance(ty, PrimitiveType):
            if ty in [i1, i8, i16, i32, i64]:
                return int((ty, value & ((1 << 64) - 1)))
            elif ty in [f16, f32, f64, f128]:
                return float(value)

        raise NotImplementedError()

    def create_struct(self, name, value):
        ty = self.module.structs[name]
        return self.create_ty_value(ty, value)

    def create_i32(self, value):
        return self.create_ty_value(i32, value)

    def create_f64(self, value):
        return self.create_ty_value(f64, value)

    def set_value(self, ir_value, value, ctx):
        ctx.values[ir_value] = value

    def get_pointer(self, value):
        address = len(self.memory)
        self.memory.append(value)
        return Pointer(self.memory, address)

    def get_constant_value(self, value: Constant):
        if isinstance(value, ConstantInt):
            return value.value
        raise NotImplementedError()

    def get_constant_expr_value(self, value: Constant):
        raise NotImplementedError()

    def get_global_pointer(self, value):
        raise NotImplementedError()

    def get_operand_value(self, value: Value, ctx: ExecutionContext):
        if isinstance(value, ConstantExpr):
            return self.get_constant_expr_value(value)
        elif isinstance(value, Constant):
            return self.get_constant_value(value)
        elif isinstance(value, GlobalValue):
            return self.get_global_pointer(value)
        else:
            return ctx.values[value]

    def visit_return(self, inst: ReturnInst):
        frame = self.exec_ctx_stack[-1]

        if len(inst.operands) > 0:
            return_value = self.get_operand_value(inst.rs, frame)
        else:
            return_value = None

        self.exec_ctx_stack.pop()

        if not self.exec_ctx_stack:
            self.exit_value = return_value
        else:
            raise NotImplementedError()

    def visit_alloca(self, inst: AllocaInst):
        frame = self.exec_ctx_stack[-1]
        data_layout = self.module.data_layout

        num_elem = self.get_operand_value(inst.count, frame)

        ty = inst.ty.elem_ty
        type_size = data_layout.get_type_alloc_size(ty)

        mem = ctypes.create_string_buffer(type_size)
        ptr = ctypes.pointer(mem)
        addr = ctypes.addressof(ptr.contents)
        self.set_value(inst, addr, frame)

        frame.alloc_mems.append(mem)

    def load_int_value(self, ptr, size):
        src = (ctypes.c_char*size).from_address(ptr)
        bys = bytes(src[:size])
        return int.from_bytes(bys, "little")

    def load_bytes(self, ptr, size):
        src = (ctypes.c_char*size).from_address(ptr)
        return bytes(src[:size])

    def load_value(self, ptr, ty: Type):
        load_bytes = self.module.data_layout.get_type_alloc_size(ty)

        if isinstance(ty, PrimitiveType):
            if ty.name.startswith("i"):
                return self.load_int_value(ptr, load_bytes)

        elif isinstance(ty, StructType):
            return self.load_bytes(ptr, load_bytes)

        raise NotImplementedError()

    def visit_load(self, inst: LoadInst):
        frame = self.exec_ctx_stack[-1]

        src = self.get_operand_value(inst.rs, frame)
        val = self.load_value(src, inst.ty)
        self.set_value(inst, val, frame)

    def store_int_value(self, val: int, ptr, size):
        dest = (ctypes.c_char*size).from_address(ptr)
        dest[:size] = val.to_bytes(size, "little")

    def store_bytes(self, val: bytes, ptr, size):
        dest = (ctypes.c_char*size).from_address(ptr)
        dest[:size] = val

    def store_value(self, val, ptr, ty: Type):
        store_bytes = self.module.data_layout.get_type_alloc_size(ty)

        if isinstance(ty, PrimitiveType):
            if ty.name.startswith("i"):
                self.store_int_value(val, ptr, store_bytes)
                return

        elif isinstance(ty, StructType):
            self.store_bytes(val, ptr, store_bytes)
            return
        raise NotImplementedError()

    def visit_store(self, inst):
        frame = self.exec_ctx_stack[-1]

        dest = self.get_operand_value(inst.rd, frame)
        val = self.get_operand_value(inst.rs, frame)
        self.store_value(val, dest, inst.rs.ty)

    def visit_get_element_ptr(self, inst: GetElementPtrInst):
        frame = self.exec_ctx_stack[-1]

        ptr = self.get_operand_value(inst.rs, frame)

        indexed_ty = inst.pointee_ty
        offset = 0
        for idx in inst.idx:
            if isinstance(indexed_ty, PointerType):
                offset += self.module.data_layout.get_type_alloc_size(
                    indexed_ty.elem_ty) * idx.value
                indexed_ty = indexed_ty.elem_ty
            elif isinstance(indexed_ty, StructType):
                offset += self.module.data_layout.get_elem_offset(
                    indexed_ty, idx.value)
                indexed_ty = indexed_ty.get_elem_type(idx.value)

        new_ptr = ptr + offset

        self.set_value(inst, new_ptr, frame)
        return new_ptr

    def visit_binary(self, inst: BinaryInst):
        frame = self.exec_ctx_stack[-1]

        src1 = self.get_operand_value(inst.rs, frame)
        src2 = self.get_operand_value(inst.rt, frame)

        bits, _ = self.module.data_layout.get_type_size_in_bits(inst.ty)

        if inst.op == "add":
            result = src1 + src2
        elif inst.op == "sub":
            result = (src1 - src2) & ((0x1 << bits) - 1)
        elif inst.op == "and":
            result = src1 & src2
        elif inst.op == "or":
            result = src1 | src2
        elif inst.op == "xor":
            result = src1 ^ src2
        elif inst.op == "lshr":
            result = (src1 >> src2) & ((0x1 << bits) - 1)
        elif inst.op == "shl":
            result = (src1 << src2) & ((0x1 << bits) - 1)
        else:
            raise ValueError("Invalid operand")

        self.set_value(inst, result, frame)

    def switch_bb(self, bb):
        frame = self.exec_ctx_stack[-1]
        frame.bb_ptr = frame.func.bbs.index(bb)
        frame.inst_ptr = 0

    def visit_jump(self, inst: JumpInst):
        frame = self.exec_ctx_stack[-1]

        dest = inst.goto_target
        self.switch_bb(dest)

    def visit_branch(self, inst: BranchInst):
        frame = self.exec_ctx_stack[-1]

        cond = self.get_operand_value(inst.cond, frame)

        if cond > 0:
            dest = inst.then_target
        else:
            dest = inst.else_target

        self.switch_bb(dest)

    def visit_cmp(self, inst: CmpInst):
        frame = self.exec_ctx_stack[-1]

        src1 = self.get_operand_value(inst.rs, frame)
        src2 = self.get_operand_value(inst.rt, frame)

        bits, _ = self.module.data_layout.get_type_size_in_bits(inst.ty)

        def unordered(value):
            return value & ((0x1 << bits) - 1)

        if inst.op == "eq":
            result = 1 if src1 == src2 else 0
        elif inst.op == "ne":
            result = 1 if src1 != src2 else 0
        elif inst.op == "ugt":
            result = 1 if unordered(src1) > unordered(src2) else 0
        elif inst.op == "ult":
            result = 1 if unordered(src1) < unordered(src2) else 0
        elif inst.op == "slt":
            result = 1 if src1 < src2 else 0
        else:
            raise ValueError("Invalid operand")

        self.set_value(inst, result, frame)

    def visit_bitcast(self, inst: CmpInst):
        frame = self.exec_ctx_stack[-1]

        ptr = self.get_operand_value(inst.rs, frame)

        self.set_value(inst, ptr, frame)

    def visit(self, inst):
        if isinstance(inst, AllocaInst):
            self.visit_alloca(inst)
        elif isinstance(inst, LoadInst):
            self.visit_load(inst)
        elif isinstance(inst, StoreInst):
            self.visit_store(inst)
        elif isinstance(inst, GetElementPtrInst):
            self.visit_get_element_ptr(inst)
        elif isinstance(inst, BitCastInst):
            self.visit_bitcast(inst)
        elif isinstance(inst, CmpInst):
            self.visit_cmp(inst)
        elif isinstance(inst, BinaryInst):
            self.visit_binary(inst)
        elif isinstance(inst, JumpInst):
            self.visit_jump(inst)
        elif isinstance(inst, BranchInst):
            self.visit_branch(inst)
        elif isinstance(inst, ReturnInst):
            self.visit_return(inst)
        else:
            raise NotImplementedError()

    def run(self):
        stack = self.exec_ctx_stack
        while stack:
            exec_ctx = stack[-1]

            inst = exec_ctx.inst
            exec_ctx.next_inst()

            self.visit(inst)
