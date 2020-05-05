from ir.types import *
from ir.data_layout import *
from enum import Enum, auto


class ValueType(Enum):
    OTHER = "ch"

    I1 = "i1"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    I128 = "i128"

    F16 = "f16"
    F32 = "f32"
    F64 = "f64"
    F80 = "f80"
    F128 = "f128"

    V1I1 = "v1i1"
    V2I1 = "v2i1"
    V4I1 = "v4i1"
    V8I1 = "v8i1"
    V16I1 = "v16i1"
    V32I1 = "v32i1"
    V64I1 = "v64i1"
    V128I1 = "v128i1"
    V256I1 = "v256i1"
    V512I1 = "v512i1"
    V1024I1 = "v1024i1"

    V1I32 = "v1i32"
    V2I32 = "v2i32"
    V4I32 = "v4i32"
    V8I32 = "v8i32"
    V16I32 = "v16i32"
    V32I32 = "v32i32"
    V64I32 = "v64i32"
    V128I32 = "v128i32"
    V256I32 = "v256i32"
    V512I32 = "v512i32"
    V1024I32 = "v1024i32"

    V1I64 = "v1I64"
    V2I64 = "v2I64"
    V4I64 = "v4I64"
    V8I64 = "v8I64"
    V16I64 = "v16I64"
    V32I64 = "v32I64"
    V64I64 = "v64I64"
    V128I64 = "v128I64"
    V256I64 = "v256I64"
    V512I64 = "v512I64"
    V1024I64 = "v1024I64"

    V1F32 = "v1f32"
    V2F32 = "v2f32"
    V4F32 = "v4f32"
    V8F32 = "v8f32"
    V16F32 = "v16f32"
    V32F32 = "v32f32"
    V64F32 = "v64f32"
    V128F32 = "v128f32"
    V256F32 = "v256f32"
    V512F32 = "v512f32"
    V1024F32 = "v1024f32"

    V1F64 = "v1f64"
    V2F64 = "v2f64"
    V4F64 = "v4f64"
    V8F64 = "v8f64"
    V16F64 = "v16f64"
    V32F64 = "v32f64"
    V64F64 = "v64f64"
    V128F64 = "v128f64"
    V256F64 = "v256f64"
    V512F64 = "v512f64"
    V1024F64 = "v1024f64"

    GLUE = "glue"

    IPTR = "iptr"


class MachineValueType:
    def __init__(self, value_type: ValueType):
        assert(isinstance(value_type, ValueType))
        self.value_type = value_type

    def __str__(self):
        return self.value_type.value

    def __hash__(self,):
        return hash(self.value_type)

    def __eq__(self, other):
        if not isinstance(other, MachineValueType):
            return False
        return self.value_type == other.value_type

    def get_size_in_byte(self):
        return int((self.get_size_in_bits() + 7) / 8)

    def get_size_in_bits(self):
        if self.value_type == ValueType.I1:
            return 1
        elif self.value_type == ValueType.I8:
            return 8
        elif self.value_type == ValueType.I16:
            return 16
        elif self.value_type == ValueType.I32:
            return 32
        elif self.value_type == ValueType.I64:
            return 64
        elif self.value_type == ValueType.F16:
            return 16
        elif self.value_type == ValueType.F32:
            return 32
        elif self.value_type == ValueType.F64:
            return 64
        elif self.value_type in [ValueType.V4F32]:
            return 128

        raise ValueError("Can't get the type size.")

    def get_num_vector_elems(self):
        if self.value_type == ValueType.V4F32:
            return 4

        raise ValueError("This is not a vector type.")

    def get_vector_elem_type(self):
        if self.value_type == ValueType.V4F32:
            return MachineValueType(ValueType.F32)

        raise ValueError("This is not a vector type.")

    def get_vector_elem_size_in_bits(self):
        return self.get_vector_elem_type().get_size_in_bits()

    def get_ir_type(self):
        if self.value_type == ValueType.I1:
            return i1
        if self.value_type == ValueType.I8:
            return i8
        if self.value_type == ValueType.I32:
            return i32
        if self.value_type == ValueType.I64:
            return i64
        if self.value_type == ValueType.F32:
            return f32
        if self.value_type == ValueType.F64:
            return f64
        if self.value_type == ValueType.V4F32:
            return VectorType("", f32, 4)

        raise ValueError("This type is not supported in ir.")

    @property
    def is_vector(self):
        return self.value_type.value.startswith("v")


OtherVT = MachineValueType(ValueType.OTHER)


def get_int_value_type(bitwidth):
    if bitwidth == 1:
        return MachineValueType(ValueType.I1)
    elif bitwidth == 8:
        return MachineValueType(ValueType.I8)
    elif bitwidth == 16:
        return MachineValueType(ValueType.I16)
    elif bitwidth == 32:
        return MachineValueType(ValueType.I32)
    elif bitwidth == 64:
        return MachineValueType(ValueType.I64)

    raise ValueError("Invalid bit width.")


# def get_machine_value_types(ty: Type, data_layout: DataLayout):
#     if isinstance(ty, PointerType):
#         vt = get_int_value_type(data_layout.get_pointer_size_in_bits(0))
#         return [vt]

#     PRIMITIVE_CVT_TABLE = {
#         i8: ValueType.I8,
#         i16: ValueType.I16,
#         i32: ValueType.I32,
#         i64: ValueType.I64,
#         f16: ValueType.F16,
#         f32: ValueType.F32,
#         f128: ValueType.F128,
#     }

#     if isinstance(ty, PrimitiveType):
#         vt = MachineValueType(PRIMITIVE_CVT_TABLE[ty])
#         return [vt]

#     print(ty)
#     raise NotImplementedError()
