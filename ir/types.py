#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from enum import Enum


class Type:
    def __init__(self):
        pass


class VoidType(Type):
    def __init__(self):
        pass

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        return "void"


class LabelType(Type):
    def __init__(self):
        pass

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        return "label"


class PointerType(Type):
    def __init__(self, elem_ty, addr_space):
        assert(isinstance(elem_ty, Type))
        assert(isinstance(addr_space, int))
        self.elem_ty = elem_ty
        self.addr_space = addr_space

    def __hash__(self):
        return hash((self.elem_ty, self.addr_space))

    def __eq__(self, other):
        if not isinstance(other, PointerType):
            return False

        return self.elem_ty == other.elem_ty and self.addr_space == other.addr_space

    @property
    def name(self):
        if self.addr_space != 0:
            return f"{self.elem_ty.name} addrspace({self.addr_space})*"
        return f"{self.elem_ty.name}*"


class PrimitiveType(Type):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __hash__(self):
        return hash(tuple([self.name]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)


class IntegerType(PrimitiveType):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.name = f"i{width}"

    def __hash__(self):
        return hash(tuple([self.name]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)


class CompositeType(Type):
    def __init__(self):
        super().__init__()

    def get_elem_type(self, idx):
        raise NotImplementedError()


class StructType(CompositeType):
    def __init__(self, name=None, fields=None, is_packed=False):
        super().__init__()
        self.name = name if name else ""
        self.fields = fields
        self.is_packed = is_packed

    def get_elem_type(self, idx):
        return self.fields[idx]

    def __hash__(self):
        return hash(tuple([self.name]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name and self.fields == other.fields

    def __ne__(self, other):
        return not self.__eq__(other)


class SequentialType(CompositeType):
    def __init__(self, elem_ty, size):
        super().__init__()
        self.elem_ty = elem_ty
        self.size = size

    def get_elem_type(self, idx):
        return self.elem_ty

    def __hash__(self):
        return hash(tuple([self.elem_ty, self.size]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.elem_ty == other.elem_ty and self.size == other.size


class ArrayType(SequentialType):
    def __init__(self, elem_ty, size):
        assert(isinstance(size, int))
        super().__init__(elem_ty, size)

    @property
    def name(self):
        return f"[{self.size} x {self.elem_ty.name}]"


class VectorType(SequentialType):
    def __init__(self, name, elem_ty, size):
        assert(isinstance(size, int))
        super().__init__(elem_ty, size)

    @property
    def name(self):
        return f"[{self.size} x {self.elem_ty.name}]"


class FunctionType(Type):
    def __init__(self, return_ty, params, is_variadic=False):
        super().__init__()
        self.return_ty = return_ty
        self.params = params
        self.is_variadic = is_variadic

        param_ty_list = ", ".join(
            [f"{param_ty.name}" for param_ty in params if param_ty])
        return_ty_name = return_ty.name

        if is_variadic:
            param_ty_list += ", ..."

        self.name = f"{return_ty_name} ({param_ty_list})"

    def __hash__(self):
        return hash(tuple([self.return_ty, tuple(self.params)]))

    def __eq__(self, other):
        if not isinstance(other, FunctionType):
            return False

        return self.return_ty == other.return_ty and self.params == other.params


# Instances
void = VoidType()

i1 = PrimitiveType("i1")
i8 = PrimitiveType("i8")
i16 = PrimitiveType("i16")
i32 = PrimitiveType("i32")
i64 = PrimitiveType("i64")

f16 = PrimitiveType("f16")
f32 = PrimitiveType("f32")
f64 = PrimitiveType("f64")
f128 = PrimitiveType("f128")


def get_integer_type(width):
    assert(isinstance(width, int))

    if width == 1:
        return i8
    elif width == 2:
        return i16
    elif width == 4:
        return i32
    elif width == 8:
        return i64

    return PrimitiveType(f"i{width}")


def get_array_type(elem_ty, size):
    return ArrayType(elem_ty, size)


def get_primitive_size(ty):
    if isinstance(ty, PrimitiveType):
        if ty.name == "i1":
            return 1
        elif ty.name == "i8":
            return 8
        elif ty.name == "i16":
            return 16
        elif ty.name == "i32":
            return 32
        elif ty.name == "i64":
            return 64
        elif ty.name == "f32":
            return 32
        elif ty.name == "f64":
            return 64
    elif isinstance(ty, VectorType):
        return get_primitive_size(ty.elem_ty) * ty.size
    else:
        return 0


def is_integer_ty(ty):
    if isinstance(ty, PrimitiveType):
        return ty.name in ["i1", "i8", "i16", "i32", "i64"]

    return False
