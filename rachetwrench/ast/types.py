#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from enum import Enum, Flag, auto


class Type:
    def __init__(self):
        pass


class VoidType(Type):
    def __init__(self):
        pass

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, VoidType)

    @property
    def name(self):
        return "void"


class PointerType(Type):
    def __init__(self, elem_ty):
        self._elem_ty = elem_ty

    def __hash__(self):
        return hash(tuple([self.elem_ty]))

    def __eq__(self, other):
        if not isinstance(other, PointerType):
            return False

        return self._elem_ty == other._elem_ty

    @property
    def elem_ty(self):
        if isinstance(self._elem_ty, QualType):
            return self._elem_ty.ty

        return self._elem_ty

    @property
    def name(self):
        return f"{self.elem_ty.name}*"


class PrimitiveType(Type):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __hash__(self):
        return hash(tuple([self.name]))

    def __eq__(self, other):
        if not isinstance(other, PrimitiveType):
            return False

        return self.name == other.name


class VectorType(Type):
    def __init__(self, elem_ty, size):
        assert(isinstance(size, int))
        self.elem_ty = elem_ty
        self.size = size

    @property
    def name(self):
        if self.elem_ty.name == "bool":
            name = "bvec"
        elif self.elem_ty.name == "int":
            name = "ivec"
        elif self.elem_ty.name == "uint":
            name = "uvec"
        elif self.elem_ty.name == "float":
            name = "vec"
        elif self.elem_ty.name == "double":
            name = "dvec"
        else:
            raise NotImplementedError()

        return name + str(self.size)

    def __hash__(self):
        return hash(self.elem_ty)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.elem_ty == other.elem_ty

    def __ne__(self, other):
        return not self.__eq__(other)


class ArrayType(Type):
    def __init__(self, elem_ty, size):
        assert(isinstance(size, int))
        self.elem_ty = elem_ty
        self.size = size

    def __hash__(self):
        return hash(self.elem_ty)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.elem_ty == other.elem_ty

    def __ne__(self, other):
        return not self.__eq__(other)


class CompositeType(Type):
    def __init__(self, name, fields, is_union=False):
        super().__init__()
        self.name = name
        self._fields = None
        self._fields_index = []
        self.fields = fields
        self.is_union = is_union

    @property
    def fields(self):
        return self._fields

    @fields.setter
    def fields(self, value):
        self._fields = value

        if value:
            self._fields_index = {name: i for i,
                                  (ty, name, arr) in enumerate(value)}

    def get_field_by_idx(self, idx):
        return self.fields[idx]

    def get_field_by_name(self, name):
        idx = 0
        bit_pos = 0
        for ty, field_name, bit in self.fields:
            if bit is None:
                if name == field_name:
                    return self.fields[idx]

                bit_pos = 0
                idx += 1
            else:
                raise NotImplementedError()

                bit_pos += bit

                if bit_pos >= 32:
                    idx += 1
                    bit_pos -= 32

        idx = self._fields_index[name]
        return self.fields[idx]

    def get_field_type_by_name(self, name):
        idx = self._fields_index[name]
        ty, _, bit = self.fields[idx]
        assert(not bit)
        return ty

    def get_field_idx(self, name):
        idx = self._fields_index[name]
        _, _, bit = self.fields[idx]
        assert(not bit)
        return idx

    def contains_field(self, name):
        return name in self._fields_index

    def __hash__(self):
        return hash(tuple([self.name, frozenset(self.fields)]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name and self.fields == other.fields

    def __ne__(self, other):
        return not self.__eq__(other)


class EnumType(Type):
    def __init__(self, values=None):
        self._values = None
        self.values = values

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, vals):
        self._values = vals
        if vals:
            for val_name, value in vals.items():
                assert(isinstance(val_name, str))

    def __eq__(self, other):
        if not isinstance(other, EnumType):
            return False

        return self._values == other._values


class FunctionType(Type):
    def __init__(self, return_ty, params, is_variadic=False):
        super().__init__()
        self.return_ty = return_ty
        self.params = params
        self.is_variadic = is_variadic

    def __eq__(self, other):
        if not isinstance(other, FunctionType):
            return False

        return self.return_ty == other.return_ty and [param for param, _, _ in self.params] == [param for param, _, _ in other.params]


class Qualifier(Flag):
    Undefined = 0
    Const = auto()
    Restrict = auto()
    Volatile = auto()


class QualType:
    def __init__(self, ty, quals):
        self.ty = ty
        self.quals = quals

    @property
    def elem_ty(self):
        return QualType(self.ty.elem_ty, self.quals)


buildin_type_reg_size = {
    'int': 1,
    'uint': 1,
    'float': 1,
}


def compute_type_size(ty):
    if isinstance(ty, PrimitiveType):
        return buildin_type_reg_size[ty.name]
    elif isinstance(ty, CompositeType):
        size = 0
        for field_name, field_type in ty.fields.items():
            field_ty, arr = field_type
            if arr is not None:
                raise NotImplementedError
            size += compute_type_size(field_ty)
        return size

    raise NotImplementedError


def compute_type_field_offset(ty, field):
    if isinstance(ty, CompositeType):
        size = 0
        for field_name, field_type in ty.fields.items():
            field_ty, arr = field_type
            if arr is not None:
                raise NotImplementedError
            field_size = compute_type_size(field_ty)

            if field_name == field:
                return (size, field_size)

            size += field_size

    raise NotImplementedError
