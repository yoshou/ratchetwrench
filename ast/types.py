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


class PointerType(Type):
    def __init__(self, elem_ty):
        self.elem_ty = elem_ty

    def __hash__(self):
        return hash(tuple([self.elem_ty]))


class PrimitiveType(Type):
    def __init__(self, name, scope):
        super().__init__()
        self.name = name
        self.scope = scope

    def __hash__(self):
        return hash(tuple([self.name, self.scope]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name and self.scope == other.scope

    def __ne__(self, other):
        return not self.__eq__(other)


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
    def __init__(self, name, fields, scope):
        super().__init__()
        self.name = name
        self.fields = fields
        self.scope = scope
        self.fields_index = {name: i for i,
                             (ty, name, arr) in enumerate(self.fields)}

    def get_field_by_idx(self, idx):
        return self.fields[idx]

    def get_field_by_name(self, name):
        idx = self.fields_index[name]
        return self.fields[idx]

    def get_field_idx(self, name):
        return self.fields_index[name]

    def contains_field(self, name):
        return name in self.fields_index

    def __hash__(self):
        return hash(tuple([self.name, frozenset(self.fields), self.scope]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.name == other.name and self.fields == other.fields and self.scope == other.scope

    def __ne__(self, other):
        return not self.__eq__(other)


class FunctionType(Type):
    def __init__(self, name, return_ty, params, scope):
        super().__init__()
        self.name = name
        self.return_ty = return_ty
        self.params = params
        self.scope = scope


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
