#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, auto
from ir.types import *


class Endianess(Enum):
    Big = auto()
    Little = auto()


def upper_bound(lst, value):
    for i, x in enumerate(lst):
        if x > value:
            return i

    return -1


def align_to(value, align):
    return int(int((value + align - 1) / align) * align)


class DataLayout:

    DEFAULT_ALIGN = [
        # (type, bitwidth, abi, pref)
        ("i1", 1, 8, 8),
        ("i8", 8, 8, 8),
        ("i16", 16, 16, 16),
        ("i32", 32, 32, 32),
        ("i64", 64, 32, 64),
        ("i128", 128, 128, 128),
        ("f16", 16, 16, 16),
        ("f32", 32, 32, 32),
        ("f64", 64, 64, 64),
        ("f128", 128, 128, 128),
        ("v64", 64, 64, 64),
        ("v96", 96, 128, 128),
        ("v128", 128, 128, 128),
        ("iptr", 64, 64, 64),
        ("a", 0, 0, 64),
    ]

    def __init__(self):
        self.align = {}
        self.pointer_align = {}
        self.stack_align = 0
        self.endian = Endianess.Little

        for ty, bitwidth, abi, pref in DataLayout.DEFAULT_ALIGN:
            self.align[ty] = (bitwidth, abi, pref)

        self.set_pointer_align(0, 64, 64, 64)

    def set_pointer_align(self, addr_space, abi, pref, size, idx=None):
        if idx == None:
            idx = size

        self.pointer_align[addr_space] = (abi, pref, size, idx)

    def get_pointer_size_in_bits(self, addr_space=0):
        abi, pref, size, idx = self.pointer_align[addr_space]
        return size

    def get_pref_type_alignment(self, ty):
        size, align = self.get_type_size_in_bits(ty)
        return align

    def get_type_size_in_bits(self, ty):
        if isinstance(ty, PrimitiveType):
            if ty.name in self.align:
                align = self.align[ty.name]
                return align[0], align[2]
        elif isinstance(ty, PointerType):
            align = self.pointer_align[0]
            return align[0], align[2]
        elif isinstance(ty, VectorType):
            if ty.elem_ty.name in self.align:
                align = self.align[ty.elem_ty.name]
                size = align[0] * ty.size

                align = self.align[f"v{size}"]

                return align[0], align[2]
        elif isinstance(ty, ArrayType):
            elem_size, elem_align = self.get_type_size_in_bits(ty.elem_ty)

            size = ty.size * align_to(elem_size, elem_align)
            align = elem_align

            return size, align
        elif isinstance(ty, StructType):
            size = 0
            max_align = 0
            for i, field_ty in enumerate(ty.fields):
                elem_size, elem_align = self.get_type_size_in_bits(field_ty)

                size = int(
                    int((size + elem_align - 1) / elem_align) * elem_align)
                size += elem_size
                max_align = max(max_align, elem_align)

            size = int(int((size + max_align - 1) / max_align) * max_align)

            return size, max_align
        elif isinstance(ty, VoidType):
            return 0, 0

        raise NotImplementedError

    def get_type_alloc_size(self, ty):
        size, align = self.get_type_size_in_bits(ty)
        align = max(1, align)

        return int(int((size + align - 1) / align) * align / 8)

    def get_elem_offset_in_bits(self, ty, idx):
        if isinstance(ty, PointerType):
            elem_size, elem_align = self.get_type_size_in_bits(ty.elem_ty)
            return (idx * elem_size, elem_size)

        if isinstance(ty, VectorType):
            elem_size, elem_align = self.get_type_size_in_bits(ty.elem_ty)
            return (idx * elem_size, elem_size)

        if isinstance(ty, ArrayType):
            elem_size, elem_align = self.get_type_size_in_bits(ty.elem_ty)
            return (idx * elem_size, elem_size)

        if isinstance(ty, StructType):
            if idx > len(ty.fields):
                raise IndexError("idx")

            offset = 0
            for i, field_ty in enumerate(ty.fields):
                field_size, field_align = self.get_type_size_in_bits(field_ty)

                offset = align_to(offset, field_align)

                if i == idx:
                    return (offset, field_size)

                offset += field_size

        raise NotImplementedError(
            f"Type {ty.__class__.__name__} is not supporting.")

    def get_elem_offset(self, ty, idx):
        offset, size = self.get_elem_offset_in_bits(ty, idx)
        assert(offset % 8 == 0)
        return int(offset / 8)

    def get_elem_containing_offset(self, ty, offset):
        raise NotImplementedError()
        idx = upper_bound()
