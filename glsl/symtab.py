#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

from ast.types import *


class SymbolScope:
    def __init__(self, parent):
        self.types = {}
        self.funcs = {}
        self.vars = {}
        self.children = []
        self.parent = parent

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def register_func(self, name, ty, params):
        symbol = FunctionSymbol(name, ty, params, self)
        self.funcs[name] = symbol
        return symbol

    def register_var(self, name, ty, ty_quals):
        symbol = Symbol(name, ty, ty_quals, self)
        self.vars[name] = symbol
        return symbol

    def register_composite_type(self, name, fields):
        symbol = CompositeType(name, fields, self)
        self.types[name] = symbol
        return symbol

    def register_primitive_type(self, name):
        symbol = PrimitiveType(name, self)
        self.types[name] = symbol
        return symbol


class FunctionSymbol:
    def __init__(self, name, ty, params, scope):
        self.name = name
        self.ty = FunctionType(name, ty, params, scope)


class Symbol:
    def __init__(self, name, ty, ty_qual, scope):
        self.name = name
        self.ty = ty
        self.ty_qual = ty_qual
        self.scope = scope

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, Symbol):
            return False
        return self.name == other.name and self.scope == other.scope

    def __hash__(self):
        return hash(tuple([self.name, self.scope]))

    def __str__(self):
        return self.name


class SymbolTable:
    def __init__(self, types=[]):
        self.types = types
        self.global_scope = SymbolScope(None)
        self.pointer = self.global_scope
        self.register_buildin_types()

    def register_buildin_types(self):
        bool_ty = self.pointer.register_primitive_type('bool')
        int_ty = self.pointer.register_primitive_type('int')
        uint_ty = self.pointer.register_primitive_type('uint')
        float_ty = self.pointer.register_primitive_type('float')
        double_ty = self.pointer.register_primitive_type('double')
        self.pointer.types['void'] = VoidType()

        for i in range(4):
            size = i + 1
            name = f'bvec{size}'
            self.pointer.types[name] = VectorType(bool_ty, 4)

        for i in range(4):
            size = i + 1
            name = f'ivec{size}'
            self.pointer.types[name] = VectorType(int_ty, 4)

        for i in range(4):
            size = i + 1
            name = f'uvec{size}'
            self.pointer.types[name] = VectorType(uint_ty, 4)

        for i in range(4):
            size = i + 1
            name = f'vec{size}'
            self.pointer.types[name] = VectorType(float_ty, 4)

        for i in range(4):
            size = i + 1
            name = f'dvec{size}'
            self.pointer.types[name] = VectorType(double_ty, 4)

    @property
    def top_scope(self):
        return self.pointer

    def register_func(self, name, ty, params):
        return self.pointer.register_func(name, ty, params)

    def register_var(self, name, ty, ty_qual):
        return self.pointer.register_var(name, ty, ty_qual)

    def register_composite_type(self, name, fields):
        return self.pointer.register_composite_type(name, fields)

    def register_primitive_type(self, name):
        return self.pointer.register_primitive_type(name)

    def find_func(self, name):
        scope = self.pointer
        while scope is not None:
            if name in scope.funcs:
                return scope.funcs[name]

            scope = scope.parent

        return None

    def find_var(self, name):
        scope = self.pointer
        while scope is not None:
            if name in scope.vars:
                return scope.vars[name]

            scope = scope.parent

        return None

    def find_type(self, name, array_specifier=None):
        scope = self.pointer
        while scope is not None:
            if name in scope.types:
                if array_specifier is not None:
                    return ArrayType(scope.types[name], array_specifier.sizes)
                return scope.types[name]

            scope = scope.parent

        return None

    def find_type_and_field(self, name, field):
        scope = self.pointer
        while scope is not None:
            if name in scope.types:
                ty = scope.types[name]
                if isinstance(ty, VectorType):
                    return (ty.elem_ty, None)

                assert(isinstance(ty, CompositeType))

                if ty.contains_field(field):
                    return ty.get_field_by_name(field)

            scope = scope.parent

        return None

    def push_scope(self):
        scope = SymbolScope(self.pointer)
        self.pointer.children.append(scope)
        self.pointer = scope

    def pop_scope(self):
        assert(self.pointer != self.global_scope)
        scope = self.pointer
        self.pointer = self.pointer.parent
        return scope

    @property
    def is_global(self):
        return self.pointer.depth == 0

    @property
    def depth(self):
        return self.pointer.depth
