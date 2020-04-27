#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

from ast.types import *


class TypeSymbol:
    def __init__(self, name, ty, scope):
        self.name = name
        self.ty = ty
        self.scope = scope

    def __eq__(self, other):
        if not isinstance(other, TypeSymbol):
            return False
        return self.name == other.name and self.ty == other.ty and self.scope == other.scope

    def __hash__(self):
        return hash(tuple([self.name, self.ty, self.scope]))

    def __str__(self):
        return self.name


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

    def register_func(self, name, ty, params, is_variadic):
        for param_ty, _, _ in params:
            assert(isinstance(param_ty, Type))

        symbol = FunctionSymbol(name, ty, params, is_variadic, self)
        self.funcs[name] = symbol
        return symbol

    def register_var(self, name, ty, ty_quals):
        if not ty:
            raise ValueError()

        symbol = Symbol(name, ty, ty_quals, self)
        self.vars[name] = symbol
        return symbol

    def register_composite_type(self, name, fields, is_union):
        ty = CompositeType(name, fields)
        ty.is_union = is_union
        self.types[name] = ty
        return TypeSymbol(name, ty, self)

    def register_primitive_type(self, name):
        ty = PrimitiveType(name)
        self.types[name] = ty
        return TypeSymbol(name, ty, self)

    def register_function_type(self, name, ty, params, is_variadic):
        for param in params:
            assert(isinstance(param, Type))

        ty = FunctionType(ty, params, is_variadic)
        self.types[name] = ty
        return TypeSymbol(name, ty, self)

    def register_alias_type(self, name, ty, ty_quals):
        self.types[name] = ty
        return TypeSymbol(name, ty, self)


class FunctionSymbol:
    def __init__(self, name, ty, params, is_variadic, scope):
        self.name = name
        self.ty = FunctionType(ty, params, is_variadic)


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
        bool_ty = self.pointer.register_primitive_type('bool').ty
        int_ty = self.pointer.register_primitive_type('int').ty
        uint_ty = self.pointer.register_primitive_type('unsigned int').ty
        float_ty = self.pointer.register_primitive_type('float').ty
        double_ty = self.pointer.register_primitive_type('double').ty
        short_ty = self.pointer.register_primitive_type('short').ty
        short_ty = self.pointer.register_primitive_type('char').ty
        self.pointer.types['void'] = VoidType()

        self.pointer.register_primitive_type('_Bool')
        self.pointer.register_primitive_type('_Complex')
        self.pointer.register_primitive_type('unsigned short')
        self.pointer.register_primitive_type('long')
        self.pointer.register_primitive_type('unsigned long')
        self.pointer.register_primitive_type('long double')
        self.pointer.register_primitive_type('unsigned char')

        self.pointer.register_primitive_type('__int64')
        self.pointer.register_primitive_type('unsigned __int64')
        self.pointer.register_alias_type('unsigned', uint_ty, [])

    @property
    def top_scope(self):
        return self.pointer

    def register_func(self, name, ty, params, is_variadic):
        return self.pointer.register_func(name, ty, params, is_variadic)

    def register_var(self, name, ty, ty_qual):
        return self.pointer.register_var(name, ty, ty_qual)

    def register_composite_type(self, name, fields, is_union=False):
        assert(name)
        return self.pointer.register_composite_type(name, fields, is_union)

    def register_alias_type(self, name, ty, ty_quals):
        return self.pointer.register_alias_type(name, ty, ty_quals)

    def register_primitive_type(self, name):
        return self.pointer.register_primitive_type(name)

    def register_function_type(self, name, ty, params, is_variadic):
        return self.pointer.register_function_type(name, ty, params, is_variadic)

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
                return self.get_type_and_field(ty, field)

            scope = scope.parent

        return None

    def get_type_and_field(self, ty, field):
        if isinstance(ty, VectorType):
            return (ty.elem_ty, None)

        assert(isinstance(ty, CompositeType))

        if ty.contains_field(field):
            return ty.get_field_by_name(field)

        for field_ty, field_name, _ in ty.fields:
            if field_name.startswith("struct.anon"):
                result = self.get_type_and_field(field_ty, field)
                if result:
                    return result

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
