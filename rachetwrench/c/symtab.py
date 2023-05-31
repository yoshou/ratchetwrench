#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys

from rachetwrench.ast.types import *


class Symbol:
    def __init__(self, name, scope):
        self.name = name
        self.scope = scope

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self.name == other.name and self.scope == other.scope

    def __hash__(self):
        return hash(tuple([self.name, self.scope]))

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Symbol(name={self.name}, scope={id(self.scope)})"


class SymbolScope:
    def __init__(self, parent, table):
        self.children = []
        self.parent = parent
        self.table = table

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def register_object(self, name, obj):
        assert(isinstance(obj, (FunctionSymbol, VariableSymbol)))

        symbol = Symbol(name, self)
        if symbol in self.table.objects:
            return self.table.objects[symbol]

        self.table.objects[symbol] = obj
        return obj

    def register(self, name, value):
        symbol = Symbol(name, self)
        self.table.symbols[symbol] = value
        return symbol


class FunctionSymbol:
    def __init__(self, name, ty):
        self.name = name
        self.ty = ty


class VariableSymbol:
    def __init__(self, name, ty):
        self.name = name
        self.ty = ty

    # def __eq__(self, other):
    #     if other is None:
    #         return False
    #     if not isinstance(other, VariableSymbol):
    #         return False
    #     return self.name == other.name

    # def __hash__(self):
    #     return hash(tuple([self.name]))

    def __str__(self):
        return self.name


class SymbolTable:
    def __init__(self, types=[]):
        self.global_scope = SymbolScope(None, self)
        self.pointer = self.global_scope
        self.symbols = {}
        self.objects = {}

    @property
    def top_scope(self):
        return self.pointer

    def register_object(self, name, obj):
        return self.pointer.register_object(name, obj)

    def register(self, name, value):
        return self.pointer.register(name, value)

    def find_object(self, symbol):
        assert(isinstance(symbol, Symbol))
        while symbol.scope:
            if symbol in self.objects:
                return self.objects[symbol]
            symbol = Symbol(symbol.name, symbol.scope.parent)

        return None

    def find_type(self, symbol):
        assert(isinstance(symbol, Symbol))
        while symbol.scope:
            if symbol in self.symbols:
                return self.symbols[symbol]
            symbol = Symbol(symbol.name, symbol.scope.parent)

        return None

    def find_type_and_field(self, name, field):
        ty = self.find_type(name)
        return self.get_type_and_field(ty, field)

    def get_type_and_field(self, ty, field):
        if isinstance(ty, VectorType):
            return (ty.elem_ty, None)

        assert(isinstance(ty, CompositeType))

        if ty.contains_field(field):
            return ty.get_field_by_name(field)

        for field_ty, field_name, _ in ty.fields:
            if not field_name:
                result = self.get_type_and_field(field_ty, field)
                if result:
                    return result

        return None

    def push_scope(self):
        scope = SymbolScope(self.pointer, self)
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
