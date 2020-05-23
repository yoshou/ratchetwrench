#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from c.keywords import keywords_list

whitespace = re.compile(r"[\r\n \t\v\f]",
                        re.DOTALL | re.MULTILINE)

identifier = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")

decimal_constant = re.compile(r'[1-9][0-9]*')
hexadecimal_constant = re.compile(r'0[xX][0-9a-fA-F]+')
octet_constant = re.compile(r'0[0-7]*')

floating_constant = re.compile(
    r'((?:(?:[-+]?[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?)|(?:[0-9]+[eE][-+]?[0-9]+)))')

floating_suffix = re.compile(r'([flFL])')

comment = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE
)

integer_suffix = re.compile(r'(?:([uU])(ll|LL|l|L)?)|(?:(ll|LL|l|L)([uU])?)')

string_literal = re.compile(r'(u8|u|U|L)?"([^"]*)"')

character_constant = re.compile(r'(?:u|U|L)?\'(\\.|[^\'])*\'')


def escape_str(s):
    return re.escape(s)


def excape_strs(strs):
    return [escape_str(s) for s in strs]


operators_list = excape_strs([
    "[", "]", "(", ")", "{", "}", ".", "->",
    "++", "--", "&", "*", "+", "-", "~", "!",
    "/", "%", "<<", ">>", "<", ">", "<=", ">=", "==", "!=", "^", "|", "&&", "||",
    "?", ":", ";", "...",
    "=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "^=", "|=",
    ",", "#", "##",
    "<:", ":>", "<%", "%>", "%:", "%:%:"
])

operators_list.sort(reverse=True)

operators = re.compile("(" + "|".join(operators_list) + ")")


class Keyword:
    def __init__(self, value, span):
        self.value = value
        self.span = span

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class Identifier:
    def __init__(self, value, span):
        self.value = value
        self.span = span

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self.value)


class IntegerConstant:
    def __init__(self, value, span):
        self.value = value
        self.span = span

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class CharacterConstant:
    def __init__(self, value, span):
        self.value = value
        self.span = span

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"\'{str(self.value)}\'"


class FloatingConstant:
    def __init__(self, value, suffix, span):
        self.value = value
        self.suffix = suffix
        self.span = span

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class StringLiteral:
    def __init__(self, value, span, prefix):
        self.value = value
        self.span = span
        self.prefix = prefix

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"\"{str(self.value)}\""


class Span:
    def __init__(self, src, start, end):
        self.src = src
        self.start = start
        self.end = end

    @property
    def value(self):
        return self.src[self.start:self.end]


def match_identifier(src, pos):
    result = identifier.match(src, pos)
    if result:
        if result.group() in keywords_list:
            span = Span(src, result.start(), result.end())
            return (len(result.group()), Keyword(result.group(), span))

        span = Span(src, result.start(), result.end())
        return (len(result.group()), Identifier(result.group(), span))

    return None


def match_integer_constant(src, pos):
    result = hexadecimal_constant.match(src, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = integer_suffix.match(src, match_end)
        if suffix:
            match_len += len(suffix.group())

        span = Span(src, match_beg, match_end)
        return (match_len, IntegerConstant(result.group(), span))

    result = decimal_constant.match(src, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = integer_suffix.match(src, match_end)
        if suffix:
            match_len += len(suffix.group())

        span = Span(src, match_beg, match_end)
        return (match_len, IntegerConstant(result.group(), span))

    result = octet_constant.match(src, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = integer_suffix.match(src, match_end)
        if suffix:
            match_len += len(suffix.group())

        span = Span(src, match_beg, match_end)
        return (match_len, IntegerConstant(result.group(), span))

    return None


def match_floating_constant(src, pos):
    result = floating_constant.match(src, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = floating_suffix.match(src, match_end)
        if suffix:
            suffix = suffix.group()
            match_len += len(suffix)
        else:
            suffix = ""

        span = Span(src, match_beg, match_end)
        return (match_len, FloatingConstant(result.group(), suffix, span))

    return None


def match_character_constant(src, pos):
    result = character_constant.match(src, pos)
    if result:
        span = Span(src, result.start(), result.end())
        return (len(result.group()), CharacterConstant(result.groups()[0], span))

    return None


def match_string_literal(src, pos):
    result = string_literal.match(src, pos)
    if result:
        prefix, value = result.groups()

        span = Span(src, result.start(), result.end())
        return (len(result.group()), StringLiteral(value, span, prefix))

    return None


block_semicolon = re.compile(r"[;{}]")


def peek_token(src, pos, cnt):
    matched = match_identifier(src, pos)
    if matched:
        return matched

    matched = match_string_literal(src, pos)
    if matched:
        return matched

    matched = match_character_constant(src, pos)
    if matched:
        return matched

    result = comment.match(src, pos)
    if result:
        return (len(result.group()), None)

    result = whitespace.match(src, pos)
    if result:
        return (len(result.group()), None)

    matched = match_floating_constant(src, pos)
    if matched:
        return matched

    matched = match_integer_constant(src, pos)
    if matched:
        return matched

    result = operators.match(src, pos)
    if result:
        span = Span(src, result.start(), result.end())
        return (len(result.group()), Keyword(result.group(), span))

    result = block_semicolon.match(src, pos)
    if result:
        span = Span(src, result.start(), result.end())
        return (len(result.group()), Keyword(result.group(), span))

    a = src[pos:]

    raise ValueError("Invalid string")


def tokenize(src):
    pos = 0
    cnt = len(src)

    tokens = []

    while pos < cnt:
        (size, token) = peek_token(src, pos, cnt)

        if token:
            tokens.append(token)

        pos += size

    return tokens
