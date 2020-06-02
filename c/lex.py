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

character_constant = re.compile(r'(?:u|U|L)?\'((?:\\.|[^\'])*)\'')


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
    def __init__(self, value, suffix, span):
        self.value = value
        self.suffix = suffix
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
    result = src.match(identifier, pos)

    if result:
        if result.group() in keywords_list:
            span = Span(src, result.start(), result.end())
            return (len(result.group()), Keyword(result.group(), span))

        span = Span(src, result.start(), result.end())
        return (len(result.group()), Identifier(result.group(), span))

    return None


def match_integer_constant(src, pos):
    result = src.match(hexadecimal_constant, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = src.match(integer_suffix, match_end)
        if suffix:
            match_len += len(suffix.group())

        span = Span(src, match_beg, match_end)
        return (match_len, IntegerConstant(result.group(), suffix.group() if suffix else "", span))

    result = src.match(decimal_constant, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = src.match(integer_suffix, match_end)
        if suffix:
            match_len += len(suffix.group())

        span = Span(src, match_beg, match_end)
        return (match_len, IntegerConstant(result.group(), suffix.group() if suffix else "", span))

    result = src.match(octet_constant, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = src.match(integer_suffix, match_end)
        if suffix:
            match_len += len(suffix.group())

        span = Span(src, match_beg, match_end)
        return (match_len, IntegerConstant(result.group(), suffix.group() if suffix else "", span))

    return None


def match_floating_constant(src, pos):
    result = src.match(floating_constant, pos)
    if result:
        match_beg, match_end = result.span()
        match_len = match_end - match_beg

        suffix = src.match(floating_suffix, match_end)
        if suffix:
            suffix = suffix.group()
            match_len += len(suffix)
        else:
            suffix = ""

        span = Span(src, match_beg, match_end)
        return (match_len, FloatingConstant(result.group(), suffix, span))

    return None


def match_character_constant(src, pos):
    result = src.match(character_constant, pos)
    if result:
        span = Span(src, result.start(), result.end())
        return (len(result.group()), CharacterConstant(result.groups()[0], span))

    return None


def match_string_literal(src, pos):
    result = src.match(string_literal, pos)
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

    result = src.match(comment, pos)
    if result:
        return (len(result.group()), None)

    result = src.match(whitespace, pos)
    if result:
        return (len(result.group()), None)

    matched = match_floating_constant(src, pos)
    if matched:
        return matched

    matched = match_integer_constant(src, pos)
    if matched:
        return matched

    result = src.match(operators, pos)
    if result:
        span = Span(src, result.start(), result.end())
        return (len(result.group()), Keyword(result.group(), span))

    result = src.match(block_semicolon, pos)
    if result:
        span = Span(src, result.start(), result.end())
        return (len(result.group()), Keyword(result.group(), span))

    a = src[pos:]

    raise ValueError("Invalid string")


class Source:
    def match(self, pattern: re.Pattern, pos):
        raise NotImplementedError()


class SourceStream:
    def advance(self, n):
        raise NotImplementedError()

    @property
    def line_column(self):
        raise NotImplementedError()


def _compute_line_column(line_col, s):
    line, col = line_col
    for ch in s:
        if ch == "\n":
            line += 1
            col = 1
            continue

        col += 1

    return (line, col)


class RawSource(Source):
    def __init__(self, s: str):
        self.s = s

    def match(self, pattern: re.Pattern, pos):
        return pattern.match(self.s, pos)

    @property
    def source(self):
        return self.s

    def __len__(self):
        return len(self.source)

    def stream(self, pos):
        class RawSourceStream(SourceStream):
            def __init__(self, src, pos):
                self.src = src
                self.pos = pos
                self._line_col = _compute_line_column(
                    (1, 1), src.source[pos:])

            def advance(self, n):
                self._line_col = _compute_line_column(
                    self._line_col, self.src.source[self.pos:self.pos+n])
                self.pos += n

            @property
            def line_column(self):
                return self._line_col

        return RawSourceStream(self, 0)


class LexedSource(Source):
    def __init__(self, s, token_indices, tokens):
        self.s = s
        self.token_indices = token_indices
        self.tokens = tokens

    def match(self, pattern: re.Pattern, pos):
        return pattern.match(self.s, pos)

    def stream(self, pos):
        class LexedSourceStream(SourceStream):
            def __init__(self, src: LexedSource, pos):
                self.src = src
                self.pos = pos

            def advance(self, n):
                self.pos += n

            @property
            def line_column(self):
                token_idx, token_pos = self.src.token_indices[self.pos]
                line, col = self.src.tokens[token_idx].line_col
                return (line, col + token_pos)

        return LexedSourceStream(self, 0)

    @property
    def source(self):
        return self.s

    def __len__(self):
        return len(self.source)


def tokenize(src):
    if isinstance(src, str):
        src = RawSource(src)

    pos = 0
    stream = src.stream(0)
    cnt = len(src)

    tokens = []

    while pos < cnt:
        line_col = stream.line_column
        (size, token) = peek_token(src, pos, cnt)

        if token:
            tokens.append(token)

        pos += size

    return tokens
