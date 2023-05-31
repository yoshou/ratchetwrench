#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from rachetwrench.glsl.keywords import keywords_list

whitespace = re.compile(r"[\r\n \t\v\f]",
                        re.DOTALL | re.MULTILINE)

identifier = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")

keywords = re.compile("^(" + "|".join(keywords_list) + ")$")

integer_constant = re.compile(r'[-+]?([1-9]\d*|0)')
hexadecimal_constant = re.compile(r'0[xX][0-9a-fA-F]+')

floating_constant = re.compile(
    r'([-+]?[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?)((?:f|F|lf|LF)?)')

comment = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE
)


def escape_str(s):
    result = []

    for ch in s:
        if ch in ["(", ")", "[", "]", ".", "+", "*", "/", "^", "|"]:
            result.append("\\" + ch)
        else:
            result.append(ch)

    return "".join(result)


def excape_strs(strs):
    return [re.escape(s) for s in strs]


operators_list = excape_strs([
    "(", ")",
    "[", "]",
    ".",
    "++", "--",
    "+", "-", "~", "!",
    "*", "/", "%",
    "<<", ">>",
    "<=", ">=", "<", ">",
    "==", "!=",
    "&", "^", "|",
    "&&", "^^", "||",
    "?", ":",
    "=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|=",
    ","
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
        return str(self.value)

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


class FloatingConstant:
    def __init__(self, value, suffix, span):
        self.value = value
        self.suffix = suffix
        self.span = span

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


def is_whitespace(ch):
    return ch in ['\n', '\r', ' ', '\t', '\v', '\f']


class Span:
    def __init__(self, src, start, end):
        self.src = src
        self.start = start
        self.end = end


def peek_token(src, pos, cnt):
    string = src[pos:]

    result = re.match(comment, string)
    if result:
        return (len(result.group()), None)

    result = re.match(whitespace, string)
    if result:
        return (len(result.group()), None)

    # TODO: Bad implementation.
    result = re.match(identifier, string)
    if result:
        result2 = re.match(keywords, result.group())
        if result2:
            span = Span(src, pos + result2.span()[0], pos + result2.span()[1])
            return (len(result2.group()), Keyword(result2.group(), span))

        span = Span(src, pos + result.span()[0], pos + result.span()[1])
        return (len(result.group()), Identifier(result.group(), span))

    result = floating_constant.match(src, pos)
    if result:
        value, suffix = result.groups()
        span = Span(src, pos + result.span()[0], pos + result.span()[1])
        return (len(result.group()), FloatingConstant(value, suffix, span))

    result = re.match(hexadecimal_constant, string)
    if result:
        span = Span(src, pos + result.span()[0], pos + result.span()[1])
        return (len(result.group()), IntegerConstant(result.group(), span))

    result = re.match(integer_constant, string)
    if result:
        span = Span(src, pos + result.span()[0], pos + result.span()[1])
        return (len(result.group()), IntegerConstant(result.group(), span))

    result = re.match(operators, string)
    if result:
        span = Span(src, pos + result.span()[0], pos + result.span()[1])
        return (len(result.group()), Keyword(result.group(), span))

    result = re.match(r"[;{}]", string)
    if result:
        span = Span(src, pos + result.span()[0], pos + result.span()[1])
        return (len(result.group()), Keyword(result.group(), span))

    print("error")
    print(string)
    sys.exit()


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


if __name__ == "__main__":

    result = re.match(integer_constant, "1.0")
    print(result)
