import re
from enum import Enum, auto

from c.parse_expr_pp import *


class Span:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Token:
    def __init__(self, filename, source: str, span, ty, line_col):
        self.filename = filename
        self.source = source
        self.span = span
        self.ty = ty
        self.line_col = line_col

    @property
    def value(self):
        return self.source[self.span.start:self.span.end]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class TokenType(Enum):
    HeaderName = auto()
    Identifier = auto()
    PPNumber = auto()
    CharacterConstant = auto()
    StringLiteral = auto()
    Punctuator = auto()
    Other = auto()


header_name_pattern = re.compile(r'((?:<[^>]+>)|(?:"[^"]+"))')

whitespace = re.compile(r"[ \t\v\f]")
newline = re.compile(r'([\r\n]+)|$')
non_whitespace = re.compile(r'[^ \r\n\t\v\f]+')


comment = re.compile(
    r'//.*?$|/\*.*?\*/',
    re.DOTALL | re.MULTILINE
)

identifier = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")

string_literal = re.compile(r'"(?:\\.|[^"\\]*)*"')

character_constant = re.compile(r"(?:u|U|L)?'(\\.|[^'])*'")


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

pp_number = re.compile(r'[\.]?[0-9](([eEpP][\+\-])|[0-9_a-zA-Z]|[\.])*')


class EvalMask:
    def __init__(self, masked):
        self.evaluated = masked
        self.masked = masked


class IncludeDirective:
    def __init__(self, header_name):
        self.header_name = header_name


class DefineReplacementDirective:
    def __init__(self, identifier, replacements, params=None):
        self.identifier = identifier
        self.replacements = replacements
        self.params = params

    @property
    def is_func_style(self):
        return self.params is not None


class UndefDirective:
    def __init__(self, identifier):
        self.identifier = identifier


class IfDefDirective:
    def __init__(self, ident, inv):
        self.ident = ident
        self.inv = inv


class IfDirective:
    def __init__(self, expr_tokens):
        self.expr_tokens = expr_tokens


class ElseDirective:
    def __init__(self):
        pass


class ElIfDirective:
    def __init__(self, expr_tokens):
        self.expr_tokens = expr_tokens


class EndIfDirective:
    def __init__(self):
        pass


def evalute_constant_expr(expr):
    while isinstance(expr, list):
        if len(expr) != 1:
            raise ValueError()
        expr = expr[0]

    if isinstance(expr, BinaryOp):
        op = expr.op
        lhs = evalute_constant_expr(expr.lhs)
        if op == "||":
            if lhs:
                return 1

        rhs = evalute_constant_expr(expr.rhs)

        if op == "||":
            return lhs or rhs
        elif op == "&&":
            return lhs and rhs
        elif op == ">=":
            return lhs >= rhs
        elif op == ">":
            return lhs > rhs
        elif op == "<=":
            return lhs <= rhs
        elif op == "<":
            return lhs < rhs
        elif op == "==":
            return lhs == rhs
        elif op == "!=":
            return lhs != rhs
        elif op == "|":
            return lhs | rhs
        elif op == "+":
            return lhs + rhs
        elif op == "-":
            return lhs - rhs
        elif op == "*":
            return lhs * rhs
        elif op == "/":
            return lhs / rhs
        elif op == "%":
            return lhs % rhs
    elif isinstance(expr, ConditionalExpr):
        cond_expr = evalute_constant_expr(expr.cond_expr)
        true_expr = evalute_constant_expr(expr.true_expr)
        false_expr = evalute_constant_expr(expr.false_expr)
        return true_expr if cond_expr != 0 else false_expr
    elif isinstance(expr, UnaryOp):
        value = evalute_constant_expr(expr.expr)
        if expr.op == "!":
            return not value
    elif isinstance(expr, IntegerConstantExpr):
        return expr.val
    elif isinstance(expr, FloatingConstantExpr):
        return expr.val

    print(expr)
    raise ValueError()


class Preprocessor:
    def __init__(self, filename, source, include_dirs, system_include_dirs, eval_masks=None):
        self.filename = filename
        self.source = source.replace("\\\n", "")
        self.include_dirs = include_dirs
        self.system_include_dirs = system_include_dirs
        self.pos = 0
        self.line_column = (1, 1)
        self.context_stack = []
        self.groups = []
        self.groups.append(DefineReplacementDirective("_MSC_VER", ["1700"]))
        self.groups.append(DefineReplacementDirective(
            "_MSVC_LANG", ["201402"]))
        self.groups.append(DefineReplacementDirective(
            "_STL_LANG", ["201402"]))
        self.groups.append(DefineReplacementDirective(
            "_CRT_DECLARE_NONSTDC_NAMES", ["0"]))
        self.groups.append(DefineReplacementDirective(
            "__STDC_VERSION__", ["201112L"]))
        self.groups.append(DefineReplacementDirective(
            "__STDC__", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "_USE_DECLSPECS_FOR_SAL", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "_USE_ATTRIBUTES_FOR_SAL", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "_M_IX86_FP", ["0"]))
        self.groups.append(DefineReplacementDirective(
            "__STDC_WANT_SECURE_LIB__", ["0"]))
        self.groups.append(DefineReplacementDirective(
            "_M_X64", ["100"]))
        self.groups.append(DefineReplacementDirective(
            "__STDC_WANT_SECURE_LIB__", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "_NO_CRT_STDIO_INLINE", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "_WIN64", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "X64", ["1"]))
        self.groups.append(DefineReplacementDirective(
            "DEBUG", ["0"]))

        # For linux predefined macros.
        self.groups.append(DefineReplacementDirective(
            "_FILE_OFFSET_BITS", ["64"]))
        self.groups.append(DefineReplacementDirective(
            "_FORTIFY_SOURCE", ["0"]))
        self.groups.append(DefineReplacementDirective(
            "__OPTIMIZE__", ["0"]))

        if not eval_masks:
            self.eval_masks = [EvalMask(True)]
        else:
            self.eval_masks = eval_masks

    def save_pos(self):
        self.context_stack.append((self.pos, self.line_column))

    def restore_pos(self):
        (self.pos, self.line_column) = self.context_stack.pop()

    def advance(self, count):
        line, col = self.line_column
        for ch in self.source[self.pos:(self.pos+count)]:
            if ch == "\n":
                line += 1
                col = 1
                continue

            col += 1

        self.line_column = (line, col)

        self.pos += count

    def create_token(self, source, span, ty):
        return Token(self.filename, source, span, ty, self.line_column)

    def parse_identifier(self):
        m = identifier.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.Identifier)

        return None

    def parse_identifier_list(self):
        idents = []
        ident = self.parse_identifier()
        if ident:
            idents.append(ident)
        while True:
            pos = self.pos
            self.skip_whitespaces()
            ident = None
            if self.source[self.pos:].startswith(","):
                self.advance(1)
                self.skip_whitespaces()
                ident = self.parse_identifier()
                idents.append(ident)

            if not ident:
                self.pos = pos
                break

        return idents

    def parse_header_name(self):
        m = header_name_pattern.match(self.source, self.pos)

        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.HeaderName)

        return None

    def parse_token(self):
        m = identifier.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.Identifier)

        m = string_literal.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.StringLiteral)

        m = character_constant.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.CharacterConstant)

        m = pp_number.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.PPNumber)

        m = operators.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.Punctuator)

        m = non_whitespace.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            return self.create_token(self.source, Span(m.start(), m.end()), TokenType.Other)

        return None

    def skip_whitespaces(self):
        while True:
            pos = self.pos
            m = whitespace.match(self.source, self.pos)
            if m:
                self.advance(m.end() - m.start())
            m = comment.match(self.source, self.pos)
            if m:
                self.advance(m.end() - m.start())
            if pos == self.pos:
                break

    def process_if_group(self):
        self.save_pos()

        if self.source[self.pos].startswith("#"):
            self.advance(1)
            self.skip_whitespaces()

            src = self.source[self.pos:]
            if src.startswith("ifdef"):
                self.advance(len("ifdef"))
                self.skip_whitespaces()
                ident = self.parse_identifier()
                if ident:
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.advance(m.end() - m.start())
                        self.groups.append(IfDefDirective(ident, False))
                        group = self.process_group(True)
                        return group
                raise ValueError()
            elif src.startswith("ifndef"):
                self.advance(len("ifndef"))
                self.skip_whitespaces()
                ident = self.parse_identifier()
                if ident:
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.advance(m.end() - m.start())
                        self.groups.append(IfDefDirective(ident, True))
                        group = self.process_group(True)
                        return group
                raise ValueError()
            elif src.startswith("if"):
                self.advance(len("if"))
                self.skip_whitespaces()
                src = self.source[self.pos:]

                # The expression is evaluated after other processings.
                expr = self.process_tokens()

                if expr:
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.advance(m.end() - m.start())
                        self.groups.append(IfDirective(expr))
                        group = self.process_group(True)
                        return group

                raise ValueError()

        self.restore_pos()
        return False

    def process_endif_line(self):
        self.save_pos()
        self.skip_whitespaces()

        length = len(self.source)
        if self.source[self.pos].startswith("#"):
            self.advance(1)
            self.skip_whitespaces()

            src = self.source[self.pos:]
            if src.startswith("endif"):
                self.advance(len("endif"))
                self.skip_whitespaces()
                m = newline.match(self.source, self.pos)
                if m:
                    self.advance(m.end() - m.start())
                    self.groups.append(EndIfDirective())
                    return True

        self.restore_pos()
        return False

    def process_tokens(self):
        self.skip_whitespaces()

        tokens = []
        while True:
            token = self.parse_token()
            if not token:
                break
            tokens.append(token)
            self.skip_whitespaces()

        return tokens

    def process_elif_group(self):
        self.save_pos()
        self.skip_whitespaces()

        if self.source[self.pos].startswith("#"):
            self.advance(1)
            self.skip_whitespaces()

            src = self.source[self.pos:]
            if src.startswith("elif"):
                self.advance(len("elif"))
                self.skip_whitespaces()
                src = self.source[self.pos:]

                # The expression is evaluated after other processings.
                expr = self.process_tokens()

                if expr:
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.advance(m.end() - m.start())
                        self.groups.append(ElIfDirective(expr))
                        group = self.process_group(True)
                        return group
                raise ValueError()

        self.restore_pos()
        return False

    def process_else_group(self):
        self.save_pos()
        self.skip_whitespaces()

        if self.source[self.pos].startswith("#"):
            self.advance(1)
            self.skip_whitespaces()

            src = self.source[self.pos:]
            if src.startswith("else"):
                self.advance(len("else"))
                self.skip_whitespaces()
                m = newline.match(self.source, self.pos)
                if m:
                    self.advance(m.end() - m.start())
                    self.groups.append(ElseDirective())
                    group = self.process_group(True)
                    return group

        self.restore_pos()
        return False

    @property
    def rememaining_source(self):
        return self.source[self.pos:]

    def process_if_section(self):
        self.save_pos()
        self.skip_whitespaces()

        if self.process_if_group():
            elif_groups = []
            while True:
                elif_group = self.process_elif_group()
                if not elif_group:
                    break
                elif_groups.append(elif_group)

            else_group = self.process_else_group()
            if self.process_endif_line():
                return True

            raise NotImplementedError()

        self.restore_pos()
        return False

    def process_control_line(self):
        self.save_pos()
        self.skip_whitespaces()

        if self.source[self.pos].startswith("#"):
            self.advance(1)
            self.skip_whitespaces()

            src = self.source[self.pos:]
            if src.startswith("include"):
                self.advance(len("include"))
                src2 = self.source[self.pos:]
                self.skip_whitespaces()
                src3 = self.source[self.pos:]
                header_name = self.parse_header_name()
                if header_name:
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.advance(m.end() - m.start())
                        self.groups.append(
                            IncludeDirective(header_name))
                        return True

                raise ValueError()
            elif src.startswith("define"):
                self.advance(len("define"))
                self.skip_whitespaces()
                ident = self.parse_identifier()
                if ident:
                    self.skip_whitespaces()
                    src = self.source[self.pos:]
                    pos = self.pos
                    if src.startswith("("):
                        self.advance(1)
                        self.skip_whitespaces()

                        idents = self.parse_identifier_list()
                        self.skip_whitespaces()

                        src = self.source[self.pos:]
                        if src.startswith(")"):
                            self.advance(1)
                            self.skip_whitespaces()

                            replacements = self.process_tokens()
                            self.skip_whitespaces()
                            m = newline.match(self.source, self.pos)
                            if m:
                                self.advance(m.end() - m.start())
                                self.groups.append(
                                    DefineReplacementDirective(ident, replacements, idents))
                                return True

                    self.pos = pos

                    replacements = self.process_tokens()
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.advance(m.end() - m.start())
                        self.groups.append(
                            DefineReplacementDirective(ident, replacements))
                        return True

                raise NotImplementedError()
            elif src.startswith("undef"):
                self.advance(len("undef"))
                self.skip_whitespaces()
                ident = self.parse_identifier()
                if ident:
                    self.skip_whitespaces()
                    m = newline.match(self.source, self.pos)
                    if m:
                        self.pos = m.end()
                        self.groups.append(
                            UndefDirective(ident))
                        return True
                raise NotImplementedError()
            elif src.startswith("line"):
                self.advance(len("line"))
                raise NotImplementedError()
            elif src.startswith("error"):
                self.advance(len("error"))
                tokens = self.process_tokens()
                self.skip_whitespaces()
                m = newline.match(self.source, self.pos)
                if m:
                    self.advance(m.end() - m.start())
                    if tokens:
                        self.groups.extend(tokens)
                    return True
            elif src.startswith("pragma"):
                self.advance(len("pragma"))
                self.skip_whitespaces()
                self.process_tokens()
                self.skip_whitespaces()
                m = newline.match(self.source, self.pos)
                if m:
                    self.advance(m.end() - m.start())
                    return True

        self.restore_pos()
        return False

    def process_text_line(self):
        tokens = self.process_tokens()

        self.skip_whitespaces()
        m = newline.match(self.source, self.pos)
        if m:
            self.advance(m.end() - m.start())
            if tokens:
                self.groups.extend(tokens)
            return True

        raise ValueError()

    def process_group_part(self):
        self.skip_whitespaces()

        result = self.process_control_line()
        if result:
            return True

        result = self.process_if_section()
        if result:
            return True

        self.save_pos()

        if self.source[self.pos:].startswith("#"):
            self.advance(1)
            self.skip_whitespaces()

            if self.source[self.pos].startswith("error"):
                print("")

                raise Exception("")

            if self.source[self.pos].startswith("war"):
                print("")

                raise Exception("")

        self.restore_pos()
        return self.process_text_line()

    def process_group(self, in_if_section=False):
        parts = []
        length = len(self.source)
        while self.pos < length:
            if in_if_section:
                self.save_pos()
                self.skip_whitespaces()
                if self.source[self.pos].startswith("#"):
                    self.advance(1)
                    self.skip_whitespaces()

                    src = self.source[self.pos:]
                    if src.startswith("endif") or src.startswith("elif") or src.startswith("else"):
                        self.restore_pos()
                        break
                self.restore_pos()

            part = self.process_group_part()
            parts.append(part)

        if len(parts) > 0:
            return parts

        return None

    def include(self, filename):
        import os
        fullpath = None
        filename = str(filename)[1:-1]
        for incdir in self.include_dirs:
            path = os.path.join(incdir, filename)
            if os.path.exists(path):
                fullpath = path
                break

        if not fullpath:
            for incdir in self.system_include_dirs:
                path = os.path.join(incdir, filename)
                if os.path.exists(path):
                    fullpath = path
                    break

        if not fullpath:
            fullpath = os.path.abspath(filename)

        if not os.path.exists(fullpath):
            raise FileNotFoundError(
                "Can't found the include file : {0}".format(fullpath))

        with open(fullpath, "r") as f:
            source = f.read()

        cpp = Preprocessor(
            fullpath, source, self.include_dirs, self.system_include_dirs, self.eval_masks)

        cpp.process(self.replacements)
        return cpp

    def do_replacement(self, tokens, pos, defines):
        if str(tokens[pos]) in defines:
            macro = defines[str(tokens[pos])]
            if macro.is_func_style:
                a = tokens[pos:]
                macro_start = pos

                pos += 1
                parenthesis_nest = 1
                assert str(tokens[pos]) == "("
                pos += 1

                new_param_token = True
                params = []

                while True:
                    if str(tokens[pos]) == "(":
                        parenthesis_nest += 1
                    elif str(tokens[pos]) == ")":
                        parenthesis_nest -= 1
                        if parenthesis_nest == 0:
                            break
                    elif str(tokens[pos]) == ",":
                        if parenthesis_nest == 1:
                            new_param_token = True
                            pos += 1
                            continue

                    if new_param_token:
                        params.append([])
                        new_param_token = False

                    params[-1].append(tokens[pos])
                    pos += 1

                assert(str(tokens[pos]) == ")")
                assert(len(macro.params) == len(params))

                macro_end = pos

                pos += 1

                args = {str(arg): param for arg,
                        param in zip(macro.params, params)}

                result = []
                for token in macro.replacements:
                    if str(token) in args:
                        param = args[str(token)]
                        result.extend(param)
                    else:
                        result.append(token)

                while macro_end >= macro_start:
                    tokens.pop(macro_start)
                    macro_end -= 1

                pos = macro_start

                for r in result:
                    tokens.insert(pos, r)
                    pos += 1
            else:
                tokens.pop(pos)
                if macro.replacements:
                    for r in macro.replacements:
                        tokens.insert(pos, r)
                        pos += 1

            return True

        return False

    def process_replacement(self, tokens, replacements):
        pos = 0
        while pos < len(tokens):
            token = tokens[pos]

            if str(token) == "defined":
                macro_start = pos
                result = []

                if str(tokens[pos + 1]) == "(":
                    assert(tokens[pos + 2].ty == TokenType.Identifier)
                    if str(tokens[pos + 2]) in replacements:
                        result.append(
                            self.create_token("1", Span(0, 1), TokenType.Other))
                    else:
                        result.append(
                            self.create_token("0", Span(0, 1), TokenType.Other))
                    assert(str(tokens[pos + 3]) == ")")

                    macro_end = pos + 4
                else:
                    assert(tokens[pos + 1].ty == TokenType.Identifier)
                    if str(tokens[pos + 1]) in replacements:
                        result.append(
                            self.create_token("1", Span(0, 1), TokenType.Other))
                    else:
                        result.append(
                            self.create_token("0", Span(0, 1), TokenType.Other))

                    macro_end = pos + 2

                while macro_end > macro_start:
                    tokens.pop(macro_start)
                    macro_end -= 1

                for r in result:
                    tokens.insert(macro_start, r)
                    macro_start += 1

                continue
            elif str(tokens[pos]) == "#":
                lhs_token = tokens[pos - 1]
                rhs_token = tokens[pos + 1]
                pos += 1
                new_tok_val = f'"{str(tokens[pos])}"'
                result.append(self.create_token(
                    new_tok_val, Span(0, len(new_tok_val)), TokenType.StringLiteral))
                pos += 1
            else:
                if self.do_replacement(tokens, pos, replacements):
                    continue

            pos += 1

        return tokens

    def evalute_if_cond(self, expr_str):
        from c.lex import tokenize
        from c.parse import Context

        class TokenIndexer:
            def __init__(self, tokens):
                self.tokens = tokens

            def __getitem__(self, index):
                if index >= len(self.tokens):
                    return ""

                return self.tokens[index]

        _, expr = parse_conditional_expression(
            TokenIndexer(tokenize(expr_str)), 0, Context())
        return evalute_constant_expr(expr)

    def process(self, defines=None):
        if not defines:
            defines = {}

        self.process_group()
        self.replacements = defines
        self.processed_tokens = processed_tokens = []

        pos = 0
        while pos < len(self.groups):
            group = self.groups[pos]

            if isinstance(group, IfDirective):
                if self.eval_masks[-1].masked:
                    expr_str = " ".join([str(token) for token in self.process_replacement(
                        group.expr_tokens, self.replacements)])
                    value = self.evalute_if_cond(expr_str)
                else:
                    value = False

                self.eval_masks.append(
                    EvalMask(self.eval_masks[-1].masked and value))

                pos += 1
            elif isinstance(group, IfDefDirective):
                if self.eval_masks[-1].masked:
                    value = str(group.ident) in self.replacements
                    if group.inv:
                        value = not value
                else:
                    value = False

                self.eval_masks.append(
                    EvalMask(self.eval_masks[-1].masked and value))

                pos += 1
            elif isinstance(group, ElseDirective):
                self.eval_masks[-1].masked = self.eval_masks[-2].masked and not self.eval_masks[-1].masked
                self.eval_masks[-1].evaluated = True

                pos += 1
            elif isinstance(group, ElIfDirective):
                if not self.eval_masks[-1].evaluated:
                    expr_str = " ".join([str(token) for token in self.process_replacement(
                        group.expr_tokens, self.replacements)])

                    if self.eval_masks[-2].masked:
                        value = self.evalute_if_cond(expr_str)

                    self.eval_masks[-1].masked = self.eval_masks[-2].masked and value
                    if self.eval_masks[-1].masked:
                        self.eval_masks[-1].evaluated = True

                pos += 1
            elif isinstance(group, EndIfDirective):
                self.eval_masks.pop()

                pos += 1
            elif self.eval_masks[-1].masked:
                if isinstance(group, DefineReplacementDirective):
                    self.replacements[str(group.identifier)] = group
                    processed_tokens.append(group)

                    pos += 1
                elif isinstance(group, UndefDirective):
                    if str(group.identifier) in self.replacements:
                        self.replacements.pop(str(group.identifier))

                    pos += 1
                elif isinstance(group, IncludeDirective):
                    included = self.include(group.header_name)
                    processed_tokens.extend(included.processed_tokens)
                    self.replacements.update(**included.replacements)

                    pos += 1
                else:
                    if not self.do_replacement(
                            self.groups, pos, self.replacements):
                        processed_tokens.append(self.groups[pos])
                        pos += 1
            else:
                pos += 1

        return [group for group in processed_tokens if isinstance(group, Token)]
