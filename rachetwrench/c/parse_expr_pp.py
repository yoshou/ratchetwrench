
from rachetwrench.c.lex import *
from rachetwrench.ast.node import *
from rachetwrench.c.parse_utils import *

assignment_operator = [
    "=", "*=", "/=", "%=", "+=", "-=", "<=", ">=", "&=", "^=", "|="
]


def parse_primary_expression(tokens, pos, ctx):
    save_pos = []

    # identifier
    if isinstance(tokens[pos], Identifier):
        return (pos + 1, IdentExpr(tokens[pos].value))

    from rachetwrench.ast.types import PrimitiveType

    # constant
    if isinstance(tokens[pos], FloatingConstant):
        token = tokens[pos]

        t = PrimitiveType('double')
        if "f" in token.suffix or "F" in token.suffix:
            t = PrimitiveType('float')
        val = float(tokens[pos].value)
        return (pos + 1, FloatingConstantExpr(val, t))

    if isinstance(tokens[pos], IntegerConstant):
        val = tokens[pos].value
        suffix = tokens[pos].suffix

        if val.startswith('0x'):
            base = 0
        elif val.startswith('0'):
            base = 8
        else:
            base = 10

        is_unsigned = "U" in suffix or "u" in suffix
        is_long = "L" in suffix or "l" in suffix

        value = int(val, base)
        if base in [0, 8]:
            if (value & ((1 << 31) - 1)) == value:
                t = PrimitiveType('int')
                bits = 32
            elif (value & ((1 << 32) - 1)) == value:
                t = PrimitiveType('unsigned int')
                bits = 32
            elif (value & ((1 << 63) - 1)) == value:
                t = PrimitiveType('long')
                bits = 64
            elif (value & ((1 << 64) - 1)) == value:
                t = PrimitiveType('unsigned long')
                bits = 64
            else:
                raise ValueError("The constant isn't representive.")
        else:
            if (value & ((1 << 32) - 1)) == value:
                t = PrimitiveType('int')
                bits = 32
            elif (value & ((1 << 64) - 1)) == value:
                t = PrimitiveType('long')
                bits = 64
            else:
                raise ValueError("The constant isn't representive.")

        if is_unsigned:
            if is_long:
                t = PrimitiveType("unsigned long")
            else:
                t = PrimitiveType("unsigned int")
        else:
            if is_long:
                t = PrimitiveType("long")

        def sign_extend(value, bits):
            sign_bit = 1 << (bits - 1)
            return (value & (sign_bit - 1)) - (value & sign_bit)

        if t.name in ["int", "long"]:
            value = sign_extend(value, bits)

        return (pos + 1, IntegerConstantExpr(value, t))

    def unescape(val: str):
        val = val.replace("\\'", "\'")
        val = val.replace("\\\"", "\"")
        val = val.replace("\\?", "?")
        val = val.replace("\\\\", "\\")
        val = val.replace("\\a", "\a")
        val = val.replace("\\b", "\b")
        val = val.replace("\\f", "\f")
        val = val.replace("\\n", "\n")
        val = val.replace("\\r", "\r")
        val = val.replace("\\t", "\t")
        val = val.replace("\\v", "\v")

        while True:
            m = re.search(r"\\(?:(x[0-9A-Fa-f]+)|([0-9]))", val)

            if not m:
                break

            hex_val, octet_val = m.groups()

            if hex_val:
                hex_num = int("0" + hex_val, 0)
                assert(hex_num >= 0 and hex_num < 256)
                val = val[:m.start()] + chr(hex_num) + val[m.end():]

            if octet_val in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                octet_num = int(octet_val, 8)
                assert(octet_num >= 0 and octet_num < 8)
                val = val[:m.start()] + chr(octet_num) + val[m.end():]

        return val

    if isinstance(tokens[pos], CharacterConstant):
        val = tokens[pos].value
        val = unescape(val)
        val = ord(val)
        t = PrimitiveType('char')
        return (pos + 1, IntegerConstantExpr(val, t))

    from rachetwrench.ast.types import PointerType, PrimitiveType, ArrayType

    # string-literal
    if isinstance(tokens[pos], StringLiteral):
        val = tokens[pos].value
        val = unescape(val)
        t = ArrayType(PrimitiveType("char"), len(val) + 1)
        return (pos + 1, StringLiteralExpr(val, t))

    # ( expression )
    save_pos.append(pos)
    if tokens[pos].value == "(":
        pos += 1
        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr:
            if tokens[pos].value == ")":
                return (pos + 1, expr)
    pos = save_pos.pop()

    # generic-selection
    save_pos.append(pos)
    if tokens[pos].value == "_Generic":
        raise NotImplementedError()
    pos = save_pos.pop()

    return (pos, None)


def parse_argument_expression_list(tokens, pos, ctx):
    # assignment-expression
    # expression , assignment-expression
    return parse_list(tokens, pos, ctx, parse_assignment_expression)


def parse_postfix_expression_tail(tokens, pos, ctx, lhs):
    save_pos = []

    # [ expression ]
    save_pos.append(pos)
    if str(tokens[pos]) == "[":
        pos += 1

        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == "]":
                pos += 1
                raise NotImplementedError()
    pos = save_pos.pop()

    # ( argument-expression-list_opt )
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, arg_expr_list) = parse_argument_expression_list(tokens, pos, ctx)

        if str(tokens[pos]) == ")":
            pos += 1
            return (pos, arg_expr_list)
    pos = save_pos.pop()

    # postfix-expression . identifier
    # postfix-expression -> identifier
    save_pos.append(pos)
    if str(tokens[pos]) in [".", "->"]:
        pos += 1
        if isinstance(tokens[pos], Identifier):
            return (pos + 1, AccessorOp(lhs, Ident(str(tokens[pos]))))
    pos = save_pos.pop()

    # ++
    # --
    save_pos.append(pos)
    if str(tokens[pos]) in ["++", "--"]:
        op = str(tokens[pos])
        pos += 1
        return (pos, PostOp(op, lhs))
    pos = save_pos.pop()

    return (pos, None)


def parse_postfix_expression_head(tokens, pos, ctx):
    save_pos = []

    # ( type-name ) { initializer-list }
    # ( type-name ) { initializer-list , }
    # save_pos.append(pos)
    # if str(tokens[pos]) == "(":
    #     pos += 1
    #     (pos, typename) = parse_type_name(tokens, pos, ctx)
    #     if typename:
    #         if str(tokens[pos]) == ")":
    #             pos += 1
    #             if str(tokens[pos]) == "{":
    #                 pos += 1
    #                 (pos, init_list) = parse_initializer_list(tokens, pos, ctx)

    #                 if str(tokens[pos]) == ",":
    #                     pos += 1

    #                 if init_list:
    #                     if str(tokens[pos]) == "}":
    #                         pos += 1
    #                         raise NotImplementedError()
    # pos = save_pos.pop()

    # primary-expression
    (pos, expr) = parse_primary_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    return (pos, None)


def parse_postfix_expression(tokens, pos, ctx):
    # primary_expression
    # postfix-expression [ expression ]
    # postfix-expression ( argument-expression-list_opt )
    # postfix-expression . identifier
    # postfix-expression -> identifier
    # postfix-expression ++
    # postfix-expression --
    # ( type-name ) { initializer-list }
    # ( type-name ) { initializer-list , }
    (pos, expr) = parse_postfix_expression_head(tokens, pos, ctx)
    if expr:
        while True:
            (pos, tail) = parse_postfix_expression_tail(tokens, pos, ctx, expr)
            if not tail:
                break

            expr = tail
        return (pos, expr)

    return (pos, None)


def gen_parse_operator_generic(ops, parse_operand):
    def func(tokens, pos, ctx):
        save_pos = []

        save_pos.append(pos)
        (pos, lhs) = parse_operand(tokens, pos, ctx)

        if lhs:
            while str(tokens[pos]) in ops:
                save_pos.append(pos)
                op = str(tokens[pos])
                pos += 1

                (pos, rhs) = parse_operand(tokens, pos, ctx)

                if not rhs:
                    pos = save_pos.pop()
                    break

                lhs = BinaryOp(op, lhs, rhs)

            return (pos, lhs)

        pos = save_pos.pop()

        return (pos, None)

    return func


unary_operator = ['&', '*', '+', '-', '~', '!']


def parse_unary_expression(tokens, pos, ctx):
    save_pos = []

    # ++ unary_expression
    # -- unary_expression
    save_pos.append(pos)
    if str(tokens[pos]) in ["--", "++"]:
        op = str(tokens[pos])
        pos += 1
        (pos, expr) = parse_unary_expression(tokens, pos, ctx)
        if expr:
            return (pos, UnaryOp(op, expr))
    pos = save_pos.pop()

    # postfix_expression
    (pos, expr) = parse_postfix_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    # unary-operator cast-expression
    save_pos.append(pos)
    if str(tokens[pos]) in unary_operator:
        op = str(tokens[pos])
        pos += 1
        (pos, expr) = parse_cast_expression(tokens, pos, ctx)
        if expr:
            return (pos, UnaryOp(op, expr))
    pos = save_pos.pop()

    if str(tokens[pos]) == "sizeof":
        raise NotImplementedError()

    if str(tokens[pos]) == "_Alignof":
        raise NotImplementedError()

    return (pos, None)


def parse_cast_expression(tokens, pos, ctx):
    save_pos = []

    # unary-expression
    (pos, unary_expr) = parse_unary_expression(tokens, pos, ctx)
    if unary_expr:
        return (pos, unary_expr)

    # ( type-name ) cast-expression
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, typename) = parse_type_name(tokens, pos, ctx)
        if typename:
            if str(tokens[pos]) == ")":
                pos += 1
                (pos, expr) = parse_cast_expression(tokens, pos, ctx)
                if expr:
                    return (pos, CastExpr(expr, typename))
    pos = save_pos.pop()

    return (pos, None)


parse_multiplicative_expression = gen_parse_operator_generic(
    ["*", "/", "%"], parse_cast_expression)

parse_additive_expression = gen_parse_operator_generic(
    ["+", "-"], parse_multiplicative_expression)

parse_shift_expression = gen_parse_operator_generic(
    ["<<", ">>"], parse_additive_expression)

parse_relational_expression = gen_parse_operator_generic(
    ["<", ">", "<=", ">="], parse_shift_expression)

parse_equality_expression = gen_parse_operator_generic(
    ["==", "!="], parse_relational_expression)

parse_and_expression = gen_parse_operator_generic(
    ["&"], parse_equality_expression)

parse_exclusive_or_expression = gen_parse_operator_generic(
    ["^"], parse_and_expression)

parse_inclusive_or_expression = gen_parse_operator_generic(
    ["|"], parse_exclusive_or_expression)

parse_logical_and_expression = gen_parse_operator_generic(
    ["&&"], parse_inclusive_or_expression)

parse_logical_or_expression = gen_parse_operator_generic(
    ["||"], parse_logical_and_expression)


def parse_conditional_expression(tokens, pos, ctx):
    save_pos = []

    # logical-OR-expression
    # logical-OR-expression ? expression : conditional-expression
    (pos, cond_expr) = parse_logical_or_expression(tokens, pos, ctx)
    if cond_expr:
        save_pos.append(pos)
        if str(tokens[pos]) == "?":
            pos += 1

            (pos, true_expr) = parse_expression(tokens, pos, ctx)
            if true_expr:
                if str(tokens[pos]) == ":":
                    pos += 1

                    (pos, false_expr) = parse_conditional_expression(
                        tokens, pos, ctx)
                    if false_expr:
                        return (pos, ConditionalExpr(cond_expr, true_expr, false_expr))
        pos = save_pos.pop()

        return (pos, cond_expr)

    return (pos, None)


def parse_assignment_expression(tokens, pos, ctx):
    save_pos = []

    # unary-expression assignment-operator assignment-expression
    save_pos.append(pos)
    (pos, lhs) = parse_unary_expression(tokens, pos, ctx)
    if lhs:
        if str(tokens[pos]) in assignment_operator:
            op = str(tokens[pos])
            pos += 1
            (pos, rhs) = parse_assignment_expression(tokens, pos, ctx)
            if rhs:
                return (pos, BinaryOp(op, lhs, rhs))
    pos = save_pos.pop()

    # conditional-expression
    (pos, expr) = parse_conditional_expression(tokens, pos, ctx)
    if expr:
        return (pos, expr)

    return (pos, None)


def parse_expression(tokens, pos, ctx):
    # assignment-expression
    # expression , assignment-expression
    lst = parse_list(tokens, pos, ctx, parse_assignment_expression)
    if len(lst) == 1:
        return lst[0]

    return lst
