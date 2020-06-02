
from c.lex import *
from ast.node import *
from c.parse_utils import *

assignment_operator = [
    "=", "*=", "/=", "%=", "+=", "-=", "<=", ">=", "&=", "^=", "|="
]


def parse_primary_expression(tokens, pos, ctx):
    save_pos = []

    # identifier
    if isinstance(tokens[pos], Identifier):
        return (pos + 1, IdentExpr(str(tokens[pos])))

    # constant
    if isinstance(tokens[pos], FloatingConstant):
        t = Type('float', None)
        val = float(str(tokens[pos]))
        return (pos + 1, FloatingConstantExpr(val, t))

    if isinstance(tokens[pos], IntegerConstant):
        val = str(tokens[pos])
        if val.startswith('0x'):
            val = val[2:]

            if len(val) <= 8:
                val = struct.unpack('>i', bytes.fromhex(val))[0]
                t = Type('int', None)
            else:
                raise NotImplementedError()

        else:
            val = int(val)
            t = Type('int', None)
        return (pos + 1, IntegerConstantExpr(val, t))

    # string-literal
    if isinstance(tokens[pos], StringLiteral):
        raise NotImplementedError()

    # ( expression )
    save_pos.append(pos)
    if str(tokens[pos]) == "(":
        pos += 1
        (pos, expr) = parse_expression(tokens, pos, ctx)
        if expr:
            if str(tokens[pos]) == ")":
                return (pos + 1, expr)
    pos = save_pos.pop()

    # generic-selection
    save_pos.append(pos)
    if str(tokens[pos]) == "_Generic":
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
