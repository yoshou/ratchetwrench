

def parse_list(tokens, pos, ctx, func, delim=','):
    save_pos = []

    lst = []

    (pos, param_decl) = func(tokens, pos, ctx)
    if param_decl:
        lst.append(param_decl)

    while True:
        save_pos.append(pos)
        if str(tokens[pos]) != delim:
            break

        pos += 1
        (pos, param_decl) = func(tokens, pos, ctx)
        if not param_decl:
            pos = save_pos.pop()
            break

        lst.append(param_decl)

    if len(lst) > 0:
        return (pos, lst)

    return (pos, None)
