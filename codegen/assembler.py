from codegen.mc import (
    MCCodeEmitter,
    MCSymbol, MCSymbolRefExpr,
    MCSection, MCInst,
    MCFragment, MCRelaxableFragment, MCDataFragment, MCAlignFragment,
    MCExpr, MCExprType, MCConstantExpr,
    MCFixup, MCFixupKind, get_fixup_size_by_kind, get_fixup_kind_by_size,
    MCBinaryExpr, MCBinaryOpcode, MCContext, MCValue, create_mc_constant_value, create_mc_value)


class MCAsmLayout:
    def __init__(self, asm):
        self.asm = asm
        self.sections = list(sorted(asm.sections, key=lambda sec: sec.name))
        self.symbols = list(sorted(asm.symbols, key=lambda sym: sym.name))

    def get_fragment_offset(self, fragment: MCFragment):
        if fragment is None:
            return 0

        section = fragment.section

        offset = 0
        for sibling in section.fragments:
            if sibling == fragment:
                return offset

            if isinstance(sibling, MCAlignFragment):
                align = sibling.alignment
                offset = int(int((offset + align - 1) / align) * align)
            else:
                offset += len(sibling.contents)

        raise ValueError("The fragment doesn't have valid section.")

    def get_label_offset(self, symbol: MCSymbol):
        return self.get_fragment_offset(symbol.fragment) + symbol.offset

    def get_symbol_offset(self, symbol: MCSymbol):
        return self.get_label_offset(symbol)


def to_unsigned(val, n):
    return (val + (0x1 << n)) & ((0x1 << n) - 1)


def to_signed(val, n):
    n = n & ((0x1 << n) - 1)
    return (n ^ (0x1 << n)) - (0x1 << n)


def lshr(val, n):
    return val >> n if val >= 0 else (val + 0x100000000) >> n


def compute_symbol_offset_diff(asm, layout: MCAsmLayout, a, b, addend):
    if a is None or b is None:
        return (a, b, addend)

    syma = a.symbol
    symb = b.symbol

    if syma is None or symb is None:
        return (a, b, addend)

    if not asm.writer.can_fully_resolve_symbol_rel_diff(syma, symb):
        return (a, b, addend)

    if syma.fragment == symb.fragment:
        addend += (syma.offset - symb.offset)
        return (None, None, addend)

    if syma.fragment.section == symb.fragment.section:
        addend += (layout.get_symbol_offset(syma) -
                   layout.get_symbol_offset(symb))
        return (None, None, addend)

    return (a, b, addend)


def symbolic_add(asm, layout: MCAsmLayout, lhs: MCValue, rhs_syma: MCSymbolRefExpr, rhs_symb: MCSymbolRefExpr, rhs_val: int):
    lhs_syma = lhs.symbol1
    lhs_symb = lhs.symbol2
    lhs_val = lhs.value

    lhs_syma, lhs_symb, lhs_val = compute_symbol_offset_diff(
        asm, layout, lhs_syma, lhs_symb, lhs_val)
    lhs_syma, rhs_symb, lhs_val = compute_symbol_offset_diff(
        asm, layout, lhs_syma, rhs_symb, lhs_val)

    rhs_syma, lhs_symb, lhs_val = compute_symbol_offset_diff(
        asm, layout, rhs_syma, lhs_symb, lhs_val)
    rhs_syma, rhs_symb, lhs_val = compute_symbol_offset_diff(
        asm, layout, rhs_syma, rhs_symb, lhs_val)

    if lhs_syma is not None and rhs_syma is not None:
        return None
    if lhs_symb is not None and rhs_symb is not None:
        return None

    syma = lhs_syma if lhs_syma is not None else rhs_syma
    symb = lhs_symb if lhs_symb is not None else rhs_symb
    val = lhs_val + rhs_val

    return create_mc_value(val, syma, symb)


def evaluate_expr_as_relocatable(expr: MCExpr, asm, layout: MCAsmLayout, fixup: MCFixup):
    if expr.ty == MCExprType.Constant:
        return create_mc_constant_value(expr.value)
    elif expr.ty == MCExprType.SymbolRef:
        sym = expr.symbol

        return create_mc_value(0, expr, None)
    elif expr.ty == MCExprType.Unary:
        raise NotImplementedError()
    elif expr.ty == MCExprType.Target:
        return expr.evaluate_expr_as_relocatable(layout, fixup)
    elif expr.ty == MCExprType.Binary:
        lhs, rhs = expr.operand1, expr.operand2

        lhs_val = evaluate_expr_as_relocatable(lhs, asm, layout, fixup)
        if not lhs_val:
            return None

        rhs_val = evaluate_expr_as_relocatable(rhs, asm, layout, fixup)
        if not rhs_val:
            return None

        if not lhs_val.is_constant or not rhs_val.is_constant:
            if expr.opcode not in [MCBinaryOpcode.Add, MCBinaryOpcode.Sub]:
                return None

            rhs_a = rhs_val.symbol1
            rhs_b = rhs_val.symbol2

            rhs_cst = rhs_val.value
            if expr.opcode == MCBinaryOpcode.Sub:
                rhs_a, rhs_b = rhs_b, rhs_a
                rhs_cst = -rhs_cst

            return symbolic_add(asm, layout, lhs_val, rhs_a, rhs_b, rhs_cst)

        opcode = expr.opcode
        a = lhs_val.value
        b = rhs_val.value

        if opcode == MCBinaryOpcode.AShr:
            result = a >> b
        elif opcode == MCBinaryOpcode.Add:
            result = a + b
        elif opcode == MCBinaryOpcode.And:
            result = a & b
        elif opcode == MCBinaryOpcode.Div:
            result = a / b
        elif opcode == MCBinaryOpcode.Mod:
            result = a % b
        elif opcode == MCBinaryOpcode.EQ:
            result = a == b
        elif opcode == MCBinaryOpcode.GT:
            result = a > b
        elif opcode == MCBinaryOpcode.GTE:
            result = a >= b
        elif opcode == MCBinaryOpcode.LAnd:
            result = a and b
        elif opcode == MCBinaryOpcode.LOr:
            result = a or b
        elif opcode == MCBinaryOpcode.LShr:
            result = lshr(a, b)
        elif opcode == MCBinaryOpcode.LT:
            result = a < b
        elif opcode == MCBinaryOpcode.LTE:
            result = a <= b
        elif opcode == MCBinaryOpcode.Mul:
            result = a * b
        elif opcode == MCBinaryOpcode.NE:
            result = a != b
        elif opcode == MCBinaryOpcode.Or:
            result = a | b
        elif opcode == MCBinaryOpcode.Shl:
            result = a << b
        elif opcode == MCBinaryOpcode.Sub:
            result = a - b
        elif opcode == MCBinaryOpcode.Xor:
            result = a ^ b

        if opcode in [MCBinaryOpcode.EQ, MCBinaryOpcode.GT, MCBinaryOpcode.GTE, MCBinaryOpcode.LT, MCBinaryOpcode.LTE, MCBinaryOpcode.NE]:
            return create_mc_constant_value(1 if result else 0)

        return create_mc_constant_value(result)


def evaluate_expr_as_constant(expr: MCExpr, asm, layout: MCAsmLayout):
    if isinstance(expr, MCConstantExpr):
        return expr.value

    value = evaluate_expr_as_relocatable(expr, asm, layout, None)

    if value is not None and value.is_constant:
        return value.value

    return None


class MCAsmInfo:
    def __init__(self):
        pass

    def get_8bit_data_directive(self):
        raise NotImplementedError()

    def get_16bit_data_directive(self):
        raise NotImplementedError()

    def get_32bit_data_directive(self):
        raise NotImplementedError()

    def get_64bit_data_directive(self):
        raise NotImplementedError()

    @property
    def has_dot_type_dot_size_directive(self):
        return True


class MCFixupKindInfo:
    def __init__(self, name, offset, size, flags):
        self.name = name
        self.offset = offset
        self.size = size
        self.flags = flags


from enum import Flag, auto


class MCFixupKindInfoFlag(Flag):
    Non = auto()
    IsPCRel = auto()
    IsAlignedDownTo32Bits = auto()


class MCAsmBackend:
    def __init__(self):
        pass

    def may_need_relaxation(self, inst: MCInst):
        raise NotImplementedError()

    def relax_instruction(self, inst: MCInst):
        pass

    def get_fixup_kind_info(self, kind):
        table = {
            MCFixupKind.PCRel_1: MCFixupKindInfo("PCRel_1", 0, 8, MCFixupKindInfoFlag.IsPCRel),
            MCFixupKind.PCRel_2: MCFixupKindInfo("PCRel_2", 0, 16, MCFixupKindInfoFlag.IsPCRel),
            MCFixupKind.PCRel_4: MCFixupKindInfo("PCRel_4", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            MCFixupKind.PCRel_8: MCFixupKindInfo("PCRel_8", 0, 64, MCFixupKindInfoFlag.IsPCRel),
            MCFixupKind.Data_1: MCFixupKindInfo("Data_1", 0, 8, MCFixupKindInfoFlag.Non),
            MCFixupKind.Data_2: MCFixupKindInfo("Data_2", 0, 16, MCFixupKindInfoFlag.Non),
            MCFixupKind.Data_4: MCFixupKindInfo("Data_4", 0, 32, MCFixupKindInfoFlag.Non),
            MCFixupKind.Data_8: MCFixupKindInfo("Data_8", 0, 64, MCFixupKindInfoFlag.Non),
            MCFixupKind.SecRel_1: MCFixupKindInfo("SecRel_1", 0, 8, MCFixupKindInfoFlag.Non),
            MCFixupKind.SecRel_2: MCFixupKindInfo("SecRel_2", 0, 16, MCFixupKindInfoFlag.Non),
            MCFixupKind.SecRel_4: MCFixupKindInfo("SecRel_4", 0, 32, MCFixupKindInfoFlag.Non),
            MCFixupKind.SecRel_8: MCFixupKindInfo("SecRel_8", 0, 64, MCFixupKindInfoFlag.Non),
        }

        return table[kind]

    def apply_fixup(self, fixup: MCFixup, fixed_value: int, contents):
        raise NotImplementedError()

    def should_force_relocation(self, asm, fixup, target):
        return False


class MCAssembler:
    def __init__(self, context: MCContext, backend: MCAsmBackend, emitter: MCCodeEmitter, writer):
        self.context = context
        self.backend = backend
        self.emitter = emitter
        self.writer = writer
        self.sections = set()
        self.symbols = set()

    def register_section(self, section):
        if not section in self.sections:
            self.sections.add(section)
            return True

        return False

    def register_symbol(self, symbol):
        if symbol.is_registered:
            return

        self.symbols.add(symbol)
        symbol.is_registered = True

    def evaluate_fixup(self, layout: MCAsmLayout, fragment: MCFragment, fixup: MCFixup):
        target = evaluate_expr_as_relocatable(fixup.value, self, layout, fixup)

        assert(target is not None)

        is_pcrel = self.backend.get_fixup_kind_info(
            fixup.kind).flags & MCFixupKindInfoFlag.IsPCRel == MCFixupKindInfoFlag.IsPCRel

        resolved = True
        if is_pcrel:
            if target.symbol2 is not None:
                resolved = False
            elif target.symbol1 is None:
                resolved = False
            else:
                resolved = self.writer.can_fully_resolve_symbol_rel_diff(
                    target.symbol1.symbol, fragment)
        else:
            resolved = target.is_constant

        value = target.value
        syma = target.symbol1
        symb = target.symbol2
        if syma and syma.symbol.fragment:
            value += layout.get_symbol_offset(syma.symbol)
        if symb and symb.symbol.fragment:
            value += layout.get_symbol_offset(symb.symbol)

        if is_pcrel:
            offset = layout.get_fragment_offset(fragment) + fixup.offset

            value -= offset

        if self.backend.should_force_relocation(self, fixup, target):
            resolved = False

        return (resolved, target, value)

    def relax_instruction(self, layout: MCAsmLayout, fragment: MCRelaxableFragment):
        if not self.backend.may_need_relaxation(fragment.inst):
            return False

        need_relaxation = False

        for fixup in fragment.fixups:
            resolved, target, evaluated = self.evaluate_fixup(
                layout, fragment, fixup)

            assert(evaluated is not None)

            if evaluated < -128 or evaluated > 127:
                need_relaxation = True
                break

        if not need_relaxation:
            return False

        inst = fragment.inst

        self.backend.relax_instruction(inst)

        from io import BytesIO

        with BytesIO() as output:
            fixups = []
            self.emitter.encode_instruction(inst, fixups, output)

            fragment.contents = bytearray(output.getvalue())
            fragment.fixups = fixups

        return True

    def step_layout_section(self, layout: MCAsmLayout, section: MCSection):
        relaxed = False
        for fragment in section.fragments:
            if isinstance(fragment, MCRelaxableFragment):
                relaxed |= self.relax_instruction(layout, fragment)

        return relaxed

    def step_layout(self, layout: MCAsmLayout):
        relaxed = False

        for section in self.sections:
            relaxed |= self.step_layout_section(layout, section)

        return relaxed

    def layout(self, layout):
        while self.step_layout(layout):
            pass

        self.writer.compute_after_layout(self, layout)

        fixups = []

        for section in layout.sections:
            for fragment in section.fragments:
                if isinstance(fragment, MCAlignFragment):
                    continue

                fixups.extend(fragment.fixups)

        for section in layout.sections:
            for fragment in section.fragments:
                if isinstance(fragment, MCDataFragment):
                    fixups = fragment.fixups
                    contents = fragment.contents
                elif isinstance(fragment, MCRelaxableFragment):
                    fixups = fragment.fixups
                    contents = fragment.contents
                else:
                    continue

                for fixup in fixups:
                    resolved, target, fixed_val = self.evaluate_fixup(
                        layout, fragment, fixup)
                    if not resolved:
                        fixed_val = self.writer.record_relocation(
                            self, layout, fragment, fixup, target, fixed_val)

                    self.backend.apply_fixup(fixup, fixed_val, contents)

    def finalize(self):
        layout = MCAsmLayout(self)
        self.layout(layout)
        self.writer.write_object(self, layout)
