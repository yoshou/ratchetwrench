from enum import Enum, auto


class MCFragment:
    def __init__(self):
        self.section = None


class MCDataFragment(MCFragment):
    def __init__(self):
        super().__init__()

        self.contents = bytearray()
        self.fixups = []


class MCRelaxableFragment(MCFragment):
    def __init__(self, inst):
        super().__init__()

        self.inst = inst
        self.contents = bytearray()
        self.fixups = []


class MCAlignFragment(MCFragment):
    def __init__(self, alignment, value, value_size):
        super().__init__()

        self.alignment = alignment
        self.value = value
        self.value_size = value_size


class MCDummyFragment:
    def __init__(self, section):
        self.section = section


class MCSection:
    def __init__(self):
        self.fragments = []
        self.alignment = 1

    def add_fragment(self, fragment: MCFragment):
        self.fragments.append(fragment)
        fragment.section = self


class MCSymbolContents(Enum):
    Unset = auto()
    Offset = auto()
    Variable = auto()
    Common = auto()
    TargetCommon = auto()


class MCSymbolAttribute(Enum):
    Invalid = 0    # Not a valid directive.

    # Various directives in alphabetical order.
    Cold = auto()                # .cold (MachO)
    ELF_TypeFunction = auto()    # .type _foo = auto() STT_FUNC  # aka @function
    ELF_TypeIndFunction = auto()  # .type _foo = auto() STT_GNU_IFUNC
    ELF_TypeObject = auto()      # .type _foo = auto() STT_OBJECT  # aka @object
    ELF_TypeTLS = auto()         # .type _foo = auto() STT_TLS     # aka @tls_object
    ELF_TypeCommon = auto()      # .type _foo = auto() STT_COMMON  # aka @common
    ELF_TypeNoType = auto()      # .type _foo = auto() STT_NOTYPE  # aka @notype
    ELF_TypeGnuUniqueObject = auto()  # .type _foo = auto() @gnu_unique_object
    Global = auto()              # .globl
    Hidden = auto()              # .hidden (ELF)
    IndirectSymbol = auto()      # .indirect_symbol (MachO)
    Internal = auto()            # .internal (ELF)
    LazyReference = auto()       # .lazy_reference (MachO)
    Local = auto()               # .local (ELF)
    NoDeadStrip = auto()         # .no_dead_strip (MachO)
    SymbolResolver = auto()      # .symbol_resolver (MachO)
    AltEntry = auto()            # .alt_entry (MachO)
    PrivateExtern = auto()       # .private_extern (MachO)
    Protected = auto()           # .protected (ELF)
    Reference = auto()           # .reference (MachO)
    Weak = auto()                # .weak
    WeakDefinition = auto()      # .weak_definition (MachO)
    WeakReference = auto()       # .weak_reference (MachO)
    WeakDefAutoPrivate = auto()   # .weak_def_can_be_hidden (MachO)


class MCSymbol:
    def __init__(self, name, temporary):
        self.name = name
        self._fragment = None
        self._offset = 0
        self._value = 0
        self.temporary = temporary
        self.flags = 0
        self.contents = MCSymbolContents.Unset
        self.is_registered = False

    @property
    def fragment(self):
        return self._fragment

    @fragment.setter
    def fragment(self, value):
        self._fragment = value

    @property
    def section(self):
        if self.fragment is None:
            return None

        return self.fragment.section

    @property
    def is_variable(self):
        return self.contents == MCSymbolContents.Variable

    @property
    def variable_value(self):
        assert(self.is_variable)
        return self._value

    @variable_value.setter
    def variable_value(self, value):
        self._value = value
        self.contents = MCSymbolContents.Variable

    @property
    def offset(self):
        assert(self.contents in [
               MCSymbolContents.Unset, MCSymbolContents.Offset])
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value
        self.contents = MCSymbolContents.Offset

    @property
    def is_undefined(self):
        return self.fragment == None


class MCExprType(Enum):
    Binary = auto()    # Binary expressions.
    Constant = auto()  # Constant expressions.
    SymbolRef = auto()  # References to labels and assigned expressions.
    Unary = auto()     # Unary expressions.
    Target = auto()     # Target specific expression.


class MCExpr:
    def __init__(self, ty):
        self.ty = ty


class MCConstantExpr(MCExpr):
    def __init__(self, value: int):
        assert(isinstance(value, int))
        super().__init__(MCExprType.Constant)
        self.value = value

    def print(self, output, asm_info):
        output.write(str(self.value))


class MCVariantKind(Enum):
    Non = auto()
    Invalid = auto()

    GOT = auto()
    GOTOFF = auto()
    GOTREL = auto()
    GOTPCREL = auto()
    GOTTPOFF = auto()
    INDNTPOFF = auto()
    NTPOFF = auto()
    GOTNTPOFF = auto()
    PLT = auto()
    TLSGD = auto()
    TLSLD = auto()
    TLSLDM = auto()
    TPOFF = auto()
    DTPOFF = auto()
    TLSCALL = auto()   # symbol(tlscall)
    TLSDESC = auto()   # symbol(tlsdesc)
    TLVP = auto()      # Mach-O thread local variable relocations
    TLVPPAGE = auto()
    TLVPPAGEOFF = auto()
    PAGE = auto()
    PAGEOFF = auto()
    GOTPAGE = auto()
    GOTPAGEOFF = auto()
    SECREL = auto()
    SIZE = auto()      # symbol@SIZE
    WEAKREF = auto()   # The link between the symbols in .weakref foo = auto() bar


class MCSymbolRefExpr(MCExpr):
    def __init__(self, symbol: MCSymbol, kind=MCVariantKind.Non):
        assert(isinstance(symbol, MCSymbol))
        super().__init__(MCExprType.SymbolRef)
        self.symbol = symbol
        self.kind = kind


class MCUnaryOpcode(Enum):
    LNot = auto()  # Logical negation.
    Minus = auto()  # Unary minus.
    Not = auto()  # Bitwise negation.
    Plus = auto()  # Unary plus.


class MCUnaryExpr(MCExpr):
    def __init__(self, opcode: MCUnaryOpcode, operand: MCExpr):
        super().__init__(MCExprType.Unary)
        self.opcode = opcode
        self.operand = operand


class MCBinaryOpcode(Enum):
    Add = auto()  # Addition.
    And = auto()  # Bitwise and.
    Div = auto()  # Signed division.
    EQ = auto()  # Equality comparison.
    GT = auto()  # Signed greater than comparison (result is either 0 or some target-specific non-zero value)
    # Signed greater than or equal comparison (result is either 0 or some target-specific non-zero value).
    GTE = auto()
    LAnd = auto()  # Logical and.
    LOr = auto()  # Logical or.
    # Signed less than comparison (result is either 0 or some target-specific non-zero value).
    LT = auto()
    # Signed less than or equal comparison (result is either 0 or some target-specific non-zero value).
    LTE = auto()
    Mod = auto()  # Signed remainder.
    Mul = auto()  # Multiplication.
    NE = auto()  # Inequality comparison.
    Or = auto()  # Bitwise or.
    Shl = auto()  # Shift left.
    AShr = auto()  # Arithmetic shift right.
    LShr = auto()  # Logical shift right.
    Sub = auto()  # Subtraction.
    Xor = auto()  # Bitwise exclusive or.


class MCBinaryExpr(MCExpr):
    def __init__(self, opcode, operand1: MCExpr, operand2: MCExpr):
        super().__init__(MCExprType.Binary)
        self.opcode = opcode
        self.operand1 = operand1
        self.operand2 = operand2


class MCTargetExpr(MCExpr):
    def __init__(self):
        super().__init__(MCExprType.Target)

    def evaluate_expr_as_relocatable(self, layout, fixup):
        raise NotImplementedError()


class MCOperandType(Enum):
    Invalid = auto()
    Reg = auto()
    Imm = auto()
    FPImm = auto()
    Expr = auto()
    Inst = auto()


class MCOperand:
    def __init__(self):
        self.target_flags = 0

    @property
    def operand_type(self):
        return MCOperandType.Invalid

    @property
    def is_imm(self):
        return isinstance(self, MCOperandImm)

    @property
    def is_expr(self):
        return isinstance(self, MCOperandExpr)


class MCOperandReg(MCOperand):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg

    @property
    def operand_type(self):
        return MCOperandType.Reg


class MCOperandImm(MCOperand):
    def __init__(self, imm):
        super().__init__()
        self.imm = imm

    @property
    def operand_type(self):
        return MCOperandType.Imm


class MCOperandFPImm(MCOperand):
    def __init__(self, imm):
        super().__init__()
        self.imm = imm

    @property
    def operand_type(self):
        return MCOperandType.FPImm


class MCOperandExpr(MCOperand):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    @property
    def operand_type(self):
        return MCOperandType.Expr


class MCOperandInst(MCOperand):
    def __init__(self, inst):
        super().__init__()
        self.inst = inst

    @property
    def operand_type(self):
        return MCOperandType.Inst


class MCInst:
    def __init__(self, opcode):
        self.opcode = opcode
        self.operands = []

    def add_operand(self, op):
        self.operands.append(op)


class MCSymbolCOFF(MCSymbol):
    def __init__(self, name, temporary):
        super().__init__(name, temporary)

        self.type = 0

    @property
    def symbol_type(self):
        return self.type

    @symbol_type.setter
    def symbol_type(self, value):
        self.type = value

    @property
    def symbol_class(self):
        return self.flags & 0xFF

    @symbol_class.setter
    def symbol_class(self, value):
        self.flags = self.flags & ~0xFF | value & 0xFF


class ELFSymbolType(Enum):
    STT_NOTYPE = 0     # Symbol's type is not specified
    STT_OBJECT = 1     # Symbol is a data object (variable array etc.)
    STT_FUNC = 2       # Symbol is executable code (function etc.)
    STT_SECTION = 3    # Symbol refers to a section
    STT_FILE = 4       # Local absolute symbol that refers to a file
    STT_COMMON = 5     # An uninitialized common block
    STT_TLS = 6        # Thread local data object
    STT_GNU_IFUNC = 10  # GNU indirect function
    STT_LOOS = 10      # Lowest operating system-specific symbol type
    STT_HIOS = 12      # Highest operating system-specific symbol type
    STT_LOPROC = 13    # Lowest processor-specific symbol type
    STT_HIPROC = 15    # Highest processor-specific symbol type


class ELFSymbolBinding(Enum):
    STB_LOCAL = 0  # Local symbol not visible outside obj file containing def
    STB_GLOBAL = 1  # Global symbol visible to all object files being combined
    STB_WEAK = 2   # Weak symbol like global but lower-precedence
    STB_GNU_UNIQUE = 10
    STB_LOOS = 10   # Lowest operating system-specific binding type
    STB_HIOS = 12   # Highest operating system-specific binding type
    STB_LOPROC = 13  # Lowest processor-specific binding type
    STB_HIPROC = 15  # Highest processor-specific binding type


class ELFSymbolVisibility(Enum):
    STV_DEFAULT = 0  # Visibility is specified by binding type
    STV_INTERNAL = 1  # Defined by processor supplements
    STV_HIDDEN = 2   # Not visible to other components
    STV_PROTECTED = 3  # Visible in other components but not preemptable


class MCSymbolELF(MCSymbol):
    def __init__(self, name, temporary):
        super().__init__(name, temporary)

        self.size = None

    @property
    def ty(self):
        val = (self.flags >> 0) & 7
        return ELFSymbolType(val)

    @ty.setter
    def ty(self, val: ELFSymbolType):
        other_flags = self.flags & ~0x7
        self.flags = other_flags | ((val.value << 0) & 7)

    @property
    def is_defined(self):
        return not self.is_undefined

    @property
    def binding(self):
        if self.is_binding_set:
            val = (self.flags >> 3) & 3
            return ELFSymbolBinding(val)

        if self.is_defined:
            return ELFSymbolBinding.STB_LOCAL

        return ELFSymbolBinding.STB_GLOBAL

    @binding.setter
    def binding(self, val: ELFSymbolBinding):
        self.set_is_binding_set()

        other_flags = self.flags & ~(0x3 << 3)
        self.flags = other_flags | ((val.value & 3) << 3)

    @property
    def is_binding_set(self):
        val = (self.flags >> 12) & 1
        return val != 0

    def set_is_binding_set(self):
        other_flags = self.flags & ~(0x1 << 12)
        self.flags = other_flags | (0x1 << 12)

    @property
    def visibility(self):
        val = (self.flags >> 5) & 3
        return ELFSymbolVisibility(val)

    @visibility.setter
    def visibility(self, val: ELFSymbolVisibility):
        other_flags = self.flags & ~(0x3 << 5)
        self.flags = other_flags | ((val.value & 3) << 5)

    @property
    def other(self):
        val = (self.flags >> 7) & 7
        return val << 5

    @other.setter
    def other(self, val: int):
        other_flags = self.flags & ~(0x7 << 5)
        self.flags = other_flags | (((val >> 5) & 7) << 7)


def get_global_prefix():
    return ""


def get_private_global_prefix():
    return ".L"


class MCContext:
    def __init__(self, obj_file_info):
        self.symbol_table = {}
        self.obj_file_info = obj_file_info
        self.name_ids = {}

    def get_or_create_symbol(self, name: str):
        if name in self.symbol_table:
            return self.symbol_table[name]

        symbol = self.symbol_table[name] = self.create_symbol(name)
        return symbol

    def create_temp_symbol(self, name: str, always_add_suffix=True):
        return self.create_symbol(get_private_global_prefix() + name, always_add_suffix)

    def create_symbol(self, name: str, always_add_suffix=False):
        is_temporary = False

        if name not in self.name_ids:
            self.name_ids[name] = 0
            if always_add_suffix:
                name = name + str(self.name_ids[name])
        else:
            self.name_ids[name] += 1
            name = name + str(self.name_ids[name])

        if str(name).startswith(get_private_global_prefix()):
            is_temporary = True

        if self.obj_file_info.is_elf:
            symbol = MCSymbolELF(name, is_temporary)
        elif self.obj_file_info.is_coff:
            symbol = MCSymbolCOFF(name, is_temporary)

        return symbol


class MCInstPrinter:
    def __init__(self):
        pass

    def print_inst(self, inst: MCInst, output):
        raise NotImplementedError()


class MCTargetWriter:
    def __init__(self):
        pass

    def print_inst(self, inst, output, printer):
        printer.print_inst(inst, output)


class MCStream:
    def __init__(self, context: MCContext, target_stream=None):
        self.context = context
        self.target_stream = target_stream

    def init_sections(self):
        pass

    def switch_section(self, section):
        pass

    def finalize(self):
        pass

    def emit_syntax_directive(self):
        pass


class MCValue:
    def __init__(self, value: int, symbol1: MCSymbol, symbol2: MCSymbol):
        self.value = value
        self.symbol1 = symbol1
        self.symbol2 = symbol2

    @property
    def is_constant(self):
        return self.symbol1 is None and self.symbol2 is None

    @property
    def access_variant(self):
        if self.symbol2 is not None:
            assert(self.symbol2.kind == MCVariantKind.Non)

        if self.symbol1 is None:
            return MCVariantKind.Non

        if self.symbol1.kind == MCVariantKind.WEAKREF:
            return MCVariantKind.Non

        return self.symbol1.kind


def create_mc_constant_value(value: int):
    return MCValue(value, None, None)


def create_mc_value(value, symbol1: MCSymbol, symbol2: MCSymbol):
    return MCValue(value, symbol1, symbol2)


class MCFixupKind(Enum):
    Noop = auto()   # A no-op fixup.
    Data_1 = auto()     # A one-byte fixup.
    Data_2 = auto()     # A two-byte fixup.
    Data_4 = auto()     # A four-byte fixup.
    Data_8 = auto()     # A eight-byte fixup.
    PCRel_1 = auto()    # A one-byte pc relative fixup.
    PCRel_2 = auto()    # A two-byte pc relative fixup.
    PCRel_4 = auto()    # A four-byte pc relative fixup.
    PCRel_8 = auto()    # A eight-byte pc relative fixup.
    GPRel_1 = auto()    # A one-byte gp relative fixup.
    GPRel_2 = auto()    # A two-byte gp relative fixup.
    GPRel_4 = auto()    # A four-byte gp relative fixup.
    GPRel_8 = auto()    # A eight-byte gp relative fixup.
    DTPRel_4 = auto()   # A four-byte dtp relative fixup.
    DTPRel_8 = auto()   # A eight-byte dtp relative fixup.
    TPRel_4 = auto()    # A four-byte tp relative fixup.
    TPRel_8 = auto()    # A eight-byte tp relative fixup.
    SecRel_1 = auto()   # A one-byte section relative fixup.
    SecRel_2 = auto()   # A two-byte section relative fixup.
    SecRel_4 = auto()   # A four-byte section relative fixup.
    SecRel_8 = auto()   # A eight-byte section relative fixup.
    Data_Add_1 = auto()  # A one-byte add fixup.
    Data_Add_2 = auto()  # A two-byte add fixup.
    Data_Add_4 = auto()  # A four-byte add fixup.
    Data_Add_8 = auto()  # A eight-byte add fixup.
    Data_Sub_1 = auto()  # A one-byte sub fixup.
    Data_Sub_2 = auto()  # A two-byte sub fixup.
    Data_Sub_4 = auto()  # A four-byte sub fixup.
    Data_Sub_8 = auto()  # A eight-byte sub fixup.


def get_fixup_kind_by_size(size, pcrel):
    lookup = {
        1: [MCFixupKind.PCRel_1, MCFixupKind.Data_1],
        2: [MCFixupKind.PCRel_2, MCFixupKind.Data_2],
        4: [MCFixupKind.PCRel_4, MCFixupKind.Data_4],
        8: [MCFixupKind.PCRel_8, MCFixupKind.Data_8],
    }

    return lookup[size][0 if pcrel else 1]


def get_fixup_size_by_kind(kind: MCFixupKind):
    if kind == MCFixupKind.Noop:
        return 0
    elif kind in [MCFixupKind.PCRel_1, MCFixupKind.SecRel_1, MCFixupKind.Data_1]:
        return 1
    elif kind in [MCFixupKind.PCRel_2, MCFixupKind.SecRel_2, MCFixupKind.Data_2]:
        return 2
    elif kind in [MCFixupKind.PCRel_4, MCFixupKind.SecRel_4, MCFixupKind.Data_4]:
        return 4
    elif kind in [MCFixupKind.PCRel_8, MCFixupKind.SecRel_8, MCFixupKind.Data_8]:
        return 8

    raise ValueError("kind")


def get_fixup_kind_for_size(size, is_pcrel):
    if size == 1:
        return MCFixupKind.PCRel_1 if is_pcrel else MCFixupKind.Data_1
    elif size == 2:
        return MCFixupKind.PCRel_2 if is_pcrel else MCFixupKind.Data_2
    elif size == 4:
        return MCFixupKind.PCRel_4 if is_pcrel else MCFixupKind.Data_4
    elif size == 8:
        return MCFixupKind.PCRel_8 if is_pcrel else MCFixupKind.Data_8

    raise ValueError("kind")


class MCFixup:
    def __init__(self, offset: int, value: MCExpr, kind: MCFixupKind):
        self.offset = offset
        self.value = value
        self.kind = kind


class MCCodeEmitter:
    def __init__(self):
        pass

    def encode_instruction(self, inst: MCInst, fixups, output):
        raise NotImplementedError()


class SectionKind(Enum):
    Metadata = auto()

    # text
    Text = auto()
    ExecuteOnly = auto()

    # rodata
    ReadOnly = auto()
    Mergeable1ByteCString = auto()
    Mergeable2ByteCString = auto()
    Mergeable4ByteCString = auto()
    MergeableConst4 = auto()
    MergeableConst8 = auto()
    MergeableConst16 = auto()
    MergeableConst32 = auto()

    # thread
    ThreadBSS = auto()
    ThreadData = auto()

    # bss
    BSS = auto()
    BSSLocal = auto()
    BSSExtern = auto()
    Common = auto()

    # data
    Data = auto()
    ReadOnlyWithRel = auto()
