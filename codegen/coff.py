from codegen.obj import (
    MCObjectStream,
    MCObjectWriter,
    MCObjectTargetWriter,
    MCObjectFileInfo)

from codegen.mc import(
    MCContext,
    MCSection,
    MCFragment,
    MCInst,
    MCSymbol, MCSymbolAttribute, MCSymbolELF,
    MCExpr, MCSymbolRefExpr, MCVariantKind,
    MCFixup, MCFixupKind,
    MCValue,
    MCDataFragment,
    MCRelaxableFragment,
    MCAlignFragment,
    MCCodeEmitter,
    SectionKind)

from codegen.assembler import MCAsmBackend, MCAssembler, MCAsmLayout, evaluate_expr_as_constant

from io import BytesIO

IMAGE_SCN_TYPE_NOLOAD = 0x00000002
IMAGE_SCN_TYPE_NO_PAD = 0x00000008
IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
IMAGE_SCN_LNK_OTHER = 0x00000100
IMAGE_SCN_LNK_INFO = 0x00000200
IMAGE_SCN_LNK_REMOVE = 0x00000800
IMAGE_SCN_LNK_COMDAT = 0x00001000
IMAGE_SCN_GPREL = 0x00008000
IMAGE_SCN_MEM_PURGEABLE = 0x00020000
IMAGE_SCN_MEM_16BIT = 0x00020000
IMAGE_SCN_MEM_LOCKED = 0x00040000
IMAGE_SCN_MEM_PRELOAD = 0x00080000
IMAGE_SCN_ALIGN_1BYTES = 0x00100000
IMAGE_SCN_ALIGN_2BYTES = 0x00200000
IMAGE_SCN_ALIGN_4BYTES = 0x00300000
IMAGE_SCN_ALIGN_8BYTES = 0x00400000
IMAGE_SCN_ALIGN_16BYTES = 0x00500000
IMAGE_SCN_ALIGN_32BYTES = 0x00600000
IMAGE_SCN_ALIGN_64BYTES = 0x00700000
IMAGE_SCN_ALIGN_128BYTES = 0x00800000
IMAGE_SCN_ALIGN_256BYTES = 0x00900000
IMAGE_SCN_ALIGN_512BYTES = 0x00A00000
IMAGE_SCN_ALIGN_1024BYTES = 0x00B00000
IMAGE_SCN_ALIGN_2048BYTES = 0x00C00000
IMAGE_SCN_ALIGN_4096BYTES = 0x00D00000
IMAGE_SCN_ALIGN_8192BYTES = 0x00E00000
IMAGE_SCN_LNK_NRELOC_OVFL = 0x01000000
IMAGE_SCN_MEM_DISCARDABLE = 0x02000000
IMAGE_SCN_MEM_NOT_CACHED = 0x04000000
IMAGE_SCN_MEM_NOT_PAGED = 0x08000000
IMAGE_SCN_MEM_SHARED = 0x10000000
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

IMAGE_REL_I386_ABSOLUTE = 0x0000
IMAGE_REL_I386_DIR16 = 0x0001
IMAGE_REL_I386_REL16 = 0x0002
IMAGE_REL_I386_DIR32 = 0x0006
IMAGE_REL_I386_DIR32NB = 0x0007
IMAGE_REL_I386_SEG12 = 0x0009
IMAGE_REL_I386_SECTION = 0x000A
IMAGE_REL_I386_SECREL = 0x000B
IMAGE_REL_I386_TOKEN = 0x000C
IMAGE_REL_I386_SECREL7 = 0x000D
IMAGE_REL_I386_REL32 = 0x0014

IMAGE_REL_AMD64_ABSOLUTE = 0x0000
IMAGE_REL_AMD64_ADDR64 = 0x0001
IMAGE_REL_AMD64_ADDR32 = 0x0002
IMAGE_REL_AMD64_ADDR32NB = 0x0003
IMAGE_REL_AMD64_REL32 = 0x0004
IMAGE_REL_AMD64_REL32_1 = 0x0005
IMAGE_REL_AMD64_REL32_2 = 0x0006
IMAGE_REL_AMD64_REL32_3 = 0x0007
IMAGE_REL_AMD64_REL32_4 = 0x0008
IMAGE_REL_AMD64_REL32_5 = 0x0009
IMAGE_REL_AMD64_SECTION = 0x000A
IMAGE_REL_AMD64_SECREL = 0x000B
IMAGE_REL_AMD64_SECREL7 = 0x000C
IMAGE_REL_AMD64_TOKEN = 0x000D
IMAGE_REL_AMD64_SREL32 = 0x000E
IMAGE_REL_AMD64_PAIR = 0x000F
IMAGE_REL_AMD64_SSPAN32 = 0x0010

IMAGE_REL_ARM_ABSOLUTE = 0x0000
IMAGE_REL_ARM_ADDR32 = 0x0001
IMAGE_REL_ARM_ADDR32NB = 0x0002
IMAGE_REL_ARM_BRANCH24 = 0x0003
IMAGE_REL_ARM_BRANCH11 = 0x0004
IMAGE_REL_ARM_TOKEN = 0x0005
IMAGE_REL_ARM_BLX24 = 0x0008
IMAGE_REL_ARM_BLX11 = 0x0009
IMAGE_REL_ARM_REL32 = 0x000A
IMAGE_REL_ARM_SECTION = 0x000E
IMAGE_REL_ARM_SECREL = 0x000F
IMAGE_REL_ARM_MOV32A = 0x0010
IMAGE_REL_ARM_MOV32T = 0x0011
IMAGE_REL_ARM_BRANCH20T = 0x0012
IMAGE_REL_ARM_BRANCH24T = 0x0014
IMAGE_REL_ARM_BLX23T = 0x0015
IMAGE_REL_ARM_PAIR = 0x0016

IMAGE_REL_ARM64_ABSOLUTE = 0x0000
IMAGE_REL_ARM64_ADDR32 = 0x0001
IMAGE_REL_ARM64_ADDR32NB = 0x0002
IMAGE_REL_ARM64_BRANCH26 = 0x0003
IMAGE_REL_ARM64_PAGEBASE_REL21 = 0x0004
IMAGE_REL_ARM64_REL21 = 0x0005
IMAGE_REL_ARM64_PAGEOFFSET_12A = 0x0006
IMAGE_REL_ARM64_PAGEOFFSET_12L = 0x0007
IMAGE_REL_ARM64_SECREL = 0x0008
IMAGE_REL_ARM64_SECREL_LOW12A = 0x0009
IMAGE_REL_ARM64_SECREL_HIGH12A = 0x000A
IMAGE_REL_ARM64_SECREL_LOW12L = 0x000B
IMAGE_REL_ARM64_TOKEN = 0x000C
IMAGE_REL_ARM64_SECTION = 0x000D
IMAGE_REL_ARM64_ADDR64 = 0x000E
IMAGE_REL_ARM64_BRANCH19 = 0x000F
IMAGE_REL_ARM64_BRANCH14 = 0x0010
IMAGE_REL_ARM64_REL32 = 0x0011

# The contents of this field are assumed to be applicable to any machine type
IMAGE_FILE_MACHINE_UNKNOWN = 0x0
IMAGE_FILE_MACHINE_AM33 = 0x1d3  # Matsushita AM33
IMAGE_FILE_MACHINE_AMD64 = 0x8664  # x64
IMAGE_FILE_MACHINE_ARM = 0x1c0  # ARM little endian
IMAGE_FILE_MACHINE_ARM64 = 0xaa64  # ARM64 little endian
IMAGE_FILE_MACHINE_ARMNT = 0x1c4  # ARM Thumb-2 little endian
IMAGE_FILE_MACHINE_EBC = 0xebc  # EFI byte code
# Intel 386 or later processors and compatible processors
IMAGE_FILE_MACHINE_I386 = 0x14c
IMAGE_FILE_MACHINE_IA64 = 0x200  # Intel Itanium processor family
IMAGE_FILE_MACHINE_M32R = 0x9041  # Mitsubishi M32R little endian
IMAGE_FILE_MACHINE_MIPS16 = 0x266  # MIPS16
IMAGE_FILE_MACHINE_MIPSFPU = 0x366  # MIPS with FPU
IMAGE_FILE_MACHINE_MIPSFPU16 = 0x466  # MIPS16 with FPU
IMAGE_FILE_MACHINE_POWERPC = 0x1f0  # Power PC little endian
IMAGE_FILE_MACHINE_POWERPCFP = 0x1f1  # Power PC with floating point support
IMAGE_FILE_MACHINE_R4000 = 0x166  # MIPS little endian
IMAGE_FILE_MACHINE_RISCV32 = 0x5032  # RISC-V 32-bit address space
IMAGE_FILE_MACHINE_RISCV64 = 0x5064  # RISC-V 64-bit address space
IMAGE_FILE_MACHINE_RISCV128 = 0x5128  # RISC-V 128-bit address space
IMAGE_FILE_MACHINE_SH3 = 0x1a2  # Hitachi SH3
IMAGE_FILE_MACHINE_SH3DSP = 0x1a3  # Hitachi SH3 DSP
IMAGE_FILE_MACHINE_SH4 = 0x1a6  # Hitachi SH4
IMAGE_FILE_MACHINE_SH5 = 0x1a8  # Hitachi SH5
IMAGE_FILE_MACHINE_THUMB = 0x1c2  # Thumb
IMAGE_FILE_MACHINE_WCEMIPSV2 = 0x169  # MIPS little-endian WCE v2

IMAGE_SYM_DEBUG = -2
IMAGE_SYM_ABSOLUTE = -1
IMAGE_SYM_UNDEFINED = 0

IMAGE_SYM_CLASS_END_OF_FUNCTION = -1  # Physical end of function
IMAGE_SYM_CLASS_NULL = 0              # No symbol
IMAGE_SYM_CLASS_AUTOMATIC = 1         # Stack variable
IMAGE_SYM_CLASS_EXTERNAL = 2          # External symbol
IMAGE_SYM_CLASS_STATIC = 3            # Static
IMAGE_SYM_CLASS_REGISTER = 4          # Register variable
IMAGE_SYM_CLASS_EXTERNAL_DEF = 5      # External definition
IMAGE_SYM_CLASS_LABEL = 6             # Label
IMAGE_SYM_CLASS_UNDEFINED_LABEL = 7   # Undefined label
IMAGE_SYM_CLASS_MEMBER_OF_STRUCT = 8  # Member of structure
IMAGE_SYM_CLASS_ARGUMENT = 9          # Function argument
IMAGE_SYM_CLASS_STRUCT_TAG = 10       # Structure tag
IMAGE_SYM_CLASS_MEMBER_OF_UNION = 11  # Member of union
IMAGE_SYM_CLASS_UNION_TAG = 12        # Union tag
IMAGE_SYM_CLASS_TYPE_DEFINITION = 13  # Type definition
IMAGE_SYM_CLASS_UNDEFINED_STATIC = 14  # Undefined static
IMAGE_SYM_CLASS_ENUM_TAG = 15         # Enumeration tag
IMAGE_SYM_CLASS_MEMBER_OF_ENUM = 16   # Member of enumeration
IMAGE_SYM_CLASS_REGISTER_PARAM = 17   # Register parameter
IMAGE_SYM_CLASS_BIT_FIELD = 18        # Bit field
# ".bb" or ".eb" - beginning or end of block
IMAGE_SYM_CLASS_BLOCK = 100
# ".bf" or ".ef" - beginning or end of function
IMAGE_SYM_CLASS_FUNCTION = 101
IMAGE_SYM_CLASS_END_OF_STRUCT = 102  # End of structure
IMAGE_SYM_CLASS_FILE = 103          # File name
# Line number reformatted as symbol
IMAGE_SYM_CLASS_SECTION = 104
IMAGE_SYM_CLASS_WEAK_EXTERNAL = 105  # Duplicate tag
# External symbol in dmert public lib
IMAGE_SYM_CLASS_CLR_TOKEN = 107

IMAGE_SYM_TYPE_NULL = 0   # No type information or unknown base type.
IMAGE_SYM_TYPE_VOID = 1   # Used with void pointers and functions.
IMAGE_SYM_TYPE_CHAR = 2   # A character (signed byte).
IMAGE_SYM_TYPE_SHORT = 3  # A 2-byte signed integer.
IMAGE_SYM_TYPE_INT = 4    # A natural integer type on the target.
IMAGE_SYM_TYPE_LONG = 5   # A 4-byte signed integer.
IMAGE_SYM_TYPE_FLOAT = 6  # A 4-byte floating-point number.
IMAGE_SYM_TYPE_DOUBLE = 7  # An 8-byte floating-point number.
IMAGE_SYM_TYPE_STRUCT = 8  # A structure.
IMAGE_SYM_TYPE_UNION = 9  # An union.
IMAGE_SYM_TYPE_ENUM = 10  # An enumerated type.
IMAGE_SYM_TYPE_MOE = 11   # A member of enumeration (a specific value).
IMAGE_SYM_TYPE_BYTE = 12  # A byte; unsigned 1-byte integer.
IMAGE_SYM_TYPE_WORD = 13  # A word; unsigned 2-byte integer.
IMAGE_SYM_TYPE_UINT = 14  # An unsigned integer of natural size.
IMAGE_SYM_TYPE_DWORD = 15  # An unsigned 4-byte integer.

IMAGE_SYM_DTYPE_NULL = 0     # No complex type; simple scalar variable.
IMAGE_SYM_DTYPE_POINTER = 1  # A pointer to base type.
IMAGE_SYM_DTYPE_FUNCTION = 2  # A function that returns a base type.
IMAGE_SYM_DTYPE_ARRAY = 3    # An array of base type.

SCT_COMPLEX_TYPE_SHIFT = 4


IMAGE_COMDAT_SELECT_NODUPLICATES = 1
IMAGE_COMDAT_SELECT_ANY = 2
IMAGE_COMDAT_SELECT_SAME_SIZE = 3
IMAGE_COMDAT_SELECT_EXACT_MATCH = 4
IMAGE_COMDAT_SELECT_ASSOCIATIVE = 5
IMAGE_COMDAT_SELECT_LARGEST = 6
IMAGE_COMDAT_SELECT_NEWEST = 7


class COFFSymbolData:
    def __init__(self):
        self.name = ""
        self.value = 0
        self.section_number = 0
        self.type = 0
        self.storage_class = 0
        self.number_of_aux_symbols = 0


class COFFSymbol:
    def __init__(self, name):
        self.name = name
        self.data = COFFSymbolData()
        self.aux = []


class COFFSectionData:
    def __init__(self):
        self.name = ""
        self.virtual_size = 0
        self.virtual_address = 0
        self.size_of_raw_data = 0
        self.pointer_to_raw_data = 0
        self.pointer_to_relocations = 0
        self.pointer_to_line_numbers = 0
        self.number_of_relocations = 0
        self.number_of_line_numbers = 0
        self.characteristics = 0


class COFFSection:
    def __init__(self, name):
        self.name = name
        self.data = COFFSectionData()
        self.relocations = []


class COFFHeaderData:
    def __init__(self):
        self.machine = IMAGE_FILE_MACHINE_UNKNOWN
        self.num_of_sections = 0
        self.time_date_stamp = 0
        self.pointer_to_symbol_table = 0
        self.number_of_symbols = 0
        self.size_of_optional_header = 0
        self.characteristics = 0


class AuxiliaryFormat5:
    def __init__(self):
        self.checksum = 0


class COFFRelocationData:
    def __init__(self):
        self.virtual_address = 0
        self.symbol_table_index = 0
        self.type = 0


class COFFRelocation:
    def __init__(self):
        self.data = COFFRelocationData()


class COFFObjectFileInfo(MCObjectFileInfo):
    def __init__(self):
        super().__init__()

        self._text_section = MCSectionCOFF(".text", IMAGE_SCN_CNT_CODE)
        self._bss_section = MCSectionCOFF(
            ".bss", IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_ALIGN_4BYTES | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)
        self._data_section = MCSectionCOFF(
            ".data", IMAGE_SCN_CNT_INITIALIZED_DATA)
        self._rodata_section = MCSectionCOFF(
            ".rdata", IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ)
        self._tls_data_section = MCSectionCOFF(
            ".tls$", IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)

        self._comdat_sections = {}

    @property
    def text_section(self) -> MCSection:
        return self._text_section

    @property
    def bss_section(self) -> MCSection:
        return self._bss_section

    @property
    def data_section(self) -> MCSection:
        return self._data_section

    @property
    def rodata_section(self) -> MCSection:
        return self._rodata_section

    @property
    def tls_data_section(self) -> MCSection:
        return self._tls_data_section

    def scalar_const_to_string(self, const):
        from ir.values import ConstantInt, ConstantFP
        from ir.types import f32, f64
        import struct

        if isinstance(const, ConstantInt):
            return hex(constant.value)
        elif isinstance(const, ConstantFP):
            if const.ty == f32:
                return hex(struct.unpack('<I', struct.pack('<f', const.value))[0])
            elif const.ty == f64:
                return hex(struct.unpack('<Q', struct.pack('<d', const.value))[0])

        raise ValueError("Not supporting")

    def get_section_for_const(self, section_kind: SectionKind, value, align):
        characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_LNK_COMDAT

        comdat_sym_name = ""
        if section_kind == SectionKind.MergeableConst4:
            if align <= 4:
                comdat_sym_name = f"__real@{self.scalar_const_to_string(value)}"
                align = 4
        elif section_kind == SectionKind.MergeableConst8:
            if align <= 8:
                comdat_sym_name = f"__real@{self.scalar_const_to_string(value)}"
                align = 8
        elif section_kind == SectionKind.MergeableConst16:
            if align <= 16:
                comdat_sym_name = f"__xmm@{self.scalar_const_to_string(value)}"
                align = 16
        elif section_kind == SectionKind.MergeableConst32:
            if align <= 32:
                comdat_sym_name = f"__xmm@{self.scalar_const_to_string(value)}"
                align = 32

        if comdat_sym_name != "":
            if comdat_sym_name not in self._comdat_sections:
                comdat_section = MCSectionCOFF(".rdata", characteristics, comdat_sym=self.ctx.get_or_create_symbol(
                    comdat_sym_name), selection=IMAGE_COMDAT_SELECT_ANY)
                self._comdat_sections[comdat_sym_name] = comdat_section
            else:
                comdat_section = self._comdat_sections[comdat_sym_name]
            return comdat_section

        raise ValueError("Invalid section kind.")

    @property
    def is_elf(self):
        return False

    @property
    def is_coff(self):
        return True


class MCSectionCOFF(MCSection):
    def __init__(self, name, characteristics, comdat_sym=None, selection=0):
        super().__init__()

        self.name = name
        self.characteristics = characteristics
        self.comdat_sym = comdat_sym
        self.selection = selection

    def __hash__(self):
        return hash((self.name, self.comdat_sym))

    def __eq__(self, other):
        if not isinstance(other, MCSectionCOFF):
            return False

        return self.name == other.name and self.comdat_sym == other.comdat_sym


class WinCOFFObjectStream(MCObjectStream):
    def __init__(self, context: MCContext, backend, writer, emitter):
        super().__init__(context, backend, writer, emitter)

        self.section_stack = [None]

    def init_sections(self):
        context = self.context

        self.switch_section(context.obj_file_info.text_section)
        self.switch_section(context.obj_file_info.data_section)
        self.switch_section(context.obj_file_info.bss_section)
        self.switch_section(context.obj_file_info.text_section)

    def emit_newline(self):
        pass

    def emit_elf_size(self, symbol, size):
        pass

    def get_or_create_data_fragment(self):
        if len(self.current_section.fragments) > 0:
            if isinstance(self.current_section.fragments[-1], MCDataFragment):
                return self.current_section.fragments[-1]

        fragment = MCDataFragment()
        self.current_section.add_fragment(fragment)
        return fragment

    def emit_instruction_data_fragment(self, inst: MCInst):
        fixups = []

        with BytesIO() as output:
            self.assembler.emitter.encode_instruction(inst, fixups, output)

            fragment = self.get_or_create_data_fragment()

            for fixup in fixups:
                fixup.offset += len(fragment.contents)

            fragment.contents.extend(output.getvalue())
            fragment.fixups.extend(fixups)

    def emit_instruction_new_fragment(self, inst: MCInst):
        fixups = []

        with BytesIO() as output:
            self.assembler.emitter.encode_instruction(inst, fixups, output)

            fragment = MCRelaxableFragment(inst)
            self.current_section.add_fragment(fragment)
            fragment.contents.extend(output.getvalue())
            fragment.fixups = fixups

    def emit_instruction(self, inst: MCInst):
        if not self.assembler.backend.may_need_relaxation(inst):
            self.emit_instruction_data_fragment(inst)
            return

        self.emit_instruction_new_fragment(inst)

    def emit_symbol_attrib(self, symbol, attrib):
        self.assembler.register_symbol(symbol)

    def emit_label(self, symbol: MCSymbol):
        self.assembler.register_symbol(symbol)

        symbol.fragment = self.get_or_create_data_fragment()
        symbol.offset = len(symbol.fragment.contents)

    def emit_int_value(self, value: int, size: int):
        order = 'little'
        fragment = self.get_or_create_data_fragment()
        fragment.contents.extend(value.to_bytes(
            size, byteorder=order, signed=True))

    def emit_zeros(self, size):
        fragment = self.get_or_create_data_fragment()
        fragment.contents.extend(bytearray(size))

    def emit_bytes(self, value: bytearray):
        fragment = self.get_or_create_data_fragment()
        fragment.contents.extend(value)

    def emit_value(self, value: MCExpr, size: int):
        raise NotImplementedError()

    def emit_value_to_alignment(self, alignment, value=0, value_size=0):
        fragment = MCAlignFragment(alignment, value, value_size)
        self.current_section.add_fragment(fragment)

        if alignment > self.current_section.alignment:
            self.current_section.alignment = alignment

    @property
    def current_section(self):
        if len(self.section_stack) > 0:
            return self.section_stack[-1]

        return None

    @current_section.setter
    def current_section(self, section):
        if len(self.section_stack) > 0:
            self.section_stack[-1] = section

        self.section_stack.append(section)

    def switch_section(self, section):
        if self.current_section != section:
            self.current_section = section

            self.assembler.register_section(section)

    def emit_coff_symbol_storage_class(self, symbol, symbol_class):
        symbol.symbol_class = symbol_class

    def emit_coff_symbol_type(self, symbol, symbol_type):
        symbol.symbol_type = symbol_type


class MCWinCOFFObjectTargetWriter(MCObjectTargetWriter):
    def __init__(self):
        super().__init__()


class WinCOFFObjectWriter(MCObjectWriter):
    def __init__(self, output):
        super().__init__()

        self.output = output

        self.sections = []
        self.symbols = []
        self.header = COFFHeaderData()
        self.section_map = {}
        self.symbol_map = {}
        self.strings = bytearray(4)

    def write_header(self, obj):
        self.header.machine = IMAGE_FILE_MACHINE_AMD64
        self.header.num_of_sections = len(obj.sections)

        order = 'little'

        self.output.write(self.header.machine.to_bytes(2, byteorder=order))
        self.output.write(
            self.header.num_of_sections.to_bytes(2, byteorder=order))
        self.output.write(
            self.header.time_date_stamp.to_bytes(4, byteorder=order))
        self.output.write(
            self.header.pointer_to_symbol_table.to_bytes(4, byteorder=order))
        self.output.write(
            self.header.number_of_symbols.to_bytes(4, byteorder=order))
        self.output.write(
            self.header.size_of_optional_header.to_bytes(2, byteorder=order))
        self.output.write(
            self.header.characteristics.to_bytes(2, byteorder=order))

    def write_section_header(self, data: COFFSectionData):
        order = 'little'

        self.output.write(self.str_to_bytes(data.name, 8))

        self.output.write(data.virtual_size.to_bytes(4, byteorder=order))
        self.output.write(data.virtual_address.to_bytes(4, byteorder=order))
        self.output.write(data.size_of_raw_data.to_bytes(4, byteorder=order))
        self.output.write(
            data.pointer_to_raw_data.to_bytes(4, byteorder=order))
        self.output.write(
            data.pointer_to_relocations.to_bytes(4, byteorder=order))
        self.output.write(
            data.pointer_to_line_numbers.to_bytes(4, byteorder=order))
        self.output.write(
            data.number_of_relocations.to_bytes(2, byteorder=order))
        self.output.write(
            data.number_of_line_numbers.to_bytes(2, byteorder=order))
        self.output.write(data.characteristics.to_bytes(4, byteorder=order))

    def write_section_headers(self, obj):
        for section in self.sections:
            self.write_section_header(section.data)

    def write_sections(self, obj):
        for section in obj.sections:
            self.write_section(section)

    def str_to_bytes(self, s, length):
        data = bytearray(length)
        bys = s.encode()
        length = length if len(bys) > length else len(bys)
        data[:length] = bys[:length]
        return data

    def write_auxiliary_symbols(self, symbols):
        order = 'little'

        for aux in symbols:
            if isinstance(aux, AuxiliaryFormat5):
                self.output.write(aux.length.to_bytes(4, byteorder=order))
                self.output.write(
                    aux.number_of_relocations.to_bytes(2, byteorder=order))
                self.output.write(
                    aux.number_of_line_numbers.to_bytes(2, byteorder=order))
                self.output.write(aux.checksum.to_bytes(4, byteorder=order))
                self.output.write(aux.number.to_bytes(2, byteorder=order))
                self.output.write(aux.selection.to_bytes(1, byteorder=order))
                self.output.write(bytearray(3))

    def write_symbol(self, symbol):
        order = 'little'

        data = symbol.data

        if len(data.name.encode()) < 8:
            self.output.write(self.str_to_bytes(data.name, 8))
        else:
            self.output.write(int(0).to_bytes(4, byteorder=order))
            offset = len(self.strings)
            self.output.write(offset.to_bytes(4, byteorder=order))
            self.strings.extend((data.name + "\0").encode())

        self.output.write(data.value.to_bytes(4, byteorder=order))
        self.output.write(data.section_number.to_bytes(
            2, signed=True, byteorder=order))
        self.output.write(data.type.to_bytes(2, byteorder=order))
        self.output.write(data.storage_class.to_bytes(1, byteorder=order))
        self.output.write(
            data.number_of_aux_symbols.to_bytes(1, byteorder=order))

        self.write_auxiliary_symbols(symbol.aux)

    def write_relocation(self, data: COFFRelocationData):
        order = 'little'

        self.output.write(data.virtual_address.to_bytes(4, byteorder=order))
        self.output.write(data.symbol_table_index.to_bytes(4, byteorder=order))
        self.output.write(data.type.to_bytes(2, byteorder=order))

    def write_symbols(self):
        for symbol in self.symbols:
            self.write_symbol(symbol)

    from io import BytesIO

    def write_section_contents(self, section):
        section_size = self.calculate_section_size(section)
        if section_size > 0:
            pos = self.output.tell()

            align = section.alignment
            pos_aligned = int(int((pos + align - 1) / align) * align)

            if pos_aligned - pos > 0:
                paddings = bytearray(pos_aligned - pos)
                self.output.write(paddings)

            with BytesIO() as byte_output:
                for fragment in section.fragments:
                    if isinstance(fragment, MCAlignFragment):
                        pos = byte_output.tell()

                        align = fragment.alignment
                        pos_aligned = int(
                            int((pos + align - 1) / align) * align)

                        if pos_aligned - pos > 0:
                            paddings = bytearray(pos_aligned - pos)
                            byte_output.write(paddings)

                        continue

                    byte_output.write(fragment.contents)

                section_data = byte_output.getvalue()

            self.output.write(section_data)

            # section.symbol.aux[0].checksum = zlib.crc32(section_data) & 0xffffffff

    def write_section(self, section):
        self.write_section_contents(section)

        sec = self.section_map[section]

        for reloc in sec.relocations:
            self.write_relocation(reloc.data)

    def calculate_section_size(self, section):
        size = 0
        for fragment in section.fragments:
            if isinstance(fragment, MCAlignFragment):
                align = fragment.alignment
                size = int(int((size + align - 1) / align) * align)
                continue

            size += len(fragment.contents)

        return size

    def calculate_section_offset(self, obj):
        Header16Size = 20
        Header32Size = 56
        NameSize = 8
        Symbol16Size = 18
        Symbol32Size = 20
        SectionSize = 40
        RelocationSize = 10

        offset = 0

        offset += Header16Size
        offset += SectionSize * len(obj.sections)

        for section in obj.sections:
            sec = self.section_map[section]

            section_size = self.calculate_section_size(section)

            sec.data.size_of_raw_data = section_size
            if section_size == 0:
                sec.data.pointer_to_raw_data = 0
            else:
                align = section.alignment
                offset = int(int((offset + align - 1) / align) * align)

                sec.data.pointer_to_raw_data = offset
                offset += section_size

            sec.data.pointer_to_relocations = offset
            sec.data.number_of_relocations = len(sec.relocations)
            offset += RelocationSize * len(sec.relocations)

            for reloc in sec.relocations:
                reloc_sym = self.symbol_map[reloc.symbol]
                reloc.data.symbol_table_index = reloc_sym.index

            aux = sec.symbol.aux[0]

            aux.length = section_size
            aux.number_of_relocations = sec.data.number_of_relocations
            aux.number_of_line_numbers = 0
            aux.number = 1 + obj.sections.index(section)

        self.header.pointer_to_symbol_table = offset

    def define_section(self, section: MCSectionCOFF, asm, obj):
        sym = self.create_symbol(section.name)
        sec = self.create_section(section.name)

        if section.selection != IMAGE_COMDAT_SELECT_ASSOCIATIVE:
            if section.comdat_sym is not None:
                comdat_sym = self.get_or_create_symbol(section.comdat_sym)
                comdat_sym.section = sec

        #sym.data.section_number = obj.sections.index(section) + 1
        sym.data.storage_class = IMAGE_SYM_CLASS_STATIC
        sym.data.number_of_aux_symbols = 1

        sym.aux = []
        sym.aux.append(AuxiliaryFormat5())
        sym.aux[0].selection = section.selection

        sym.section = sec

        sec.data.name = sec.name
        sec.data.characteristics = section.characteristics
        sec.symbol = sym
        sec.mcsec = section

        if section.alignment == 16:
            sec.data.characteristics |= IMAGE_SCN_ALIGN_16BYTES
        elif section.alignment == 32:
            sec.data.characteristics |= IMAGE_SCN_ALIGN_32BYTES

        self.section_map[section] = sec

    def get_symbol_value(self, symbol: MCSymbol, layout):
        return layout.get_symbol_offset(symbol)

    def get_or_create_symbol(self, symbol: MCSymbol):
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]

        sym = self.symbol_map[symbol] = self.create_symbol(symbol.name)
        return sym

    def get_base_symbol(self, symbol):
        if not symbol.is_variable:
            return symbol

        raise NotImplementedError()

    def define_symbol(self, symbol: MCSymbol, asm, obj):
        sym = self.get_or_create_symbol(symbol)

        base_symbol = self.get_base_symbol(symbol)
        section = None
        if base_symbol is not None and base_symbol.fragment is not None:
            section = self.section_map[symbol.fragment.section]

        if base_symbol is None:
            sym.data.section_number = IMAGE_SYM_ABSOLUTE

        sym.section = section
        sym.data.storage_class = IMAGE_SYM_CLASS_EXTERNAL
        sym.data.value = 0

        sym.aux = []

        sym.data.value = self.get_symbol_value(symbol, obj)

    def compute_after_layout(self, asm, obj):
        for section in obj.sections:
            self.define_section(section, asm, obj)

        for symbol in obj.symbols:
            if not symbol.temporary:
                self.define_symbol(symbol, asm, obj)

    def create_file_symbol(self, asm):
        sym = self.create_symbol(".file")
        sym.data.section_number = IMAGE_SYM_DEBUG
        sym.data.storage_class = IMAGE_SYM_CLASS_FILE

    def write_string_table(self):
        data = self.strings
        size = len(data)

        data[:4] = size.to_bytes(4, byteorder="little")

        self.output.write(data)

    def write_object(self, asm, obj):
        for num, section in enumerate(self.sections):
            section.number = num + 1
            section.symbol.data.section_number = num + 1
            #section.symbol.aux[0].auxsection_definition.number = num + 1

        self.header.number_of_symbols = 0
        for symbol in self.symbols:
            symbol.index = self.header.number_of_symbols
            self.header.number_of_symbols += 1
            self.header.number_of_symbols += symbol.data.number_of_aux_symbols

            if symbol.section is not None:
                symbol.data.section_number = symbol.section.number

        self.calculate_section_offset(obj)

        self.write_header(obj)
        self.write_section_headers(obj)
        self.write_sections(obj)
        self.write_symbols()
        self.write_string_table()

    def create_symbol(self, name):
        symbol = COFFSymbol(name)
        symbol.data.name = name
        self.symbols.append(symbol)
        return symbol

    def create_section(self, name):
        section = COFFSection(name)
        self.sections.append(section)
        return section

    def get_relocation_type(self, fixup: MCFixup, target: MCValue):
        fixup_kind = fixup.kind

        machine = IMAGE_FILE_MACHINE_AMD64

        from codegen.x64_asm_printer import X64FixupKind

        if machine == IMAGE_FILE_MACHINE_AMD64:
            if fixup_kind == MCFixupKind.PCRel_4:
                return IMAGE_REL_AMD64_REL32
            elif fixup_kind == MCFixupKind.Data_4:
                return IMAGE_REL_AMD64_ADDR32
            elif fixup_kind == MCFixupKind.Data_8:
                return IMAGE_REL_AMD64_ADDR64
            elif fixup_kind == MCFixupKind.Data_8:
                return IMAGE_REL_AMD64_ADDR64
            elif fixup_kind == MCFixupKind.SecRel_2:
                return IMAGE_REL_AMD64_SECTION
            elif fixup_kind == MCFixupKind.SecRel_4:
                return IMAGE_REL_AMD64_SECREL
            elif fixup_kind == X64FixupKind.Reloc_RIPRel_4:
                return IMAGE_REL_AMD64_REL32
            else:
                raise NotImplementedError()
        elif machine == IMAGE_FILE_MACHINE_I386:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def record_relocation(self, asm: MCAssembler, layout: MCAsmLayout, fragment: MCFragment, fixup: MCFixup, target: MCValue, fixed_val):
        reloc = COFFRelocation()

        section = fragment.section
        sec = self.section_map[section]

        reloc.data.virtual_address = layout.get_fragment_offset(fragment)
        reloc.data.virtual_address += fixup.offset

        reloc.data.type = self.get_relocation_type(fixup, target)

        reloc.symbol = target.symbol1.symbol

        sec.relocations.append(reloc)

        fixed_val = 0

        return fixed_val

    def can_fully_resolve_symbol_rel_diff(self, a, b):
        symbol_type = a.symbol_type

        if symbol_type >> SCT_COMPLEX_TYPE_SHIFT == IMAGE_SYM_DTYPE_FUNCTION:
            return False

        return super().can_fully_resolve_symbol_rel_diff(a, b)
