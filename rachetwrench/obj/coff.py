from rachetwrench.obj.utils import *
import struct
from rachetwrench.obj.objfile import *


class DOS_Header(Struct):
    magic = ArrayField(UInt8L(0), 2)
    used_bytes_in_the_last_page = UInt16L(0)
    file_size_in_pages = UInt16L(0)
    number_of_relocation_items = UInt16L(0)
    header_size_in_paragraphs = UInt16L(0)
    minimum_extra_paragraphs = UInt16L(0)
    maximum_extra_paragraphs = UInt16L(0)
    initial_relative_ss = UInt16L(0)
    initial_sp = UInt16L(0)
    checksum = UInt16L(0)
    initial_ip = UInt16L(0)
    initial_relative_cs = UInt16L(0)
    address_of_relocation_table = UInt16L(0)
    overlay_number = UInt16L(0)
    reserved = ArrayField(UInt16L(0), 4)
    oem_id = UInt16L(0)
    oem_info = UInt16L(0)
    reserved2 = ArrayField(UInt16L(0), 10)
    address_of_new_exe_header = UInt32L(0)


class COFF_Header(Struct):
    machine = UInt16L(0)
    number_of_sections = UInt16L(0)
    time_date_stamp = UInt32L(0)
    pointer_to_symbol_table = UInt32L(0)
    number_of_symbols = UInt32L(0)
    size_of_optional_header = UInt16L(0)
    characteristics = UInt16L(0)


class COFF_Section(Struct):
    name = ArrayField(UInt8L(0), 8)
    virtual_size = UInt32L(0)
    virtual_address = UInt32L(0)
    size_of_raw_data = UInt32L(0)
    pointer_to_raw_data = UInt32L(0)
    pointer_to_relocations = UInt32L(0)
    pointer_to_linenumbers = UInt32L(0)
    number_of_relocations = UInt16L(0)
    number_of_linenumbers = UInt16L(0)
    characteristics = UInt32L(0)


class COFF_Symbol(Struct):
    name_or_offset = ArrayField(UInt8L(0), 8)
    value = UInt32L(0)
    section_number = UInt16L(0)
    type = UInt16L(0)
    storage_class = UInt8L(0)
    number_of_aux_symbols = UInt8L(0)

    @property
    def name(self):
        return self.name_or_offset.values

    @property
    def offset(self):
        return struct.unpack("I", bytearray(self.name_or_offset.values[4:8]))[0]

    @property
    def has_offset(self):
        return struct.unpack("I", bytearray(self.name_or_offset.values[:4]))[0] == 0


class COFF_Relocation(Struct):
    virtual_address = UInt32L(0)
    symbol_table_index = UInt32L(0)
    type = UInt16L(0)


def check_size(data, size):
    return len(data) >= size


from io import BytesIO


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


class COFFObjectFile(ObjectFile):
    def __init__(self):
        self.coff_header = None
        self.coff_big_obj_header = None
        self.section_table = []
        self.symbol_table = []
        self.symbol_number_to_index = {}
        self.data = bytes()

    @property
    def is_coff(self):
        return True

    @property
    def symbol_table_entry_size(self):
        if self.coff_header is not None:
            return COFF_Symbol().size

        raise ValueError("Symbol table is not found")

    @property
    def pointer_to_symbol_table(self):
        if self.coff_header is not None:
            return self.coff_header.pointer_to_symbol_table.value

        raise ValueError("Symbol table is not found")

    @property
    def number_of_symbols(self):
        if self.coff_header is not None:
            return self.coff_header.number_of_symbols.value

        raise ValueError("Symbol table is not found")

    @property
    def sections(self):
        result = []
        for sec in self.section_table:
            result.append(SectionRef(self, SectionRefItem(sec)))

        return result

    @property
    def symbols(self):
        result = []
        for sym in self.symbol_table:
            result.append(SymbolRef(self, SymbolRefItem(sym)))

        return result

    def _get_string(self, data):
        if data[0] == 0:
            index = int.from_bytes(data[4:8], 'little')
            data = self.string_table[index:]

        data = bytes(data)

        end = 0
        while end < len(data) and data[end] != 0:
            end += 1
        return data[:end].decode()

    def get_section(self, index):
        return self.section_table[index]

    def get_section_relocations(self, section):
        num_relocs = section.number_of_relocations.value

        if num_relocs == 0:
            return []

        relocs = []
        with BytesIO(self.data[section.pointer_to_relocations.value:]) as stream:
            for i in range(num_relocs):
                reloc = COFF_Relocation()
                reloc.deserialize(stream)
                relocs.append(RelocationRef(self, RelocationRefItem(reloc)))

        return relocs

    def get_section_contents(self, section):
        if section.pointer_to_raw_data.value == 0:
            return bytearray()

        offset = section.pointer_to_raw_data.value
        size = section.size_of_raw_data.value
        return bytearray(self.data[offset:(offset+size)])

    def get_section_alignment(self, section):
        from rachetwrench.codegen.coff import IMAGE_SCN_TYPE_NO_PAD
        characteristics = section.characteristics.value
        if characteristics & IMAGE_SCN_TYPE_NO_PAD != 0:
            return 1

        shift = (characteristics >> 20) & 0xF

        if shift > 0:
            return 0x1 << (shift - 1)
        return 16

    def get_section_address(self, section):
        return section.virtual_address.value + self.image_base

    def get_section_size(self, section):
        return section.size_of_raw_data.value

    def get_section_name(self, section):
        return self._get_string(section.name.values)

    def get_section_is_text(self, section):
        from rachetwrench.codegen.coff import IMAGE_SCN_CNT_CODE
        return section.characteristics.value & IMAGE_SCN_CNT_CODE != 0

    def get_section_is_virtual(self, section):
        return section.pointer_to_raw_data.value == 0

    def get_relocated_section(self, section):
        raise NotImplementedError()

    def get_relocation_symbol(self, reloc):
        symbol = self.symbol_table[self.symbol_number_to_index[reloc.symbol_table_index.value]]
        return SymbolRef(self, SymbolRefItem(symbol))

    def get_relocation_offset(self, reloc):
        return reloc.virtual_address.value

    def get_relocation_ty(self, reloc):
        return reloc.type.value

    def get_symbol_name(self, symbol):
        return self._get_string(symbol.name)

    def get_symbol_section(self, symbol):
        if symbol.section_number.value == 0:
            return None
        section = self.get_section(symbol.section_number.value - 1)
        return SectionRef(self, SectionRefItem(section))

    def get_symbol_type(self, symbol):
        from rachetwrench.codegen.coff import IMAGE_SYM_DTYPE_FUNCTION
        SCT_COMPLEX_TYPE_SHIFT = 4

        ty = symbol.type.value
        if (ty & 0xF0) >> SCT_COMPLEX_TYPE_SHIFT == IMAGE_SYM_DTYPE_FUNCTION:
            return SymbolType.Function

        if symbol.section_number.value > 0:
            return SymbolType.Data

        return SymbolType.Other

    def get_symbol_flags(self, symbol):
        from rachetwrench.codegen.coff import IMAGE_SYM_ABSOLUTE, IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_UNDEFINED
        result = SymbolFlags.Non
        section_number = symbol.section_number.value
        value = self.get_symbol_value(symbol)

        if symbol.storage_class.value == IMAGE_SYM_CLASS_EXTERNAL:
            result |= SymbolFlags.Global

        if symbol.section_number.value == IMAGE_SYM_ABSOLUTE:
            result |= SymbolFlags.Absolute

        if symbol.storage_class.value == IMAGE_SYM_CLASS_EXTERNAL and section_number == IMAGE_SYM_UNDEFINED and value == 0:
            result |= SymbolFlags.Undefined

        return result

    @property
    def image_base(self):
        return 0

    def get_symbol_address(self, symbol):
        result = self.get_symbol_value(symbol)
        section = self.get_section(symbol.section_number.value - 1)

        result += section.virtual_address.value

        result += self.image_base

        return result

    def get_symbol_value(self, symbol):
        return symbol.value.value

    def setup_symbol_table(self):
        data = self.data

        if self.coff_header is not None:
            ptr = self.pointer_to_symbol_table
            with BytesIO(data[ptr:]) as stream:
                i = 0
                while i < self.number_of_symbols:
                    symbol = COFF_Symbol()
                    symbol.deserialize(stream)
                    self.symbol_table.append(symbol)

                    self.symbol_number_to_index[i] = len(self.symbol_table) - 1
                    i += 1

                    for j in range(symbol.number_of_aux_symbols.value):
                        stream.read(18)
                        i += 1

        string_table_offset = self.pointer_to_symbol_table + \
            self.symbol_table_entry_size * self.number_of_symbols

        self.string_table = bytearray()
        with BytesIO(data[string_table_offset:]) as stream:
            self.string_table.extend(stream.read(4))
            string_table_size = struct.unpack("I", self.string_table)[0]
            self.string_table.extend(stream.read(string_table_size - 4))

    def parse_from(self, data):
        self.data = bytes(data)

        dos_header = DOS_Header()

        position = 0
        has_pe_header = False
        if check_size(data, dos_header.size * 4):
            with BytesIO(data) as stream:
                dos_header.deserialize(stream)

            if dos_header.magic[0] == 'M' and dos_header.magic[1] == 'Z':
                has_pe_header = True

        self.coff_header = COFF_Header()
        with BytesIO(data[position:]) as stream:
            self.coff_header.deserialize(stream)

        if not has_pe_header and self.coff_header.machine == IMAGE_FILE_MACHINE_UNKNOWN and coff_header.number_of_sections == 0xFFFF:
            raise NotImplementedError()

        if self.coff_header is not None:
            position += self.coff_header.size

        if has_pe_header:
            raise NotImplementedError()

        if self.coff_header is not None:
            position += self.coff_header.size_of_optional_header.value

        with BytesIO(data[position:]) as stream:
            for i in range(self.coff_header.number_of_sections.value):
                section = COFF_Section()
                section.deserialize(stream)
                self.section_table.append(section)

        if self.pointer_to_symbol_table != 0:
            self.setup_symbol_table()
        else:
            assert(self.number_of_symbols == 0)
