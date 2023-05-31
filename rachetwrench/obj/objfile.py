from enum import Enum, Flag, auto


class ObjectFile:
    def __init__(self):
        pass

    @property
    def number_of_symbols(self):
        raise NotImplementedError()

    @property
    def symbols(self):
        raise NotImplementedError()

    @property
    def is_elf(self):
        return False

    @property
    def is_coff(self):
        return False


class SectionRef:
    def __init__(self, obj, item):
        self.obj = obj
        self.item = item

    def __eq__(self, other):
        if other is None:
            return False

        return self.item == other.item

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.item)

    @property
    def name(self):
        return self.obj.get_section_name(self.item.value)

    @property
    def relocations(self):
        return self.obj.get_section_relocations(self.item.value)

    @property
    def contents(self):
        return self.obj.get_section_contents(self.item.value)

    @property
    def alignment(self):
        return self.obj.get_section_alignment(self.item.value)

    @property
    def flags(self):
        return self.obj.get_section_flags(self.item.value)

    @property
    def is_text(self):
        return self.obj.get_section_is_text(self.item.value)

    @property
    def is_virtual(self):
        return self.obj.get_section_is_virtual(self.item.value)

    @property
    def size(self):
        return self.obj.get_section_size(self.item.value)

    @property
    def address(self):
        return self.obj.get_section_address(self.item.value)

    @property
    def type(self):
        return self.obj.get_section_type(self.item.value)

    @property
    def relocated_section(self):
        return self.obj.get_relocated_section(self.item.value)


class SymbolRefItem:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if other is None:
            return False

        return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.value)


class SymbolFlags(Flag):
    Non = 0
    Undefined = 0x1 << 0      # Symbol is defined in another object file
    Global = 0x1 << 1         # Global symbol
    Weak = 0x1 << 2           # Weak symbol
    Absolute = 0x1 << 3       # Absolute symbol
    Common = 0x1 << 4         # Symbol has common linkage
    Indirect = 0x1 << 5       # Symbol is an alias to another symbol
    Exported = 0x1 << 6       # Symbol is visible to other DSOs
    FormatSpecific = 0x1 << 7  # Specific to the object file format
    Thumb = 0x1 << 8          # Thumb symbol in a 32-bit ARM binary
    Hidden = 0x1 << 9         # Symbol has hidden visibility
    Const = 0x1 << 10         # Symbol value is constant
    Executable = 0x1 << 11    # Symbol points to an executable section


class SymbolType(Enum):
    Unknown = auto()
    Data = auto()
    Debug = auto()
    File = auto()
    Function = auto()
    Other = auto()


class SymbolRef:
    def __init__(self, obj, item):
        self.obj = obj
        self.item = item

    @property
    def name(self):
        return self.obj.get_symbol_name(self.item.value)

    @property
    def address(self):
        return self.obj.get_symbol_address(self.item.value)

    @property
    def value(self):
        return self.obj.get_symbol_value(self.item.value)

    @property
    def section(self):
        return self.obj.get_symbol_section(self.item.value)

    @property
    def ty(self):
        return self.obj.get_symbol_type(self.item.value)

    @property
    def flags(self):
        return self.obj.get_symbol_flags(self.item.value)


class SectionRefItem:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if other is None:
            return False
        return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.value)


class RelocationRefItem:
    def __init__(self, value):
        self.value = value


class RelocationRef:
    def __init__(self, obj, item):
        self.obj = obj
        self.item = item

    @property
    def symbol(self):
        return self.obj.get_relocation_symbol(self.item.value)

    @property
    def offset(self):
        return self.obj.get_relocation_offset(self.item.value)

    @property
    def ty(self):
        return self.obj.get_relocation_ty(self.item.value)

    @property
    def addend(self):
        return self.obj.get_relocation_addend(self.item.value)


class ObjFileMagic(Enum):
    Unknown = auto()
    ELF = auto()
    ELF_Relocatable = auto()
    ELF_Executable = auto()
    ELF_Shared_Object = auto()
    ELF_Core = auto()
    COFF_Object = auto()
    COFF_Import_Library = auto()
    PECOFF_Executable = auto()


def identify_magic(data):
    if data[0] == 0x7f and data[1:4].decode() == "ELF":
        data2msb = data[5] == 2
        hi = data[17]
        lo = data[16]
        if data2msb:
            hi, lo = lo, hi

        if hi == 0:
            if lo == 1:
                return ObjFileMagic.ELF_Relocatable
            elif lo == 2:
                return ObjFileMagic.ELF_Executable
            elif lo == 3:
                return ObjFileMagic.ELF_Shared_Object
            elif lo == 4:
                return ObjFileMagic.ELF_Core

        return ObjFileMagic.ELF
    if data[0] == 0:
        if data[1] == 0:
            return ObjFileMagic.COFF_Object
    if data[0] == 0x64:
        if data[1] == 0x86:
            return ObjFileMagic.COFF_Object

    return ObjFileMagic.Unknown


def parse_object_file(stream, file_ty=ObjFileMagic.Unknown):
    from rachetwrench.obj.coff import COFFObjectFile
    from rachetwrench.obj.elf import ELFObjectFile

    data = stream.read()

    if file_ty == ObjFileMagic.Unknown:
        file_ty = identify_magic(data)

    if file_ty == ObjFileMagic.COFF_Object:
        obj = COFFObjectFile()
        obj.parse_from(data)
        return obj

    if file_ty == ObjFileMagic.ELF_Relocatable:
        obj = ELFObjectFile()
        obj.parse_from(data)
        return obj

    raise ValueError("Invalid file type.")
