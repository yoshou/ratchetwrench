from rachetwrench.obj.utils import *
import struct
from rachetwrench.obj.objfile import *

Elf64_Half = UInt16L
Elf64_Word = UInt32L
Elf64_SWord = SInt32L
Elf64_XWord = UInt64L
Elf64_SXWord = SInt64L

Elf64_Addr = UInt64L
Elf64_Off = UInt64L
Elf64_Section = UInt16L
Elf64_Versym = Elf64_Half

Elf32_Half = UInt16L
Elf32_Word = UInt32L
Elf32_SWord = SInt32L
Elf32_XWord = UInt64L
Elf32_SXWord = SInt64L

Elf32_Addr = UInt32L
Elf32_Off = UInt32L
Elf32_Section = UInt16L
Elf32_Versym = Elf32_Half

EI_NIDENT = 16

ELFMAG0 = 0x70
ELFMAG1 = ord('e')
ELFMAG2 = ord('l')
ELFMAG3 = ord('f')

ELFCLASSNONE = 0
ELFCLASS32 = 1
ELFCLASS64 = 2

ELFDATANONE = 0
ELFDATA2LSB = 1
ELFDATA2MSB = 2

ET_NONE = 0
ET_REL = 1
ET_EXEC = 2
ET_DYN = 3
ET_CORE = 4
ET_LOPROC = 0xff00
ET_HIPROC = 0xffff

EM_NONE = 0
EM_M32 = 1
EM_SPARC = 2
EM_386 = 3
EM_68K = 4
EM_88K = 5
EM_860 = 7
EM_MIPS = 8
EM_ARM = 40
EM_X86_64 = 62
EM_RISCV = 243

EV_NONE = 0
EV_CURRENT = 1


SHT_NULL = 0                         # No associated section (inactive entry).
SHT_PROGBITS = 1                     # Program-defined contents.
SHT_SYMTAB = 2                       # Symbol table.
SHT_STRTAB = 3                       # String table.
SHT_RELA = 4                         # Relocation entries; explicit addends.
SHT_HASH = 5                         # Symbol hash table.
SHT_DYNAMIC = 6                      # Information for dynamic linking.
SHT_NOTE = 7                         # Information about the file.
SHT_NOBITS = 8                       # Data occupies no space in the file.
SHT_REL = 9                          # Relocation entries; no explicit addends.
SHT_SHLIB = 10                       # Reserved.
SHT_DYNSYM = 11                      # Symbol table.
SHT_INIT_ARRAY = 14                  # Pointers to initialization functions.
SHT_FINI_ARRAY = 15                  # Pointers to termination functions.
SHT_PREINIT_ARRAY = 16               # Pointers to pre-init functions.
SHT_GROUP = 17                       # Section group.
SHT_SYMTAB_SHNDX = 18                # Indices for SHN_XINDEX entries.

STB_LOCAL = 0  # Local symbol not visible outside obj file containing def
STB_GLOBAL = 1  # Global symbol visible to all object files being combined
STB_WEAK = 2   # Weak symbol like global but lower-precedence
STB_GNU_UNIQUE = 10
STB_LOOS = 10   # Lowest operating system-specific binding type
STB_HIOS = 12   # Highest operating system-specific binding type
STB_LOPROC = 13  # Lowest processor-specific binding type
STB_HIPROC = 15  # Highest processor-specific binding type


SHN_UNDEF = 0          # Undefined missing irrelevant or meaningless
SHN_LORESERVE = 0xff00  # Lowest reserved index
SHN_LOPROC = 0xff00    # Lowest processor-specific index
SHN_HIPROC = 0xff1f    # Highest processor-specific index
SHN_LOOS = 0xff20      # Lowest operating system-specific index
SHN_HIOS = 0xff3f      # Highest operating system-specific index
SHN_ABS = 0xfff1       # Symbol has absolute value; does not need relocation
SHN_COMMON = 0xfff2    # FORTRAN COMMON or C external global variables
SHN_XINDEX = 0xffff    # Mark that the index is >= SHN_LORESERVE
SHN_HIRESERVE = 0xffff  # Highest reserved index

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


# Section data should be writable during execution.
SHF_WRITE = 0x1

# Section occupies memory during program execution.
SHF_ALLOC = 0x2

# Section contains executable machine instructions.
SHF_EXECINSTR = 0x4

# The data in this section may be merged.
SHF_MERGE = 0x10


# elf x86_64 relocations
R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_PC32 = 2
R_X86_64_GOT32 = 3
R_X86_64_PLT32 = 4
R_X86_64_COPY = 5
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_GOTPCREL = 9
R_X86_64_32 = 10
R_X86_64_32S = 11
R_X86_64_16 = 12
R_X86_64_PC16 = 13
R_X86_64_8 = 14
R_X86_64_PC8 = 15
R_X86_64_DTPMOD64 = 16
R_X86_64_DTPOFF64 = 17
R_X86_64_TPOFF64 = 18
R_X86_64_TLSGD = 19
R_X86_64_TLSLD = 20
R_X86_64_DTPOFF32 = 21
R_X86_64_GOTTPOFF = 22
R_X86_64_TPOFF32 = 23
R_X86_64_PC64 = 24
R_X86_64_GOTOFF64 = 25
R_X86_64_GOTPC32 = 26
R_X86_64_GOT64 = 27
R_X86_64_GOTPCREL64 = 28
R_X86_64_GOTPC64 = 29
R_X86_64_GOTPLT64 = 30
R_X86_64_PLTOFF64 = 31
R_X86_64_SIZE32 = 32
R_X86_64_SIZE64 = 33
R_X86_64_GOTPC32_TLSDESC = 34
R_X86_64_TLSDESC_CALL = 35
R_X86_64_TLSDESC = 36
R_X86_64_IRELATIVE = 37
R_X86_64_GOTPCRELX = 41
R_X86_64_REX_GOTPCRELX = 42

# elf arm relocations

R_ARM_NONE = 0x00
R_ARM_PC24 = 0x01
R_ARM_ABS32 = 0x02
R_ARM_REL32 = 0x03
R_ARM_LDR_PC_G0 = 0x04
R_ARM_ABS16 = 0x05
R_ARM_ABS12 = 0x06
R_ARM_THM_ABS5 = 0x07
R_ARM_ABS8 = 0x08
R_ARM_SBREL32 = 0x09
R_ARM_THM_CALL = 0x0a
R_ARM_THM_PC8 = 0x0b
R_ARM_BREL_ADJ = 0x0c
R_ARM_TLS_DESC = 0x0d
R_ARM_THM_SWI8 = 0x0e
R_ARM_XPC25 = 0x0f
R_ARM_THM_XPC22 = 0x10
R_ARM_TLS_DTPMOD32 = 0x11
R_ARM_TLS_DTPOFF32 = 0x12
R_ARM_TLS_TPOFF32 = 0x13
R_ARM_COPY = 0x14
R_ARM_GLOB_DAT = 0x15
R_ARM_JUMP_SLOT = 0x16
R_ARM_RELATIVE = 0x17
R_ARM_GOTOFF32 = 0x18
R_ARM_BASE_PREL = 0x19
R_ARM_GOT_BREL = 0x1a
R_ARM_PLT32 = 0x1b
R_ARM_CALL = 0x1c
R_ARM_JUMP24 = 0x1d
R_ARM_THM_JUMP24 = 0x1e
R_ARM_BASE_ABS = 0x1f
R_ARM_ALU_PCREL_7_0 = 0x20
R_ARM_ALU_PCREL_15_8 = 0x21
R_ARM_ALU_PCREL_23_15 = 0x22
R_ARM_LDR_SBREL_11_0_NC = 0x23
R_ARM_ALU_SBREL_19_12_NC = 0x24
R_ARM_ALU_SBREL_27_20_CK = 0x25
R_ARM_TARGET1 = 0x26
R_ARM_SBREL31 = 0x27
R_ARM_V4BX = 0x28
R_ARM_TARGET2 = 0x29
R_ARM_PREL31 = 0x2a
R_ARM_MOVW_ABS_NC = 0x2b
R_ARM_MOVT_ABS = 0x2c
R_ARM_MOVW_PREL_NC = 0x2d
R_ARM_MOVT_PREL = 0x2e
R_ARM_THM_MOVW_ABS_NC = 0x2f
R_ARM_THM_MOVT_ABS = 0x30
R_ARM_THM_MOVW_PREL_NC = 0x31
R_ARM_THM_MOVT_PREL = 0x32
R_ARM_THM_JUMP19 = 0x33
R_ARM_THM_JUMP6 = 0x34
R_ARM_THM_ALU_PREL_11_0 = 0x35
R_ARM_THM_PC12 = 0x36
R_ARM_ABS32_NOI = 0x37
R_ARM_REL32_NOI = 0x38
R_ARM_ALU_PC_G0_NC = 0x39
R_ARM_ALU_PC_G0 = 0x3a
R_ARM_ALU_PC_G1_NC = 0x3b
R_ARM_ALU_PC_G1 = 0x3c
R_ARM_ALU_PC_G2 = 0x3d
R_ARM_LDR_PC_G1 = 0x3e
R_ARM_LDR_PC_G2 = 0x3f
R_ARM_LDRS_PC_G0 = 0x40
R_ARM_LDRS_PC_G1 = 0x41
R_ARM_LDRS_PC_G2 = 0x42
R_ARM_LDC_PC_G0 = 0x43
R_ARM_LDC_PC_G1 = 0x44
R_ARM_LDC_PC_G2 = 0x45
R_ARM_ALU_SB_G0_NC = 0x46
R_ARM_ALU_SB_G0 = 0x47
R_ARM_ALU_SB_G1_NC = 0x48
R_ARM_ALU_SB_G1 = 0x49
R_ARM_ALU_SB_G2 = 0x4a
R_ARM_LDR_SB_G0 = 0x4b
R_ARM_LDR_SB_G1 = 0x4c
R_ARM_LDR_SB_G2 = 0x4d
R_ARM_LDRS_SB_G0 = 0x4e
R_ARM_LDRS_SB_G1 = 0x4f
R_ARM_LDRS_SB_G2 = 0x50
R_ARM_LDC_SB_G0 = 0x51
R_ARM_LDC_SB_G1 = 0x52
R_ARM_LDC_SB_G2 = 0x53
R_ARM_MOVW_BREL_NC = 0x54
R_ARM_MOVT_BREL = 0x55
R_ARM_MOVW_BREL = 0x56
R_ARM_THM_MOVW_BREL_NC = 0x57
R_ARM_THM_MOVT_BREL = 0x58
R_ARM_THM_MOVW_BREL = 0x59
R_ARM_TLS_GOTDESC = 0x5a
R_ARM_TLS_CALL = 0x5b
R_ARM_TLS_DESCSEQ = 0x5c
R_ARM_THM_TLS_CALL = 0x5d
R_ARM_PLT32_ABS = 0x5e
R_ARM_GOT_ABS = 0x5f
R_ARM_GOT_PREL = 0x60
R_ARM_GOT_BREL12 = 0x61
R_ARM_GOTOFF12 = 0x62
R_ARM_GOTRELAX = 0x63
R_ARM_GNU_VTENTRY = 0x64
R_ARM_GNU_VTINHERIT = 0x65
R_ARM_THM_JUMP11 = 0x66
R_ARM_THM_JUMP8 = 0x67
R_ARM_TLS_GD32 = 0x68
R_ARM_TLS_LDM32 = 0x69
R_ARM_TLS_LDO32 = 0x6a
R_ARM_TLS_IE32 = 0x6b
R_ARM_TLS_LE32 = 0x6c
R_ARM_TLS_LDO12 = 0x6d
R_ARM_TLS_LE12 = 0x6e
R_ARM_TLS_IE12GP = 0x6f
R_ARM_PRIVATE_0 = 0x70
R_ARM_PRIVATE_1 = 0x71
R_ARM_PRIVATE_2 = 0x72
R_ARM_PRIVATE_3 = 0x73
R_ARM_PRIVATE_4 = 0x74
R_ARM_PRIVATE_5 = 0x75
R_ARM_PRIVATE_6 = 0x76
R_ARM_PRIVATE_7 = 0x77
R_ARM_PRIVATE_8 = 0x78
R_ARM_PRIVATE_9 = 0x79
R_ARM_PRIVATE_10 = 0x7a
R_ARM_PRIVATE_11 = 0x7b
R_ARM_PRIVATE_12 = 0x7c
R_ARM_PRIVATE_13 = 0x7d
R_ARM_PRIVATE_14 = 0x7e
R_ARM_PRIVATE_15 = 0x7f
R_ARM_ME_TOO = 0x80
R_ARM_THM_TLS_DESCSEQ16 = 0x81
R_ARM_THM_TLS_DESCSEQ32 = 0x82
R_ARM_THM_BF16 = 0x88
R_ARM_THM_BF12 = 0x89
R_ARM_THM_BF18 = 0x8a
R_ARM_IRELATIVE = 0xa0


def create_elf_ident(elfclass, data, version):
    return [ELFMAG0, ELFMAG1, ELFMAG2, ELFMAG3, elfclass, data, version] + [0] * 9


class Elf32_Ehdr(Struct):
    e_ident = ArrayField(UInt8L(0), EI_NIDENT, create_elf_ident(
        ELFCLASS32, ELFDATA2LSB, EV_CURRENT))
    e_type = Elf32_Half(0)
    e_machine = Elf32_Half(0)
    e_version = Elf32_Word(0)
    e_entry = Elf32_Addr(0)
    e_phoff = Elf32_Off(0)
    e_shoff = Elf32_Off(0)
    e_flags = Elf32_Word(0)
    e_ehsize = Elf32_Half(0)
    e_phentsize = Elf32_Half(0)
    e_phnum = Elf32_Half(0)
    e_shentsize = Elf32_Half(0)
    e_shnum = Elf32_Half(0)
    e_shstrndx = Elf32_Half(0)


class Elf64_Ehdr(Struct):
    e_ident = ArrayField(UInt8L(0), EI_NIDENT, create_elf_ident(
        ELFCLASS64, ELFDATA2LSB, EV_CURRENT))
    e_type = Elf64_Half(0)
    e_machine = Elf64_Half(0)
    e_version = Elf64_Word(0)
    e_entry = Elf64_Addr(0)
    e_phoff = Elf64_Off(0)
    e_shoff = Elf64_Off(0)
    e_flags = Elf64_Word(0)
    e_ehsize = Elf64_Half(0)
    e_phentsize = Elf64_Half(0)
    e_phnum = Elf64_Half(0)
    e_shentsize = Elf64_Half(0)
    e_shnum = Elf64_Half(0)
    e_shstrndx = Elf64_Half(0)


class Elf32_Shdr(Struct):
    sh_name = Elf32_Word(0)
    sh_type = Elf32_Word(0)
    sh_flags = Elf32_Word(0)
    sh_addr = Elf32_Addr(0)
    sh_offset = Elf32_Off(0)
    sh_size = Elf32_Word(0)
    sh_link = Elf32_Word(0)
    sh_info = Elf32_Word(0)
    sh_addralign = Elf32_Word(0)
    sh_entsize = Elf32_Word(0)


class Elf64_Shdr(Struct):
    sh_name = Elf64_Word(0)
    sh_type = Elf64_Word(0)
    sh_flags = Elf64_XWord(0)
    sh_addr = Elf64_Addr(0)
    sh_offset = Elf64_Off(0)
    sh_size = Elf64_XWord(0)
    sh_link = Elf64_Word(0)
    sh_info = Elf64_Word(0)
    sh_addralign = Elf64_XWord(0)
    sh_entsize = Elf64_XWord(0)


class Elf32_Phdr(Struct):
    p_type = Elf32_Word(0)
    p_flags = Elf32_Word(0)
    p_offset = Elf32_Off(0)
    p_vaddr = Elf32_Addr(0)
    p_paddr = Elf32_Addr(0)
    p_filesz = Elf32_XWord(0)
    p_memsz = Elf32_XWord(0)
    p_align = Elf32_XWord(0)


class Elf64_Phdr(Struct):
    p_type = Elf64_Word(0)
    p_flags = Elf64_Word(0)
    p_offset = Elf64_Off(0)
    p_vaddr = Elf64_Addr(0)
    p_paddr = Elf64_Addr(0)
    p_filesz = Elf64_XWord(0)
    p_memsz = Elf64_XWord(0)
    p_align = Elf64_XWord(0)


class Elf32_Sym(Struct):
    st_name = Elf32_Word(0)
    st_value = Elf32_Addr(0)
    st_size = Elf32_Word(0)
    st_info = UInt8L(0)
    st_other = UInt8L(0)
    st_shndx = Elf32_Half(0)

    @property
    def binding(self):
        return self.st_info.value >> 4

    @property
    def ty(self):
        return self.st_info.value & 0xF


class Elf64_Sym(Struct):
    st_name = Elf64_Word(0)
    st_info = UInt8L(0)
    st_other = UInt8L(0)
    st_shndx = Elf64_Half(0)
    st_value = Elf64_Addr(0)
    st_size = Elf64_XWord(0)

    @property
    def binding(self):
        return self.st_info.value >> 4

    @property
    def ty(self):
        return self.st_info.value & 0xF


class Elf32_Rel(Struct):
    r_offset = Elf32_Addr(0)
    r_info = Elf32_Word(0)

    @property
    def symbol(self):
        return (self.r_info.value >> 8) & 0xFFFFFF

    @property
    def type(self):
        return self.r_info.value & 0xFF


class Elf64_Rel(Struct):
    r_offset = Elf64_Addr(0)
    r_info = Elf64_XWord(0)

    @property
    def symbol(self):
        return (self.r_info.value >> 32) & 0xFFFFFFFF

    @property
    def type(self):
        return self.r_info.value & 0xFFFFFFFF


class Elf32_Rela(Struct):
    r_offset = Elf32_Addr(0)
    r_info = Elf32_Word(0)
    r_addend = Elf32_SWord(0)

    @property
    def symbol(self):
        return (self.r_info.value >> 8) & 0xFFFFFF

    @property
    def type(self):
        return self.r_info.value & 0xFF


class Elf64_Rela(Struct):
    r_offset = Elf64_Addr(0)
    r_info = Elf64_XWord(0)
    r_addend = Elf64_SXWord(0)

    @property
    def symbol(self):
        return (self.r_info.value >> 32) & 0xFFFFFFFF

    @property
    def type(self):
        return self.r_info.value & 0xFFFFFFFF


from io import BytesIO


class ElfFile:
    def __init__(self):
        pass

    def deserialize(self, data):
        self.data = data

        is_64bit = data[4] == ELFCLASS64

        self.Elf_Ehdr = Elf64_Ehdr if is_64bit else Elf32_Ehdr
        self.Elf_Shdr = Elf64_Shdr if is_64bit else Elf32_Shdr
        self.Elf_Sym = Elf64_Sym if is_64bit else Elf32_Sym
        self.Elf_Rel = Elf64_Rel if is_64bit else Elf32_Rel
        self.Elf_Rela = Elf64_Rela if is_64bit else Elf32_Rela

        with BytesIO(self.data) as s:
            self.header = self.Elf_Ehdr()
            self.header.deserialize(s)

    @property
    def sections(self):
        section_table_offset = self.header.e_shoff.value

        if section_table_offset == 0:
            return []

        assert(self.header.e_shentsize.value == self.Elf_Shdr().size)

        filesize = len(self.data)

        with BytesIO(self.data[section_table_offset:]) as s:
            for i in range(self.header.e_shnum.value):
                sec = self.Elf_Shdr()
                sec.deserialize(s)

                yield sec

    def symbols(self, section):
        offset = section.sh_offset.value
        size = section.sh_size.value

        entry_size = self.Elf_Sym().size

        assert(size % entry_size == 0)

        with BytesIO(self.data[offset:]) as s:
            for i in range(0, size, entry_size):
                sym = self.Elf_Sym()
                sym.deserialize(s)

                yield sym

    def rels(self, section):
        if section.sh_type.value & SHT_REL != SHT_REL:
            return []

        offset = section.sh_offset.value
        size = section.sh_size.value

        entry_size = self.Elf_Rel().size

        assert(size % entry_size == 0)

        with BytesIO(self.data[offset:]) as s:
            for i in range(0, size, entry_size):
                sym = self.Elf_Rel()
                sym.deserialize(s)

                yield sym

    def relas(self, section):
        if section.sh_type.value & SHT_RELA != SHT_RELA:
            return []

        offset = section.sh_offset.value
        size = section.sh_size.value

        entry_size = self.Elf_Rela().size

        assert(size % entry_size == 0)

        with BytesIO(self.data[offset:]) as s:
            for i in range(0, size, entry_size):
                sym = self.Elf_Rela()
                sym.deserialize(s)

                yield sym

    def relocations(self, section: Elf32_Shdr):
        if section.sh_type.value & SHT_REL == SHT_REL:
            return self.rels(section)
        elif section.sh_type.value & SHT_RELA == SHT_RELA:
            return self.relas(section)

        return []


class ELFSectionRefItem(SectionRefItem):
    def __init__(self, value):
        super().__init__(value)

    def __eq__(self, other):
        if other is None:
            return False
        return self.value.sh_name.value == other.value.sh_name.value

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.value.sh_name.value)


class ELFObjectFile(ObjectFile):
    def __init__(self):
        super().__init__()

    @property
    def arch(self):
        is_64bit = self.data[4] == ELFCLASS64

        from rachetwrench.codegen.spec import ArchType
        if self.header.e_machine.value == EM_ARM:
            return ArchType.ARM
        elif self.header.e_machine.value == EM_X86_64:
            return ArchType.X86_64
        elif self.header.e_machine.value == EM_RISCV:
            if is_64bit:
                return ArchType.RISCV64
            else:
                return ArchType.RISCV32

        raise ValueError()

    @property
    def is_elf(self):
        return True

    def parse_from(self, data):
        self.data = bytes(data)

        self.elf_file = ElfFile()
        self.elf_file.deserialize(self.data)

    @property
    def header(self):
        return self.elf_file.header

    @property
    def symbols(self):
        dot_symtab_section = None
        for sec in self.elf_file.sections:
            if sec.sh_type.value == SHT_SYMTAB:
                dot_symtab_section = sec
                break

        assert(dot_symtab_section)

        return [SymbolRef(self, SymbolRefItem((symbol, dot_symtab_section))) for symbol in self.elf_file.symbols(dot_symtab_section)]

    @property
    def sections(self):
        dot_symtab_section = None
        for sec in self.elf_file.sections:
            if sec.sh_type.value == SHT_SYMTAB:
                dot_symtab_section = sec
                break

        assert(dot_symtab_section)

        return [SectionRef(self, ELFSectionRefItem(sec)) for sec in self.elf_file.sections]

    def get_section_relocations(self, section):
        return [RelocationRef(self, RelocationRefItem((reloc, section))) for reloc in self.elf_file.relocations(section)]

    def get_section_contents(self, section):
        if section.sh_type.value == SHT_NOBITS:
            return bytearray()

        offset = section.sh_offset.value
        size = section.sh_size.value
        return bytearray(self.data[offset:(offset+size)])

    def get_section_alignment(self, section):
        return section.sh_addralign.value

    def get_section_address(self, section):
        return section.sh_addr.value

    def get_section_size(self, section):
        return section.sh_size.value

    def get_section_name(self, section):
        strtab = self.sections[self.header.e_shstrndx.value].item.value

        offset = strtab.sh_offset.value
        size = strtab.sh_size.value

        with BytesIO(self.data[offset:]) as s:
            strtab_data = s.read(size)

        def get_null_terminated_str(data, offset):
            start = offset
            end = offset
            while data[end] != 0:
                end += 1

            return str(data[start:end], "utf8")

        return get_null_terminated_str(strtab_data, section.sh_name.value)

    def get_section_flags(self, section):
        return section.sh_flags.value

    def get_section_type(self, section):
        return section.sh_type.value

    def get_section_is_text(self, section):
        return section.sh_flags.value & SHF_EXECINSTR == SHF_EXECINSTR

    def get_section_is_virtual(self, section):
        return section.sh_type.value == SHT_NOBITS == SHT_NOBITS

    def get_relocated_section(self, section):
        if section.sh_type.value not in [SHT_REL, SHT_RELA]:
            return None

        sec = list(self.elf_file.sections)[section.sh_info.value]
        return SectionRef(self, ELFSectionRefItem(sec))

    def get_symbol_name(self, symbol_ref):
        symbol, section = symbol_ref
        strtab = self.sections[self.header.e_shstrndx.value].item.value

        offset = strtab.sh_offset.value
        size = strtab.sh_size.value

        with BytesIO(self.data[offset:]) as s:
            strtab_data = s.read(size)

        def get_null_terminated_str(data, offset):
            start = offset
            end = offset
            while data[end] != 0:
                end += 1

            return str(data[start:end], "utf8")

        return get_null_terminated_str(strtab_data, symbol.st_name.value)

    def get_symbol_binding(self, symbol_ref):
        symbol, section = symbol_ref
        return symbol.binding

    def get_symbol_section(self, symbol_ref):
        symbol, section = symbol_ref

        if symbol.st_shndx.value == 0:
            return None

        section = list(self.elf_file.sections)[symbol.st_shndx.value]
        return SectionRef(self, ELFSectionRefItem(section))

    def get_symbol_type(self, symbol_ref):
        symbol, section = symbol_ref

        ty = symbol.ty
        if ty == STT_NOTYPE:
            return SymbolType.Unknown
        elif ty == STT_SECTION:
            return SymbolType.Debug
        elif ty == STT_FILE:
            return SymbolType.File
        elif ty == STT_FUNC:
            return SymbolType.Function
        elif ty in [STT_OBJECT, STT_COMMON, STT_TLS]:
            return SymbolType.Data

        return SymbolType.Other

    def get_symbol_flags(self, symbol_ref):
        symbol, section = symbol_ref

        result = SymbolFlags.Non

        binding = self.get_symbol_binding(symbol_ref)

        if binding != STB_LOCAL:
            result |= SymbolFlags.Global

        if symbol.st_shndx.value == SHN_ABS:
            result |= SymbolFlags.Absolute

        if symbol.st_shndx.value == SHN_UNDEF:
            result |= SymbolFlags.Undefined

        return result

    def get_symbol_address(self, symbol_ref):
        symbol, section = symbol_ref

        result = self.get_symbol_value(symbol_ref)

        if symbol.st_shndx.value in [SHN_ABS, SHN_UNDEF, SHN_COMMON]:
            return result

        section = list(self.elf_file.sections)[symbol.st_shndx.value]

        if self.header.e_type.value == ET_REL:
            result += section.sh_addr.value

        return result

    def get_symbol_value(self, symbol_ref):
        symbol, section = symbol_ref

        result = symbol.st_value.value
        if symbol.st_shndx.value == SHN_ABS:
            return result

        if self.header.e_machine.value == EM_ARM and symbol.ty == STT_FUNC:
            return result & ~1

        return result

    def get_relocation_symbol(self, reloc_ref):
        reloc, reloc_sec = reloc_ref

        dot_symtab_section = None
        for sec in self.elf_file.sections:
            if sec.sh_type.value == SHT_SYMTAB:
                dot_symtab_section = sec
                break

        assert(dot_symtab_section)

        symtab_sec = self.sections[reloc_sec.sh_link.value]

        symbol = list(self.elf_file.symbols(
            symtab_sec.item.value))[reloc.symbol]

        return SymbolRef(self, SymbolRefItem((symbol, dot_symtab_section)))

    def get_relocation_offset(self, reloc_ref):
        reloc, reloc_sec = reloc_ref

        return reloc.r_offset.value

    def get_relocation_ty(self, reloc_ref):
        reloc, reloc_sec = reloc_ref

        return reloc.type

    def get_relocation_addend(self, reloc_ref):
        reloc, reloc_sec = reloc_ref

        if isinstance(reloc, Elf32_Rel):
            return 0

        return reloc.r_addend.value
