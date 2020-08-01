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
    MCDataFragment, MCAlignFragment,
    MCRelaxableFragment,
    MCCodeEmitter,
    SectionKind)

from codegen.mc import ELFSymbolType, ELFSymbolBinding, ELFSymbolVisibility

from codegen.assembler import MCAsmBackend, MCAssembler, MCAsmLayout, evaluate_expr_as_constant

from io import BytesIO


class MCSectionELF(MCSection):
    def __init__(self, name, ty, flags, symbol, entry_size=0, associated=None, alignment=4):
        super().__init__()

        self.name = name
        self.ty = ty
        self.flags = flags
        self.entry_size = entry_size
        self.associated_section = associated
        self.symbol = symbol
        self.alignment = alignment

    def __hash__(self):
        return hash((self.name,))

    def __eq__(self, other):
        if not isinstance(other, MCSectionELF):
            return False

        return self.name == other.name


def create_elf_section(name, ty, flags, entry_size=0, associated=None, alignment=4):
    symbol = MCSymbolELF(name, False)
    symbol.binding = ELFSymbolBinding.STB_LOCAL
    symbol.ty = ELFSymbolType.STT_SECTION

    section = MCSectionELF(name, ty, flags, symbol,
                           entry_size, associated, alignment)

    fragment = MCDataFragment()
    section.add_fragment(fragment)
    symbol.fragment = fragment

    return section


def to_bytes(value, size, order):
    value = value & ((1 << (size * 8)) - 1)
    return value.to_bytes(
        size, byteorder=order, signed=False)


class ELFObjectStream(MCObjectStream):
    def __init__(self, context: MCContext, backend: MCAsmBackend, writer: MCObjectWriter, emitter: MCCodeEmitter):
        super().__init__(context, backend, writer, emitter)

        self.section_stack = [None]

    def init_sections(self):
        context = self.context

        self.switch_section(context.obj_file_info.text_section)

    def emit_newline(self):
        pass

    def get_or_create_data_fragment(self):
        if len(self.current_section.fragments) > 0:
            if isinstance(self.current_section.fragments[-1], MCDataFragment):
                return self.current_section.fragments[-1]

        fragment = MCDataFragment()
        self.current_section.add_fragment(fragment)
        return fragment

    def get_data_fragment(self):
        if len(self.current_section.fragments) > 0:
            if isinstance(self.current_section.fragments[-1], MCDataFragment):
                return self.current_section.fragments[-1]

        return None

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

    def emit_symbol_attrib(self, symbol: MCSymbol, attrib: MCSymbolAttribute):
        def merge_symbol_type(ty1, ty2):
            if ty1 == ELFSymbolType.STT_TLS:
                if ty2 in [ELFSymbolType.STT_OBJECT, ELFSymbolType.STT_NOTYPE, ELFSymbolType.STT_FUNC]:
                    return ELFSymbolType.STT_TLS

            return ty2

        if attrib == MCSymbolAttribute.ELF_TypeFunction:
            symbol.ty = merge_symbol_type(symbol.ty, ELFSymbolType.STT_FUNC)
        elif attrib == MCSymbolAttribute.Global:
            symbol.binding = ELFSymbolBinding.STB_GLOBAL
        elif attrib == MCSymbolAttribute.ELF_TypeObject:
            symbol.ty = merge_symbol_type(symbol.ty, ELFSymbolType.STT_OBJECT)
        else:
            raise NotImplementedError()

    def emit_label(self, symbol: MCSymbol):
        self.assembler.register_symbol(symbol)

        fragment = self.get_or_create_data_fragment()
        if fragment:
            symbol.fragment = fragment
            symbol.offset = len(symbol.fragment.contents)

        if self.current_section.flags & SHF_TLS:
            symbol.ty = ELFSymbolType.STT_TLS

    def emit_bytes(self, value: bytearray):
        fragment = self.get_or_create_data_fragment()
        fragment.contents.extend(value)

    def emit_int_value(self, value: int, size: int):
        order = 'little'
        fragment = self.get_or_create_data_fragment()

        fragment.contents.extend(to_bytes(value, size, order))

    def emit_zeros(self, size):
        fragment = self.get_or_create_data_fragment()
        fragment.contents.extend(bytearray(size))

    def emit_value(self, value: MCExpr, size: int):
        from codegen.mc import get_fixup_kind_for_size

        fragment = self.get_or_create_data_fragment()

        fragment.fixups.append(
            MCFixup(len(fragment.contents), value, get_fixup_kind_for_size(size, False)))
        fragment.contents.extend(bytearray(size))

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
            self.assembler.register_symbol(section.symbol)

    def emit_coff_symbol_storage_class(self, symbol, symbol_class):
        symbol.symbol_class = symbol_class

    def emit_coff_symbol_type(self, symbol, symbol_type):
        symbol.symbol_type = symbol_type

    def emit_elf_size(self, symbol, size):
        symbol.size = size


class ELFObjectTargetWriter(MCObjectTargetWriter):
    def __init__(self):
        super().__init__()


class X64ELFObjectWriter(ELFObjectTargetWriter):
    def __init__(self):
        super().__init__()

        self.os_abi = ELFOSABI_NONE
        self.abi_version = 0
        self.emachine = EM_X86_64
        self.flags = 0x0
        self.is_64bit = True
        self.has_relocation_addend = True

    def get_reloc_type64(self, context, modifier, ty, is_pcrel, kind):
        if modifier in [MCVariantKind.Non]:
            if ty == X64RelType.RT64_NONE:
                return R_X86_64_NONE
            elif ty == X64RelType.RT64_64:
                return R_X86_64_PC64 if is_pcrel else R_X86_64_64
            elif ty == X64RelType.RT64_32:
                return R_X86_64_PC32 if is_pcrel else R_X86_64_32
            elif ty == X64RelType.RT64_16:
                return R_X86_64_PC16 if is_pcrel else R_X86_64_16
            elif ty == X64RelType.RT64_8:
                return R_X86_64_PC8 if is_pcrel else R_X86_64_8
            else:
                raise NotImplementedError()
        elif modifier == MCVariantKind.TLSGD:
            return R_X86_64_TLSGD
        elif modifier == MCVariantKind.PLT:
            return R_X86_64_PLT32
        else:
            raise NotImplementedError()

    def get_reloc_type(self, context: MCContext, target: MCValue, fixup: MCFixup, is_pcrel):
        modifier = target.access_variant
        kind = fixup.kind

        from codegen.x64_asm_printer import X64FixupKind

        def get_type64(kind: MCFixupKind, modifier, is_pcrel):
            if kind == MCFixupKind.Noop:
                result = X64RelType.RT64_NONE
            elif kind == MCFixupKind.Data_8:
                result = X64RelType.RT64_64
            elif kind in [MCFixupKind.Data_4, MCFixupKind.PCRel_4]:
                result = X64RelType.RT64_32
            elif kind in [MCFixupKind.Data_2, MCFixupKind.PCRel_2]:
                result = X64RelType.RT64_16
            elif kind in [MCFixupKind.Data_1, MCFixupKind.PCRel_1]:
                result = X64RelType.RT64_8
            elif kind == X64FixupKind.Reloc_RIPRel_4:
                result = X64RelType.RT64_32
            else:
                raise NotImplementedError()

            return (result, modifier, is_pcrel)

        ty, modifier, is_pcrel = get_type64(kind, modifier, is_pcrel)

        return self.get_reloc_type64(context, modifier, ty, is_pcrel, kind)


class ARMELFObjectWriter(ELFObjectTargetWriter):
    def __init__(self):
        super().__init__()

        self.os_abi = ELFOSABI_NONE
        self.abi_version = 5
        self.emachine = EM_ARM
        self.flags = 0x5000000
        self.is_64bit = False
        self.has_relocation_addend = False

    def get_reloc_type(self, context: MCContext, target: MCValue, fixup: MCFixup, is_pcrel):
        modifier = target.access_variant
        kind = fixup.kind

        from codegen.arm_asm_printer import ARMFixupKind

        if is_pcrel:
            if kind in [ARMFixupKind.ARM_COND_BRANCH, ARMFixupKind.ARM_UNCOND_BRANCH]:
                return R_ARM_JUMP24
            elif kind in [ARMFixupKind.ARM_UNCOND_BL]:
                return R_ARM_CALL
            elif kind in [ARMFixupKind.ARM_MOVW_LO16]:
                return R_ARM_MOVW_PREL_NC
            elif kind in [ARMFixupKind.ARM_MOVT_HI16]:
                return R_ARM_MOVT_PREL
            else:
                raise NotImplementedError()

        # absolute
        if kind in [ARMFixupKind.ARM_COND_BRANCH, ARMFixupKind.ARM_UNCOND_BRANCH]:
            return R_ARM_JUMP24
        elif kind in [ARMFixupKind.ARM_UNCOND_BL]:
            return R_ARM_CALL
        elif kind in [ARMFixupKind.ARM_MOVW_LO16]:
            return R_ARM_MOVW_ABS_NC
        elif kind in [ARMFixupKind.ARM_MOVT_HI16]:
            return R_ARM_MOVT_ABS
        elif kind == MCFixupKind.Data_1:
            if modifier == MCVariantKind.Non:
                return R_ARM_ABS8
            else:
                raise NotImplementedError()
        elif kind == MCFixupKind.Data_2:
            if modifier == MCVariantKind.Non:
                return R_ARM_ABS16
            else:
                raise NotImplementedError()
        elif kind == MCFixupKind.Data_4:
            if modifier == MCVariantKind.Non:
                return R_ARM_NONE
            elif modifier == MCVariantKind.TLSGD:
                return R_ARM_TLS_GD32
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


class AArch64ELFObjectWriter(ELFObjectTargetWriter):
    def __init__(self):
        super().__init__()

        self.os_abi = ELFOSABI_NONE
        self.abi_version = 5
        self.emachine = EM_AARCH64
        self.flags = 0x5000000
        self.is_64bit = True
        self.has_relocation_addend = False

    def get_reloc_type(self, context: MCContext, target: MCValue, fixup: MCFixup, is_pcrel):
        ref_kind = target.ref_kind
        kind = fixup.kind

        from codegen.aarch64_asm_printer import AArch64FixupKind, AArch64MCExprVarKind

        if is_pcrel:
            if kind in [AArch64FixupKind.AArch64_PCREL_ADRP_IMM21]:
                if ref_kind & AArch64MCExprVarKind.ABS:
                    return R_AARCH64_ADR_PREL_PG_HI21
                elif ref_kind & AArch64MCExprVarKind.TLSDESC:
                    return R_AARCH64_TLSDESC_ADR_PAGE21
            elif kind in [AArch64FixupKind.AArch64_PCREL_CALL26]:
                return R_AARCH64_CALL26
            else:
                raise NotImplementedError()

        # absolute
        if kind in [AArch64FixupKind.AArch64_ADD_IMM12]:
            if ref_kind & AArch64MCExprVarKind.ABS:
                return R_AARCH64_ADD_ABS_LO12_NC
            elif ref_kind & AArch64MCExprVarKind.TLSDESC:
                return R_AARCH64_TLSDESC_ADD_LO12
        elif kind in [AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED8]:
            if ref_kind & AArch64MCExprVarKind.ABS:
                return R_AARCH64_ADD_ABS_LO12_NC
            elif ref_kind & AArch64MCExprVarKind.TLSDESC:
                return R_AARCH64_TLSDESC_LD64_LO12
        elif kind in [AArch64FixupKind.AArch64_TLSDESC_CALL]:
            return R_AARCH64_TLSDESC_CALL

        raise NotImplementedError()


class RISCVELFObjectWriter(ELFObjectTargetWriter):
    def __init__(self):
        super().__init__()

        self.os_abi = ELFOSABI_NONE
        self.abi_version = 0
        self.emachine = EM_RISCV
        self.flags = EF_RISCV_FLOAT_ABI_DOUBLE
        self.is_64bit = False
        self.has_relocation_addend = False

    def get_reloc_type(self, context: MCContext, target: MCValue, fixup: MCFixup, is_pcrel):
        modifier = target.access_variant
        kind = fixup.kind

        from codegen.riscv_asm_printer import RISCVFixupKind

        if is_pcrel:
            if kind in [RISCVFixupKind.RISCV_BRANCH]:
                return R_RISCV_BRANCH
            elif kind in [RISCVFixupKind.RISCV_CALL]:
                return R_RISCV_CALL
            elif kind in [RISCVFixupKind.RISCV_JAL]:
                return R_RISCV_JAL
            elif kind in [RISCVFixupKind.RISCV_PCREL_HI20]:
                return R_RISCV_PCREL_HI20
            elif kind in [RISCVFixupKind.RISCV_PCREL_LO12_I]:
                return R_RISCV_PCREL_LO12_I
            elif kind in [RISCVFixupKind.RISCV_PCREL_LO12_S]:
                return R_RISCV_PCREL_LO12_S
            elif kind in [RISCVFixupKind.RISCV_TLS_GD_HI20]:
                return R_RISCV_TLS_GD_HI20
            else:
                raise NotImplementedError()

        # absolute

        if kind in [RISCVFixupKind.RISCV_HI20]:
            return R_RISCV_HI20
        elif kind in [RISCVFixupKind.RISCV_LO12_I]:
            return R_RISCV_LO12_I
        elif kind in [RISCVFixupKind.RISCV_LO12_S]:
            return R_RISCV_LO12_S

        raise NotImplementedError()


ELF_MAGIC = bytes([0x7f, ord('E'), ord('L'), ord('F')])

ELFCLASSNONE = 0
ELFCLASS32 = 1  # 32-bit object file
ELFCLASS64 = 2  # 64-bit object file

ELFDATANONE = 0
ELFDATA2LSB = 1  # Little-endian object file
ELFDATA2MSB = 2  # Big-endian object file

EV_CURRENT = 1

# OS ABI
ELFOSABI_NONE = 0           # UNIX System V ABI
ELFOSABI_HPUX = 1           # HP-UX operating system
ELFOSABI_NETBSD = 2         # NetBSD
ELFOSABI_GNU = 3            # GNU/Linux
ELFOSABI_LINUX = 3          # Historical alias for ELFOSABI_GNU.
ELFOSABI_HURD = 4           # GNU/Hurd
ELFOSABI_SOLARIS = 6        # Solaris
ELFOSABI_AIX = 7            # AIX
ELFOSABI_IRIX = 8           # IRIX
ELFOSABI_FREEBSD = 9        # FreeBSD
ELFOSABI_TRU64 = 10         # TRU64 UNIX
ELFOSABI_MODESTO = 11       # Novell Modesto
ELFOSABI_OPENBSD = 12       # OpenBSD
ELFOSABI_OPENVMS = 13       # OpenVMS
ELFOSABI_NSK = 14           # Hewlett-Packard Non-Stop Kernel
ELFOSABI_AROS = 15          # AROS
ELFOSABI_FENIXOS = 16       # FenixOS
ELFOSABI_CLOUDABI = 17      # Nuxi CloudABI
ELFOSABI_FIRST_ARCH = 64    # First architecture-specific OS ABI
ELFOSABI_AMDGPU_HSA = 64    # AMD HSA runtime
ELFOSABI_AMDGPU_PAL = 65    # AMD PAL runtime
ELFOSABI_AMDGPU_MESA3D = 66  # AMD GCN GPUs (GFX6+) for MESA runtime
ELFOSABI_ARM = 97           # ARM
ELFOSABI_C6000_ELFABI = 64  # Bare-metal TMS320C6000
ELFOSABI_C6000_LINUX = 65   # Linux TMS320C6000
ELFOSABI_STANDALONE = 255   # Standalone (embedded) application

EI_MAG0 = 0       # File identification index.
EI_MAG1 = 1       # File identification index.
EI_MAG2 = 2       # File identification index.
EI_MAG3 = 3       # File identification index.
EI_CLASS = 4      # File class.
EI_DATA = 5       # Data encoding.
EI_VERSION = 6    # File version.
EI_OSABI = 7      # OS/ABI identification.
EI_ABIVERSION = 8  # ABI version.
EI_PAD = 9        # Start of padding bytes.
EI_NIDENT = 16     # Number of bytes in e_ident.


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

# Section data should be writable during execution.
SHF_WRITE = 0x1
# Section occupies memory during program execution.
SHF_ALLOC = 0x2
# Section contains executable machine instructions.
SHF_EXECINSTR = 0x4
# The data in this section may be merged.
SHF_MERGE = 0x10
# The data in this section is null-terminated strings.
SHF_STRINGS = 0x20
# A field in this section holds a section header table index.
SHF_INFO_LINK = 0x40
# Adds special ordering requirements for link editors.
SHF_LINK_ORDER = 0x80
# This section requires special OS-specific processing to avoid incorrect
# behavior.
SHF_OS_NONCONFORMING = 0x100
# This section is a member of a section group.
SHF_GROUP = 0x200
# This section holds Thread-Local Storage.
SHF_TLS = 0x400
# Identifies a section containing compressed data.
SHF_COMPRESSED = 0x800

ET_NONE = 0        # No file type
ET_REL = 1         # Relocatable file
ET_EXEC = 2        # Executable file
ET_DYN = 3         # Shared object file
ET_CORE = 4        # Core file
ET_LOPROC = 0xff00  # Beginning of processor-specific codes
ET_HIPROC = 0xffff  # Processor-specific


EM_NONE = 0           # No machine
EM_M32 = 1            # AT&T WE 32100
EM_SPARC = 2          # SPARC
EM_386 = 3            # Intel 386
EM_68K = 4            # Motorola 68000
EM_88K = 5            # Motorola 88000
EM_IAMCU = 6          # Intel MCU
EM_860 = 7            # Intel 80860
EM_MIPS = 8           # MIPS R3000
EM_S370 = 9           # IBM System/370
EM_MIPS_RS3_LE = 10   # MIPS RS3000 Little-endian
EM_PARISC = 15        # Hewlett-Packard PA-RISC
EM_VPP500 = 17        # Fujitsu VPP500
EM_SPARC32PLUS = 18   # Enhanced instruction set SPARC
EM_960 = 19           # Intel 80960
EM_PPC = 20           # PowerPC
EM_PPC64 = 21         # PowerPC64
EM_S390 = 22          # IBM System/390
EM_SPU = 23           # IBM SPU/SPC
EM_V800 = 36          # NEC V800
EM_FR20 = 37          # Fujitsu FR20
EM_RH32 = 38          # TRW RH-32
EM_RCE = 39           # Motorola RCE
EM_ARM = 40           # ARM
EM_ALPHA = 41         # DEC Alpha
EM_SH = 42            # Hitachi SH
EM_SPARCV9 = 43       # SPARC V9
EM_TRICORE = 44       # Siemens TriCore
EM_ARC = 45           # Argonaut RISC Core
EM_H8_300 = 46        # Hitachi H8/300
EM_H8_300H = 47       # Hitachi H8/300H
EM_H8S = 48           # Hitachi H8S
EM_H8_500 = 49        # Hitachi H8/500
EM_IA_64 = 50         # Intel IA-64 processor architecture
EM_MIPS_X = 51        # Stanford MIPS-X
EM_COLDFIRE = 52      # Motorola ColdFire
EM_68HC12 = 53        # Motorola M68HC12
EM_MMA = 54           # Fujitsu MMA Multimedia Accelerator
EM_PCP = 55           # Siemens PCP
EM_NCPU = 56          # Sony nCPU embedded RISC processor
EM_NDR1 = 57          # Denso NDR1 microprocessor
EM_STARCORE = 58      # Motorola Star*Core processor
EM_ME16 = 59          # Toyota ME16 processor
EM_ST100 = 60         # STMicroelectronics ST100 processor
EM_TINYJ = 61         # Advanced Logic Corp. TinyJ embedded processor family
EM_X86_64 = 62        # AMD x86-64 architecture
EM_PDSP = 63          # Sony DSP Processor
EM_PDP10 = 64         # Digital Equipment Corp. PDP-10
EM_PDP11 = 65         # Digital Equipment Corp. PDP-11
EM_FX66 = 66          # Siemens FX66 microcontroller
EM_ST9PLUS = 67       # STMicroelectronics ST9+ 8/16 bit microcontroller
EM_ST7 = 68           # STMicroelectronics ST7 8-bit microcontroller
EM_68HC16 = 69        # Motorola MC68HC16 Microcontroller
EM_68HC11 = 70        # Motorola MC68HC11 Microcontroller
EM_68HC08 = 71        # Motorola MC68HC08 Microcontroller
EM_68HC05 = 72        # Motorola MC68HC05 Microcontroller
EM_SVX = 73           # Silicon Graphics SVx
EM_ST19 = 74          # STMicroelectronics ST19 8-bit microcontroller
EM_VAX = 75           # Digital VAX
EM_CRIS = 76          # Axis Communications 32-bit embedded processor
EM_JAVELIN = 77       # Infineon Technologies 32-bit embedded processor
EM_FIREPATH = 78      # Element 14 64-bit DSP Processor
EM_ZSP = 79           # LSI Logic 16-bit DSP Processor
EM_MMIX = 80          # Donald Knuth's educational 64-bit processor
EM_HUANY = 81         # Harvard University machine-independent object files
EM_PRISM = 82         # SiTera Prism
EM_AVR = 83           # Atmel AVR 8-bit microcontroller
EM_FR30 = 84          # Fujitsu FR30
EM_D10V = 85          # Mitsubishi D10V
EM_D30V = 86          # Mitsubishi D30V
EM_V850 = 87          # NEC v850
EM_M32R = 88          # Mitsubishi M32R
EM_MN10300 = 89       # Matsushita MN10300
EM_MN10200 = 90       # Matsushita MN10200
EM_PJ = 91            # picoJava
EM_OPENRISC = 92      # OpenRISC 32-bit embedded processor
EM_ARC_COMPACT = 93   # ARC International ARCompact processor (old
# spelling/synonym: EM_ARC_A5)
EM_XTENSA = 94        # Tensilica Xtensa Architecture
EM_VIDEOCORE = 95     # Alphamosaic VideoCore processor
EM_TMM_GPP = 96       # Thompson Multimedia General Purpose Processor
EM_NS32K = 97         # National Semiconductor 32000 series
EM_TPC = 98           # Tenor Network TPC processor
EM_SNP1K = 99         # Trebia SNP 1000 processor
EM_ST200 = 100        # STMicroelectronics (www.st.com) ST200
EM_IP2K = 101         # Ubicom IP2xxx microcontroller family
EM_MAX = 102          # MAX Processor
EM_CR = 103           # National Semiconductor CompactRISC microprocessor
EM_F2MC16 = 104       # Fujitsu F2MC16
EM_MSP430 = 105       # Texas Instruments embedded microcontroller msp430
EM_BLACKFIN = 106     # Analog Devices Blackfin (DSP) processor
EM_SE_C33 = 107       # S1C33 Family of Seiko Epson processors
EM_SEP = 108          # Sharp embedded microprocessor
EM_ARCA = 109         # Arca RISC Microprocessor
EM_UNICORE = 110      # Microprocessor series from PKU-Unity Ltd. and MPRC
# of Peking University
EM_EXCESS = 111       # eXcess: 16/32/64-bit configurable embedded CPU
EM_DXP = 112          # Icera Semiconductor Inc. Deep Execution Processor
EM_ALTERA_NIOS2 = 113  # Altera Nios II soft-core processor
EM_CRX = 114          # National Semiconductor CompactRISC CRX
EM_XGATE = 115        # Motorola XGATE embedded processor
EM_C166 = 116         # Infineon C16x/XC16x processor
EM_M16C = 117         # Renesas M16C series microprocessors
EM_DSPIC30F = 118     # Microchip Technology dsPIC30F Digital Signal
# Controller
EM_CE = 119           # Freescale Communication Engine RISC core
EM_M32C = 120         # Renesas M32C series microprocessors
EM_TSK3000 = 131      # Altium TSK3000 core
EM_RS08 = 132         # Freescale RS08 embedded processor
EM_SHARC = 133        # Analog Devices SHARC family of 32-bit DSP
# processors
EM_ECOG2 = 134        # Cyan Technology eCOG2 microprocessor
EM_SCORE7 = 135       # Sunplus S+core7 RISC processor
EM_DSP24 = 136        # New Japan Radio (NJR) 24-bit DSP Processor
EM_VIDEOCORE3 = 137   # Broadcom VideoCore III processor
EM_LATTICEMICO32 = 138  # RISC processor for Lattice FPGA architecture
EM_SE_C17 = 139        # Seiko Epson C17 family
EM_TI_C6000 = 140      # The Texas Instruments TMS320C6000 DSP family
EM_TI_C2000 = 141      # The Texas Instruments TMS320C2000 DSP family
EM_TI_C5500 = 142      # The Texas Instruments TMS320C55x DSP family
EM_MMDSP_PLUS = 160    # STMicroelectronics 64bit VLIW Data Signal Processor
EM_CYPRESS_M8C = 161   # Cypress M8C microprocessor
EM_R32C = 162          # Renesas R32C series microprocessors
EM_TRIMEDIA = 163      # NXP Semiconductors TriMedia architecture family
EM_HEXAGON = 164       # Qualcomm Hexagon processor
EM_8051 = 165          # Intel 8051 and variants
EM_STXP7X = 166        # STMicroelectronics STxP7x family of configurable
# and extensible RISC processors
EM_NDS32 = 167         # Andes Technology compact code size embedded RISC
# processor family
EM_ECOG1 = 168         # Cyan Technology eCOG1X family
EM_ECOG1X = 168        # Cyan Technology eCOG1X family
EM_MAXQ30 = 169        # Dallas Semiconductor MAXQ30 Core Micro-controllers
EM_XIMO16 = 170        # New Japan Radio (NJR) 16-bit DSP Processor
EM_MANIK = 171         # M2000 Reconfigurable RISC Microprocessor
EM_CRAYNV2 = 172       # Cray Inc. NV2 vector architecture
EM_RX = 173            # Renesas RX family
EM_METAG = 174         # Imagination Technologies META processor
# architecture
EM_MCST_ELBRUS = 175   # MCST Elbrus general purpose hardware architecture
EM_ECOG16 = 176        # Cyan Technology eCOG16 family
EM_CR16 = 177          # National Semiconductor CompactRISC CR16 16-bit
# microprocessor
EM_ETPU = 178          # Freescale Extended Time Processing Unit
EM_SLE9X = 179         # Infineon Technologies SLE9X core
EM_L10M = 180          # Intel L10M
EM_K10M = 181          # Intel K10M
EM_AARCH64 = 183       # ARM AArch64
EM_AVR32 = 185         # Atmel Corporation 32-bit microprocessor family
EM_STM8 = 186          # STMicroeletronics STM8 8-bit microcontroller
EM_TILE64 = 187        # Tilera TILE64 multicore architecture family
EM_TILEPRO = 188       # Tilera TILEPro multicore architecture family
EM_CUDA = 190          # NVIDIA CUDA architecture
EM_TILEGX = 191        # Tilera TILE-Gx multicore architecture family
EM_CLOUDSHIELD = 192   # CloudShield architecture family
EM_COREA_1ST = 193     # KIPO-KAIST Core-A 1st generation processor family
EM_COREA_2ND = 194     # KIPO-KAIST Core-A 2nd generation processor family
EM_ARC_COMPACT2 = 195  # Synopsys ARCompact V2
EM_OPEN8 = 196         # Open8 8-bit RISC soft processor core
EM_RL78 = 197          # Renesas RL78 family
EM_VIDEOCORE5 = 198    # Broadcom VideoCore V processor
EM_78KOR = 199         # Renesas 78KOR family
EM_56800EX = 200       # Freescale 56800EX Digital Signal Controller (DSC)
EM_BA1 = 201           # Beyond BA1 CPU architecture
EM_BA2 = 202           # Beyond BA2 CPU architecture
EM_XCORE = 203         # XMOS xCORE processor family
EM_MCHP_PIC = 204      # Microchip 8-bit PIC(r) family
EM_INTEL205 = 205      # Reserved by Intel
EM_INTEL206 = 206      # Reserved by Intel
EM_INTEL207 = 207      # Reserved by Intel
EM_INTEL208 = 208      # Reserved by Intel
EM_INTEL209 = 209      # Reserved by Intel
EM_KM32 = 210          # KM211 KM32 32-bit processor
EM_KMX32 = 211         # KM211 KMX32 32-bit processor
EM_KMX16 = 212         # KM211 KMX16 16-bit processor
EM_KMX8 = 213          # KM211 KMX8 8-bit processor
EM_KVARC = 214         # KM211 KVARC processor
EM_CDP = 215           # Paneve CDP architecture family
EM_COGE = 216          # Cognitive Smart Memory Processor
EM_COOL = 217          # iCelero CoolEngine
EM_NORC = 218          # Nanoradio Optimized RISC
EM_CSR_KALIMBA = 219   # CSR Kalimba architecture family
EM_AMDGPU = 224        # AMD GPU architecture
EM_RISCV = 243         # RISC-V
EM_LANAI = 244         # Lanai 32-bit processor
EM_BPF = 247           # Linux kernel bpf virtual machine

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

STB_LOCAL = 0  # Local symbol not visible outside obj file containing def
STB_GLOBAL = 1  # Global symbol visible to all object files being combined
STB_WEAK = 2   # Weak symbol like global but lower-precedence
STB_GNU_UNIQUE = 10
STB_LOOS = 10   # Lowest operating system-specific binding type
STB_HIOS = 12   # Highest operating system-specific binding type
STB_LOPROC = 13  # Lowest processor-specific binding type
STB_HIPROC = 15  # Highest processor-specific binding type


STV_DEFAULT = 0  # Visibility is specified by binding type
STV_INTERNAL = 1  # Defined by processor supplements
STV_HIDDEN = 2   # Not visible to other components
STV_PROTECTED = 3  # Visible in other components but not preemptable


class ELFStringTableBuilder:
    def __init__(self):
        self.data = bytearray([0])

        self.string_map = {}

    def add(self, s: str):
        if s in self.string_map:
            return self.string_map[s]

        idx = len(self.data)
        self.data.extend((s + '\0').encode())
        self.string_map[s] = idx
        return idx

    def get_offset(self, s: str):
        if s in self.string_map:
            return self.string_map[s]

        raise ValueError()


class ELFSymbolData:
    def __init__(self):
        self.symbol = None
        self.section_index = 0
        self.name = ""


class ELFRelocationData:
    def __init__(self, offset: int, symbol: MCSymbol, ty, addend):
        self.offset = offset
        self.symbol = symbol
        self.ty = ty
        self.addend = addend


from enum import Enum, auto


class X64RelType(Enum):
    RT64_NONE = auto()
    RT64_64 = auto()
    RT64_32 = auto()
    RT64_32S = auto()
    RT64_16 = auto()
    RT64_8 = auto()


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


R_AARCH64_NONE = 0
R_AARCH64_ABS64 = 0x101
R_AARCH64_ABS32 = 0x102
R_AARCH64_ABS16 = 0x103
R_AARCH64_PREL64 = 0x104
R_AARCH64_PREL32 = 0x105
R_AARCH64_PREL16 = 0x106
R_AARCH64_MOVW_UABS_G0 = 0x107
R_AARCH64_MOVW_UABS_G0_NC = 0x108
R_AARCH64_MOVW_UABS_G1 = 0x109
R_AARCH64_MOVW_UABS_G1_NC = 0x10a
R_AARCH64_MOVW_UABS_G2 = 0x10b
R_AARCH64_MOVW_UABS_G2_NC = 0x10c
R_AARCH64_MOVW_UABS_G3 = 0x10d
R_AARCH64_MOVW_SABS_G0 = 0x10e
R_AARCH64_MOVW_SABS_G1 = 0x10f
R_AARCH64_MOVW_SABS_G2 = 0x110
R_AARCH64_LD_PREL_LO19 = 0x111
R_AARCH64_ADR_PREL_LO21 = 0x112
R_AARCH64_ADR_PREL_PG_HI21 = 0x113
R_AARCH64_ADR_PREL_PG_HI21_NC = 0x114
R_AARCH64_ADD_ABS_LO12_NC = 0x115
R_AARCH64_LDST8_ABS_LO12_NC = 0x116
R_AARCH64_TSTBR14 = 0x117
R_AARCH64_CONDBR19 = 0x118
R_AARCH64_JUMP26 = 0x11a
R_AARCH64_CALL26 = 0x11b
R_AARCH64_LDST16_ABS_LO12_NC = 0x11c
R_AARCH64_LDST32_ABS_LO12_NC = 0x11d
R_AARCH64_LDST64_ABS_LO12_NC = 0x11e
R_AARCH64_MOVW_PREL_G0 = 0x11f
R_AARCH64_MOVW_PREL_G0_NC = 0x120
R_AARCH64_MOVW_PREL_G1 = 0x121
R_AARCH64_MOVW_PREL_G1_NC = 0x122
R_AARCH64_MOVW_PREL_G2 = 0x123
R_AARCH64_MOVW_PREL_G2_NC = 0x124
R_AARCH64_MOVW_PREL_G3 = 0x125
R_AARCH64_LDST128_ABS_LO12_NC = 0x12b
R_AARCH64_MOVW_GOTOFF_G0 = 0x12c
R_AARCH64_MOVW_GOTOFF_G0_NC = 0x12d
R_AARCH64_MOVW_GOTOFF_G1 = 0x12e
R_AARCH64_MOVW_GOTOFF_G1_NC = 0x12f
R_AARCH64_MOVW_GOTOFF_G2 = 0x130
R_AARCH64_MOVW_GOTOFF_G2_NC = 0x131
R_AARCH64_MOVW_GOTOFF_G3 = 0x132
R_AARCH64_GOTREL64 = 0x133
R_AARCH64_GOTREL32 = 0x134
R_AARCH64_GOT_LD_PREL19 = 0x135
R_AARCH64_LD64_GOTOFF_LO15 = 0x136
R_AARCH64_ADR_GOT_PAGE = 0x137
R_AARCH64_LD64_GOT_LO12_NC = 0x138
R_AARCH64_LD64_GOTPAGE_LO15 = 0x139
R_AARCH64_TLSGD_ADR_PREL21 = 0x200
R_AARCH64_TLSGD_ADR_PAGE21 = 0x201
R_AARCH64_TLSGD_ADD_LO12_NC = 0x202
R_AARCH64_TLSGD_MOVW_G1 = 0x203
R_AARCH64_TLSGD_MOVW_G0_NC = 0x204
R_AARCH64_TLSLD_ADR_PREL21 = 0x205
R_AARCH64_TLSLD_ADR_PAGE21 = 0x206
R_AARCH64_TLSLD_ADD_LO12_NC = 0x207
R_AARCH64_TLSLD_MOVW_G1 = 0x208
R_AARCH64_TLSLD_MOVW_G0_NC = 0x209
R_AARCH64_TLSLD_LD_PREL19 = 0x20a
R_AARCH64_TLSLD_MOVW_DTPREL_G2 = 0x20b
R_AARCH64_TLSLD_MOVW_DTPREL_G1 = 0x20c
R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC = 0x20d
R_AARCH64_TLSLD_MOVW_DTPREL_G0 = 0x20e
R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC = 0x20f
R_AARCH64_TLSLD_ADD_DTPREL_HI12 = 0x210
R_AARCH64_TLSLD_ADD_DTPREL_LO12 = 0x211
R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC = 0x212
R_AARCH64_TLSLD_LDST8_DTPREL_LO12 = 0x213
R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC = 0x214
R_AARCH64_TLSLD_LDST16_DTPREL_LO12 = 0x215
R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC = 0x216
R_AARCH64_TLSLD_LDST32_DTPREL_LO12 = 0x217
R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC = 0x218
R_AARCH64_TLSLD_LDST64_DTPREL_LO12 = 0x219
R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC = 0x21a
R_AARCH64_TLSIE_MOVW_GOTTPREL_G1 = 0x21b
R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC = 0x21c
R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 = 0x21d
R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 0x21e
R_AARCH64_TLSIE_LD_GOTTPREL_PREL19 = 0x21f
R_AARCH64_TLSLE_MOVW_TPREL_G2 = 0x220
R_AARCH64_TLSLE_MOVW_TPREL_G1 = 0x221
R_AARCH64_TLSLE_MOVW_TPREL_G1_NC = 0x222
R_AARCH64_TLSLE_MOVW_TPREL_G0 = 0x223
R_AARCH64_TLSLE_MOVW_TPREL_G0_NC = 0x224
R_AARCH64_TLSLE_ADD_TPREL_HI12 = 0x225
R_AARCH64_TLSLE_ADD_TPREL_LO12 = 0x226
R_AARCH64_TLSLE_ADD_TPREL_LO12_NC = 0x227
R_AARCH64_TLSLE_LDST8_TPREL_LO12 = 0x228
R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC = 0x229
R_AARCH64_TLSLE_LDST16_TPREL_LO12 = 0x22a
R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC = 0x22b
R_AARCH64_TLSLE_LDST32_TPREL_LO12 = 0x22c
R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC = 0x22d
R_AARCH64_TLSLE_LDST64_TPREL_LO12 = 0x22e
R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC = 0x22f
R_AARCH64_TLSDESC_LD_PREL19 = 0x230
R_AARCH64_TLSDESC_ADR_PREL21 = 0x231
R_AARCH64_TLSDESC_ADR_PAGE21 = 0x232
R_AARCH64_TLSDESC_LD64_LO12 = 0x233
R_AARCH64_TLSDESC_ADD_LO12 = 0x234
R_AARCH64_TLSDESC_OFF_G1 = 0x235
R_AARCH64_TLSDESC_OFF_G0_NC = 0x236
R_AARCH64_TLSDESC_LDR = 0x237
R_AARCH64_TLSDESC_ADD = 0x238
R_AARCH64_TLSDESC_CALL = 0x239
R_AARCH64_TLSLE_LDST128_TPREL_LO12 = 0x23a
R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC = 0x23b
R_AARCH64_TLSLD_LDST128_DTPREL_LO12 = 0x23c
R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC = 0x23d
R_AARCH64_COPY = 0x400
R_AARCH64_GLOB_DAT = 0x401
R_AARCH64_JUMP_SLOT = 0x402
R_AARCH64_RELATIVE = 0x403
# 0x404 and 0x405 are now R_AARCH64_TLS_IMPDEF1 and R_AARCH64_TLS_IMPDEF2
# We follow GNU and define TLS_IMPDEF1 as TLS_DTPMOD64 and TLS_IMPDEF2 as
# TLS_DTPREL64
R_AARCH64_TLS_DTPMOD64 = 0x404
R_AARCH64_TLS_DTPREL64 = 0x405
R_AARCH64_TLS_TPREL64 = 0x406
R_AARCH64_TLSDESC = 0x407
R_AARCH64_IRELATIVE = 0x408

# R_AARCH64_P32_NONE =                         0
R_AARCH64_P32_ABS32 = 0x001
R_AARCH64_P32_ABS16 = 0x002
R_AARCH64_P32_PREL32 = 0x003
R_AARCH64_P32_PREL16 = 0x004
R_AARCH64_P32_MOVW_UABS_G0 = 0x005
R_AARCH64_P32_MOVW_UABS_G0_NC = 0x006
R_AARCH64_P32_MOVW_UABS_G1 = 0x007
R_AARCH64_P32_MOVW_SABS_G0 = 0x008
R_AARCH64_P32_LD_PREL_LO19 = 0x009
R_AARCH64_P32_ADR_PREL_LO21 = 0x00a
R_AARCH64_P32_ADR_PREL_PG_HI21 = 0x00b
R_AARCH64_P32_ADD_ABS_LO12_NC = 0x00c
R_AARCH64_P32_LDST8_ABS_LO12_NC = 0x00d
R_AARCH64_P32_LDST16_ABS_LO12_NC = 0x00e
R_AARCH64_P32_LDST32_ABS_LO12_NC = 0x00f
R_AARCH64_P32_LDST64_ABS_LO12_NC = 0x010
R_AARCH64_P32_LDST128_ABS_LO12_NC = 0x011
R_AARCH64_P32_TSTBR14 = 0x012
R_AARCH64_P32_CONDBR19 = 0x013
R_AARCH64_P32_JUMP26 = 0x014
R_AARCH64_P32_CALL26 = 0x015
R_AARCH64_P32_MOVW_PREL_G0 = 0x016
R_AARCH64_P32_MOVW_PREL_G0_NC = 0x017
R_AARCH64_P32_MOVW_PREL_G1 = 0x018
R_AARCH64_P32_GOT_LD_PREL19 = 0x019
R_AARCH64_P32_ADR_GOT_PAGE = 0x01a
R_AARCH64_P32_LD32_GOT_LO12_NC = 0x01b
R_AARCH64_P32_LD32_GOTPAGE_LO14 = 0x01c
R_AARCH64_P32_TLSGD_ADR_PREL21 = 0x050
R_AARCH64_P32_TLSGD_ADR_PAGE21 = 0x051
R_AARCH64_P32_TLSGD_ADD_LO12_NC = 0x052
R_AARCH64_P32_TLSLD_ADR_PREL21 = 0x053
R_AARCH64_P32_TLSLD_ADR_PAGE21 = 0x054
R_AARCH64_P32_TLSLD_ADD_LO12_NC = 0x055
R_AARCH64_P32_TLSLD_LD_PREL19 = 0x056
R_AARCH64_P32_TLSLD_MOVW_DTPREL_G1 = 0x057
R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0 = 0x058
R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0_NC = 0x059
R_AARCH64_P32_TLSLD_ADD_DTPREL_HI12 = 0x05a
R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12 = 0x05b
R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12_NC = 0x05c
R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12 = 0x05d
R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12_NC = 0x05e
R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12 = 0x05f
R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12_NC = 0x060
R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12 = 0x061
R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12_NC = 0x062
R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12 = 0x063
R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12_NC = 0x064
R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12 = 0x065
R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12_NC = 0x066
R_AARCH64_P32_TLSIE_ADR_GOTTPREL_PAGE21 = 0x067
R_AARCH64_P32_TLSIE_LD32_GOTTPREL_LO12_NC = 0x068
R_AARCH64_P32_TLSIE_LD_GOTTPREL_PREL19 = 0x069
R_AARCH64_P32_TLSLE_MOVW_TPREL_G1 = 0x06a
R_AARCH64_P32_TLSLE_MOVW_TPREL_G0 = 0x06b
R_AARCH64_P32_TLSLE_MOVW_TPREL_G0_NC = 0x06c
R_AARCH64_P32_TLSLE_ADD_TPREL_HI12 = 0x06d
R_AARCH64_P32_TLSLE_ADD_TPREL_LO12 = 0x06e
R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC = 0x06f
R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12 = 0x070
R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12_NC = 0x071
R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12 = 0x072
R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12_NC = 0x073
R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12 = 0x074
R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12_NC = 0x075
R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12 = 0x076
R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12_NC = 0x077
R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12 = 0x078
R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12_NC = 0x079
R_AARCH64_P32_TLSDESC_LD_PREL19 = 0x07a
R_AARCH64_P32_TLSDESC_ADR_PREL21 = 0x07b
R_AARCH64_P32_TLSDESC_ADR_PAGE21 = 0x07c
R_AARCH64_P32_TLSDESC_LD32_LO12 = 0x07d
R_AARCH64_P32_TLSDESC_ADD_LO12 = 0x07e
R_AARCH64_P32_TLSDESC_CALL = 0x07f
R_AARCH64_P32_COPY = 0x0b4
R_AARCH64_P32_GLOB_DAT = 0x0b5
R_AARCH64_P32_JUMP_SLOT = 0x0b6
R_AARCH64_P32_RELATIVE = 0x0b7
R_AARCH64_P32_TLS_DTPREL = 0x0b8
R_AARCH64_P32_TLS_DTPMOD = 0x0b9
R_AARCH64_P32_TLS_TPREL = 0x0ba
R_AARCH64_P32_TLSDESC = 0x0bb
R_AARCH64_P32_IRELATIVE = 0x0bc

EF_ARM_SOFT_FLOAT = 0x00000200     # Legacy pre EABI_VER5
EF_ARM_ABI_FLOAT_SOFT = 0x00000200  # EABI_VER5
EF_ARM_VFP_FLOAT = 0x00000400      # Legacy pre EABI_VER5
EF_ARM_ABI_FLOAT_HARD = 0x00000400  # EABI_VER5
EF_ARM_EABI_UNKNOWN = 0x00000000
EF_ARM_EABI_VER1 = 0x01000000
EF_ARM_EABI_VER2 = 0x02000000
EF_ARM_EABI_VER3 = 0x03000000
EF_ARM_EABI_VER4 = 0x04000000
EF_ARM_EABI_VER5 = 0x05000000
EF_ARM_EABIMASK = 0xFF000000

EF_RISCV_RVC = 0x0001
EF_RISCV_FLOAT_ABI = 0x0006
EF_RISCV_FLOAT_ABI_SOFT = 0x0000
EF_RISCV_FLOAT_ABI_SINGLE = 0x0002
EF_RISCV_FLOAT_ABI_DOUBLE = 0x0004
EF_RISCV_FLOAT_ABI_QUAD = 0x0006
EF_RISCV_RVE = 0x0008

R_RISCV_NONE = 0
R_RISCV_32 = 1
R_RISCV_64 = 2
R_RISCV_RELATIVE = 3
R_RISCV_COPY = 4
R_RISCV_JUMP_SLOT = 5
R_RISCV_TLS_DTPMOD32 = 6
R_RISCV_TLS_DTPMOD64 = 7
R_RISCV_TLS_DTPREL32 = 8
R_RISCV_TLS_DTPREL64 = 9
R_RISCV_TLS_TPREL32 = 10
R_RISCV_TLS_TPREL64 = 11
R_RISCV_BRANCH = 16
R_RISCV_JAL = 17
R_RISCV_CALL = 18
R_RISCV_CALL_PLT = 19
R_RISCV_GOT_HI20 = 20
R_RISCV_TLS_GOT_HI20 = 21
R_RISCV_TLS_GD_HI20 = 22
R_RISCV_PCREL_HI20 = 23
R_RISCV_PCREL_LO12_I = 24
R_RISCV_PCREL_LO12_S = 25
R_RISCV_HI20 = 26
R_RISCV_LO12_I = 27
R_RISCV_LO12_S = 28
R_RISCV_TPREL_HI20 = 29
R_RISCV_TPREL_LO12_I = 30
R_RISCV_TPREL_LO12_S = 31
R_RISCV_TPREL_ADD = 32
R_RISCV_ADD8 = 33
R_RISCV_ADD16 = 34
R_RISCV_ADD32 = 35
R_RISCV_ADD64 = 36
R_RISCV_SUB8 = 37
R_RISCV_SUB16 = 38
R_RISCV_SUB32 = 39
R_RISCV_SUB64 = 40
R_RISCV_GNU_VTINHERIT = 41
R_RISCV_GNU_VTENTRY = 42
R_RISCV_ALIGN = 43
R_RISCV_RVC_BRANCH = 44
R_RISCV_RVC_JUMP = 45
R_RISCV_RVC_LUI = 46
R_RISCV_GPREL_I = 47
R_RISCV_GPREL_S = 48
R_RISCV_TPREL_I = 49
R_RISCV_TPREL_S = 50
R_RISCV_RELAX = 51
R_RISCV_SUB6 = 52
R_RISCV_SET6 = 53
R_RISCV_SET8 = 54
R_RISCV_SET16 = 55
R_RISCV_SET32 = 56
R_RISCV_32_PCREL = 57


class ELFObjectWriter(MCObjectWriter):
    def __init__(self, output, target_writer):
        super().__init__()

        self.target_writer = target_writer
        self.output = output
        self.is_little_endian = True
        self.section_table = []

        self.string_table = ELFStringTableBuilder()

        self.string_table_index = 0
        self.symbol_table_index = 0
        self.last_local_symbol_index = 0

        self.relocations = {}

        self.reloc_sections = []

    def compute_after_layout(self, asm, obj):
        pass

    @property
    def is_64bit(self):
        return self.target_writer.is_64bit

    def get_reloc_type(self, context: MCContext, target: MCValue, fixup: MCFixup, is_pcrel):
        return self.target_writer.get_reloc_type(context, target, fixup, is_pcrel)

    def record_relocation(self, asm: MCAssembler, layout: MCAsmLayout, fragment: MCFragment, fixup: MCFixup, target: MCValue, fixed_val):
        section = fragment.section
        fixup_offset = layout.get_fragment_offset(fragment) + fixup.offset

        symrefa = target.symbol1
        symrefb = target.symbol2
        constant = target.value

        if symrefb is not None:
            raise NotImplementedError()

        assert(symrefa is not None)

        syma = symrefa.symbol

        is_pcrel = asm.backend.is_fixup_kind_pcrel(fixup)
        ty = self.get_reloc_type(asm.context, target, fixup, is_pcrel)

        reloc_with_symbol = False
        if syma.binding == ELFSymbolBinding.STB_GLOBAL:
            reloc_with_symbol = True

        reloc_with_symbol = True

        if reloc_with_symbol:
            fixed_val = constant

        if section not in self.relocations:
            self.relocations[section] = []

        self.relocations[section].append(
            ELFRelocationData(fixup_offset, syma, ty, constant))

        return fixed_val

    def write_word(self, value):
        order = 'little'

        if self.is_64bit:
            self.output.write(value.to_bytes(8, byteorder=order))
        else:
            self.output.write(value.to_bytes(4, byteorder=order))

    def write_header(self, asm):
        order = 'little'

        elf_class = ELFCLASS64 if self.is_64bit else ELFCLASS32

        endian = ELFDATA2LSB
        # endian = ELFDATA2MSB

        self.output.write(ELF_MAGIC)
        self.output.write(elf_class.to_bytes(1, byteorder=order))
        self.output.write(endian.to_bytes(1, byteorder=order))
        self.output.write(EV_CURRENT.to_bytes(1, byteorder=order))
        self.output.write(
            int(self.target_writer.os_abi).to_bytes(1, byteorder=order))
        self.output.write(
            int(self.target_writer.abi_version).to_bytes(1, byteorder=order))
        self.output.write(bytes(EI_NIDENT - EI_PAD))
        self.output.write(ET_REL.to_bytes(2, byteorder=order))
        self.output.write(
            int(self.target_writer.emachine).to_bytes(2, byteorder=order))
        self.output.write(EV_CURRENT.to_bytes(4, byteorder=order))

        self.write_word(0)
        self.write_word(0)
        self.write_word(0)

        self.output.write(self.target_writer.flags.to_bytes(
            4, byteorder=order))  # e_flags

        header_size = 64 if self.is_64bit else 52
        self.output.write(header_size.to_bytes(2, byteorder=order))
        self.output.write(int(0).to_bytes(2, byteorder=order))
        self.output.write(int(0).to_bytes(2, byteorder=order))

        sh_size = 64 if self.is_64bit else 40
        self.output.write(sh_size.to_bytes(2, byteorder=order))
        self.output.write(int(0).to_bytes(2, byteorder=order))

        self.output.write(self.string_table_index.to_bytes(2, byteorder=order))

    def write_section_header_entry(self, name: int, ty: int, flags: int,
                                   address: int, offset: int, size: int, link: int, info: int,
                                   alignment: int, entry_size: int):

        order = 'little'

        self.output.write(name.to_bytes(4, byteorder=order))
        self.output.write(ty.to_bytes(4, byteorder=order))
        self.write_word(flags)
        self.write_word(address)
        self.write_word(offset)
        self.write_word(size)
        self.output.write(link.to_bytes(4, byteorder=order))
        self.output.write(info.to_bytes(4, byteorder=order))
        self.write_word(alignment)
        self.write_word(entry_size)

    def write_section(self, section: MCSectionELF, offset, size, section_indices):
        sh_link = 0
        sh_info = 0

        if section.ty == SHT_SYMTAB:
            sh_link = self.string_table_index
            sh_info = self.last_local_symbol_index

        if section.ty in [SHT_REL, SHT_RELA]:
            sh_link = self.symbol_table_index
            sh_info = section_indices[section.associated_section]

        alignment = section.alignment
        entry_size = section.entry_size

        self.write_section_header_entry(
            self.string_table.get_offset(
                section.name), section.ty, section.flags, 0, offset, size,
            sh_link, sh_info, alignment, entry_size)

    def write_section_header(self, layout, section_indices, section_offsets):
        # Null section first.
        first_section_size = 0
        self.write_section_header_entry(
            0, 0, 0, 0, 0, first_section_size, 0, 0, 0, 0)

        for section in self.section_table:
            start_pos, end_pos = section_offsets[section]
            offset = start_pos
            size = end_pos - start_pos
            self.write_section(section, offset, size, section_indices)

    def write_section_data(self, asm, section, layout):
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

    def align(self, alignment):
        pos = self.output.tell()
        pos_aligned = int(int((pos + alignment - 1) / alignment) * alignment)

        if pos_aligned - pos > 0:
            paddings = bytearray(pos_aligned - pos)
            self.output.write(paddings)

    def write_symbol_entry(self, name: int, info: int, value: int,
                           size: int, other: int, shndx: int, reserved: bool):

        order = 'little'

        index = shndx

        if self.is_64bit:
            self.output.write(name.to_bytes(4, byteorder=order))
            self.output.write(info.to_bytes(1, byteorder=order))
            self.output.write(other.to_bytes(1, byteorder=order))
            self.output.write(index.to_bytes(2, byteorder=order))
            self.output.write(value.to_bytes(8, byteorder=order))
            self.output.write(size.to_bytes(8, byteorder=order))
        else:
            self.output.write(name.to_bytes(4, byteorder=order))
            self.output.write(value.to_bytes(4, byteorder=order))
            self.output.write(size.to_bytes(4, byteorder=order))
            self.output.write(info.to_bytes(1, byteorder=order))
            self.output.write(other.to_bytes(1, byteorder=order))
            self.output.write(index.to_bytes(2, byteorder=order))

    def write_symbol(self, string_idx: int, symbol_data: ELFSymbolData, asm: MCAssembler, layout: MCAsmLayout):
        symbol = symbol_data.symbol

        ty = symbol.ty.value
        binding = symbol.binding.value
        info = (binding << 4) | ty

        value = self.get_symbol_value(symbol_data.symbol, layout)

        size_expr = symbol_data.symbol.size
        size = 0
        if size_expr:
            size = evaluate_expr_as_constant(
                symbol_data.symbol.size, asm, layout)
            assert(size is not None)

        other = symbol.visibility.value | symbol.other

        self.write_symbol_entry(string_idx, info, value,
                                size, other, symbol_data.section_index, False)

    def is_in_symtab(self, layout, symbol: MCSymbol, used, renamed):
        # if symbol.temporary:
        #     return False

        return True

    def get_symbol_value(self, symbol, layout: MCAsmLayout):
        return layout.get_symbol_offset(symbol)

    def write_symbol_table(self, asm: MCAssembler, layout, section_indices, section_offsets):
        entry_size = 24 if self.is_64bit else 16

        symtab_section = create_elf_section(
            ".symtab", SHT_SYMTAB, 0, entry_size)
        symtab_section.alignment = 8 if self.is_64bit else 4

        self.section_table.append(symtab_section)
        self.string_table.add(symtab_section.name)
        self.symbol_table_index = len(self.section_table)

        self.align(symtab_section.alignment)

        section_start_pos = self.output.tell()

        self.write_symbol_entry(0, 0, 0, 0, 0, 0, False)

        local_symbol_data = []
        external_symbol_data = []

        for symbol in asm.symbols:
            used = True
            weakref_used = True
            is_signature = True
            if not self.is_in_symtab(layout, symbol, used or wekref_used or is_signature, False):
                continue

            data = ELFSymbolData()

            data.symbol = symbol

            if symbol.is_undefined:
                data.section_index = SHN_UNDEF
            else:
                section = symbol.fragment.section
                data.section_index = section_indices[section]

            data.name = symbol.name

            self.string_table.add(symbol.name)

            is_local = symbol.binding == ELFSymbolBinding.STB_LOCAL

            if is_local:
                local_symbol_data.append(data)
            else:
                external_symbol_data.append(data)

        def cmp(a, b):
            return (a > b)-(a < b)

        def cmp_symbol_data(a: ELFSymbolData, b: ELFSymbolData):
            if a.symbol.ty == STT_SECTION and b.symbol.ty == STT_SECTION:
                return cmp(a.section_index, b.section_index)
            elif a.symbol.ty != STT_SECTION and b.symbol.ty == STT_SECTION:
                return 1
            elif a.symbol.ty == STT_SECTION and b.symbol.ty != STT_SECTION:
                return -1

            return cmp(a.name, b.name)

        import functools

        local_symbol_data.sort(key=functools.cmp_to_key(cmp_symbol_data))
        external_symbol_data.sort(key=functools.cmp_to_key(cmp_symbol_data))

        filenames = ["example1.glsl.ir"]
        for filename in filenames:
            self.string_table.add(filename)

        for filename in filenames:
            name = self.string_table.get_offset(filename)
            info = STT_FILE | STB_LOCAL
            other = STV_DEFAULT
            shndx = SHN_ABS
            self.write_symbol_entry(name, info, 0, 0, other, shndx, True)

        index = len(filenames) + 1

        for symbol_data in local_symbol_data:
            string_idx = self.string_table.get_offset(symbol_data.name)
            self.write_symbol(string_idx, symbol_data, asm, layout)

            symbol_data.symbol.index = index
            index += 1

        self.last_local_symbol_index = index

        for symbol_data in external_symbol_data:
            string_idx = self.string_table.get_offset(symbol_data.name)
            self.write_symbol(string_idx, symbol_data, asm, layout)

            symbol_data.symbol.index = index
            index += 1

            assert(symbol_data.symbol.binding != ELFSymbolBinding.STB_LOCAL)

        section_end_pos = self.output.tell()

        section_offsets[symtab_section] = (section_start_pos, section_end_pos)

    def create_relocation_section(self, context: MCContext, section: MCSectionELF):
        if section not in self.relocations:
            return None

        section_name = section.name

        if self.target_writer.has_relocation_addend:
            rela_section_name = ".rela"
            rel = SHT_RELA
        else:
            rela_section_name = ".rel"
            rel = SHT_REL

        rela_section_name += section_name

        def create_elf_rel_section(name: str, ty, flags, entry_size):
            return create_elf_section(name, ty, flags, entry_size, associated=section)

        flags = 0
        if self.is_64bit:
            if self.target_writer.has_relocation_addend:
                entry_size = 24
            else:
                entry_size = 16
        else:
            if self.target_writer.has_relocation_addend:
                entry_size = 12
            else:
                entry_size = 8

        rela_section = create_elf_rel_section(
            rela_section_name, rel, flags, entry_size)

        return rela_section

    def write_relocations(self, asm: MCAssembler, section: MCSectionELF):
        relocs = self.relocations[section]

        order = 'little'

        def get_info_by_symbol_type64(symbol, ty):
            return ((symbol & 0xFFFFFFFF) << 32) | (ty & 0xFFFFFFFF)

        def get_info_by_symbol_type32(symbol, ty):
            return ((symbol & 0xFFFFFF) << 8) | (ty & 0xFF)

        for entry in relocs:
            if self.is_64bit:
                self.output.write(entry.offset.to_bytes(
                    8, byteorder=order, signed=True))

                index = entry.symbol.index
                ty = entry.ty
                info = get_info_by_symbol_type64(index, ty)

                self.output.write(info.to_bytes(8, byteorder=order))
                if self.target_writer.has_relocation_addend:
                    self.output.write(entry.addend.to_bytes(
                        8, byteorder=order, signed=True))
            else:
                self.output.write(entry.offset.to_bytes(
                    4, byteorder=order, signed=True))

                index = entry.symbol.index
                ty = entry.ty
                info = get_info_by_symbol_type32(index, ty)

                self.output.write(info.to_bytes(4, byteorder=order))

                if self.target_writer.has_relocation_addend:
                    self.output.write(entry.addend.to_bytes(
                        4, byteorder=order, signed=True))

    def write_object(self, asm: MCAssembler, layout: MCAsmBackend):
        strtab_section = create_elf_section(".strtab", SHT_STRTAB, 0)
        self.section_table.append(strtab_section)
        self.string_table.add(strtab_section.name)
        self.string_table_index = len(self.section_table)

        self.write_header(asm)

        section_indices = {}
        section_offsets = {}

        for section in asm.sections:
            self.align(4)

            section_start_pos = self.output.tell()
            self.write_section_data(asm, section, layout)
            section_end_pos = self.output.tell()

            section_offsets[section] = (section_start_pos, section_end_pos)

            self.section_table.append(section)
            self.string_table.add(section.name)
            section_idx = len(self.section_table)

            section_indices[section] = section_idx

            rel_section = self.create_relocation_section(asm.context, section)

            if rel_section is not None:
                self.reloc_sections.append(rel_section)

                self.section_table.append(rel_section)
                self.string_table.add(rel_section.name)
                section_indices[rel_section] = len(self.section_table)

        self.write_symbol_table(asm, layout, section_indices, section_offsets)

        for rel_section in self.reloc_sections:
            section_start_pos = self.output.tell()
            self.write_relocations(asm, rel_section.associated_section)
            section_end_pos = self.output.tell()

            section_offsets[rel_section] = (section_start_pos, section_end_pos)

        # write strtab section
        section = strtab_section
        section_start_pos = self.output.tell()
        self.output.write(self.string_table.data)
        section_end_pos = self.output.tell()

        section_offsets[section] = (section_start_pos, section_end_pos)

        natural_align = 8 if self.is_64bit else 4
        self.align(natural_align)

        section_header_offset = self.output.tell()

        self.write_section_header(layout, section_indices, section_offsets)

        order = 'little'
        saved_pos = self.output.tell()

        if self.is_64bit:
            self.output.seek(40)
            self.output.write(
                section_header_offset.to_bytes(8, byteorder=order))
        else:
            self.output.seek(32)
            self.output.write(
                section_header_offset.to_bytes(4, byteorder=order))

        num_sections = len(self.section_table) + 1
        if self.is_64bit:
            self.output.seek(60)
        else:
            self.output.seek(48)
        self.output.write(num_sections.to_bytes(2, byteorder=order))

        self.output.seek(saved_pos)


class X64ELFTargetObjectFile(MCObjectFileInfo):
    def __init__(self):
        super().__init__()

        self._text_section = create_elf_section(
            ".text", SHT_PROGBITS, SHF_EXECINSTR | SHF_ALLOC, alignment=16)
        self._bss_section = create_elf_section(
            ".bss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC)
        self._data_section = create_elf_section(
            ".data", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC)
        self._rodata_section = create_elf_section(
            ".rodata", SHT_PROGBITS, SHF_ALLOC)
        self._mergeable_const4_section = create_elf_section(
            ".rodata.cst4", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 4, alignment=4)
        self._mergeable_const8_section = create_elf_section(
            ".rodata.cst8", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 8, alignment=8)
        self._mergeable_const16_section = create_elf_section(
            ".rodata.cst16", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 16, alignment=16)
        self._mergeable_const32_section = create_elf_section(
            ".rodata.cst32", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 32, alignment=32)

        self._tls_bss_section = create_elf_section(
            ".tbss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS, alignment=16)
        self._tls_data_section = create_elf_section(
            ".tdata", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS, alignment=16)

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

    def get_section_for_const(self, section_kind: SectionKind, value, align):
        if section_kind == SectionKind.MergeableConst4:
            return self._mergeable_const4_section
        elif section_kind == SectionKind.MergeableConst8:
            return self._mergeable_const8_section
        elif section_kind == SectionKind.MergeableConst16:
            return self._mergeable_const16_section
        elif section_kind == SectionKind.MergeableConst32:
            return self._mergeable_const32_section
        elif section_kind == SectionKind.ReadOnly:
            return self._rodata_section

        raise ValueError("Invalid section kind.")

    @property
    def tls_data_section(self) -> MCSection:
        return self._tls_data_section

    @property
    def is_elf(self):
        return True

    @property
    def is_coff(self):
        return False


class ARMELFTargetObjectFile(MCObjectFileInfo):
    def __init__(self):
        super().__init__()

        self._text_section = create_elf_section(
            ".text", SHT_PROGBITS, SHF_EXECINSTR | SHF_ALLOC)
        self._bss_section = create_elf_section(
            ".bss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC)
        self._data_section = create_elf_section(
            ".data", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC)
        self._rodata_section = create_elf_section(
            ".rodata", SHT_PROGBITS, SHF_ALLOC)

        self._tls_bss_section = create_elf_section(
            ".tbss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS)
        self._tls_data_section = create_elf_section(
            ".tdata", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS)

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

    def get_section_for_const(self, section_kind: SectionKind, value, align):
        return self._text_section

    @property
    def is_elf(self):
        return True

    @property
    def is_coff(self):
        return False


class AArch64ELFTargetObjectFile(MCObjectFileInfo):
    def __init__(self):
        super().__init__()

        self._text_section = create_elf_section(
            ".text", SHT_PROGBITS, SHF_EXECINSTR | SHF_ALLOC)
        self._bss_section = create_elf_section(
            ".bss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC)
        self._data_section = create_elf_section(
            ".data", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC)
        self._rodata_section = create_elf_section(
            ".rodata", SHT_PROGBITS, SHF_ALLOC)
        self._mergeable_const4_section = create_elf_section(
            ".rodata.cst4", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 4, alignment=4)
        self._mergeable_const8_section = create_elf_section(
            ".rodata.cst8", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 8, alignment=8)
        self._mergeable_const16_section = create_elf_section(
            ".rodata.cst16", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 16, alignment=16)
        self._mergeable_const32_section = create_elf_section(
            ".rodata.cst32", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 32, alignment=32)

        self._tls_bss_section = create_elf_section(
            ".tbss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS)
        self._tls_data_section = create_elf_section(
            ".tdata", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS)

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

    def get_section_for_const(self, section_kind: SectionKind, value, align):
        if section_kind == SectionKind.MergeableConst4:
            return self._mergeable_const4_section
        elif section_kind == SectionKind.MergeableConst8:
            return self._mergeable_const8_section
        elif section_kind == SectionKind.MergeableConst16:
            return self._mergeable_const16_section
        elif section_kind == SectionKind.MergeableConst32:
            return self._mergeable_const32_section
        elif section_kind == SectionKind.ReadOnly:
            return self._rodata_section

        raise ValueError("Invalid section kind.")

    @property
    def is_elf(self):
        return True

    @property
    def is_coff(self):
        return False


class RISCVELFTargetObjectFile(MCObjectFileInfo):
    def __init__(self):
        super().__init__()

        self._text_section = create_elf_section(
            ".text", SHT_PROGBITS, SHF_EXECINSTR | SHF_ALLOC)
        self._bss_section = create_elf_section(
            ".bss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC)
        self._data_section = create_elf_section(
            ".data", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC)
        self._sdata_section = create_elf_section(
            ".sdata", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC)
        self._rodata_section = create_elf_section(
            ".rodata", SHT_PROGBITS, SHF_ALLOC)
        self._mergeable_const4_section = create_elf_section(
            ".rodata.cst4", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 4, alignment=4)
        self._mergeable_const8_section = create_elf_section(
            ".rodata.cst8", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 8, alignment=8)
        self._mergeable_const16_section = create_elf_section(
            ".rodata.cst16", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 16, alignment=16)
        self._mergeable_const32_section = create_elf_section(
            ".rodata.cst32", SHT_PROGBITS, SHF_ALLOC | SHF_MERGE, 32, alignment=32)

        self._tls_bss_section = create_elf_section(
            ".tbss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS)
        self._tls_data_section = create_elf_section(
            ".tdata", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC | SHF_TLS)

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

    def get_section_for_const(self, section_kind: SectionKind, value, align):
        if section_kind == SectionKind.MergeableConst4:
            return self._mergeable_const4_section
        elif section_kind == SectionKind.MergeableConst8:
            return self._mergeable_const8_section
        elif section_kind == SectionKind.MergeableConst16:
            return self._mergeable_const16_section
        elif section_kind == SectionKind.MergeableConst32:
            return self._mergeable_const32_section
        elif section_kind == SectionKind.ReadOnly:
            return self._rodata_section

        raise ValueError("Invalid section kind.")

    @property
    def is_elf(self):
        return True

    @property
    def is_coff(self):
        return False
