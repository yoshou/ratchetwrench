from io import BytesIO
from rachetwrench.codegen.passes import PassManager
from rachetwrench.obj.objfile import *

from rachetwrench.codegen.mir import MachineFunction
from rachetwrench.codegen.spec import Triple, ArchType, OS, Environment, ObjectFormatType

# passes
from rachetwrench.opt.mem2reg import Mem2Reg
from rachetwrench.opt.unreachable_bb_elim import UnreachableBlockElim
from rachetwrench.opt.simplify_cfg import SimplifyCFG

from rachetwrench.codegen.isel import InstructionSelection
from rachetwrench.codegen.live_intervals import LiveIntervals
from rachetwrench.codegen.regalloc import FastRegisterAllocation
from rachetwrench.codegen.dead_machine_code_elim import DeadMachineCodeElim
from rachetwrench.codegen.linear_scan_regalloc import LinearScanRegisterAllocation
from rachetwrench.codegen.virtual_reg_rewriter import VirtualRegisterRewriter
from rachetwrench.codegen.prolog_epilog_insertion import PrologEpilogInsertion
from rachetwrench.codegen.peephole_optimize import PeepholeOptimize
from rachetwrench.codegen.expand_pseudos_postra import ExpandPseudosPostRA
from rachetwrench.codegen.two_address_inst import TwoAddressInst
from rachetwrench.codegen.unreachable_bb_elim import UnreachableBBElim
from rachetwrench.codegen.machine_cp import MachineCopyProp
from rachetwrench.codegen.mir_printing import MIRPrinting

# mc
from rachetwrench.codegen.mc import MCContext
from rachetwrench.codegen.coff import COFFObjectFileInfo
from rachetwrench.codegen.elf import X64ELFTargetObjectFile

from rachetwrench.codegen.elf import *
from rachetwrench.codegen.coff import *


from ctypes import *

import os

if os.name == "nt":
    from ctypes.wintypes import *

    is_64bit = sizeof(c_void_p) == sizeof(c_ulonglong)

    PWORD = POINTER(WORD)
    PDWORD = POINTER(DWORD)
    PHMODULE = POINTER(HMODULE)
    LONG_PTR = c_longlong if is_64bit else LONG
    ULONG_PTR = c_ulonglong if is_64bit else DWORD
    UINT_PTR = c_ulonglong if is_64bit else c_uint
    SIZE_T = ULONG_PTR
    POINTER_TYPE = ULONG_PTR
    LP_POINTER_TYPE = POINTER(POINTER_TYPE)
    FARPROC = CFUNCTYPE(None)
    PFARPROC = POINTER(FARPROC)
    c_uchar_p = POINTER(c_ubyte)
    c_ushort_p = POINTER(c_ushort)

    _kernel32 = ctypes.windll.kernel32

    VirtualAlloc = _kernel32.VirtualAlloc
    VirtualAlloc.restype = LPVOID
    VirtualAlloc.argtypes = [LPVOID, SIZE_T, DWORD, DWORD]

    RtlMoveMemory = _kernel32.RtlMoveMemory
    RtlMoveMemory.restype = None
    RtlMoveMemory.argtypes = [LPVOID, LPVOID, SIZE_T]

    RtlFillMemory = _kernel32.RtlFillMemory
    RtlFillMemory.restype = None
    RtlFillMemory.argtypes = [LPVOID, SIZE_T, INT]

    VirtualProtect = _kernel32.VirtualProtect
    VirtualProtect.restype = BOOL
    VirtualProtect.argtypes = [LPVOID, SIZE_T, DWORD, PDWORD]

    TlsAlloc = _kernel32.TlsAlloc
    TlsAlloc.restype = DWORD
    TlsAlloc.argtypes = []

    TlsSetValue = _kernel32.TlsSetValue
    TlsSetValue.restype = BOOL
    TlsSetValue.argtypes = [DWORD, LPVOID]

    TlsGetValue = _kernel32.TlsGetValue
    TlsGetValue.restype = LPVOID
    TlsGetValue.argtypes = [DWORD]
elif os.name == "posix":
    dll = CDLL(None)
    assert(hasattr(dll, "mmap"))
    assert(hasattr(dll, "munmap"))
    assert(hasattr(dll, "mprotect"))
    assert(hasattr(dll, "memcpy"))

    is_64bit = sizeof(c_void_p) == sizeof(c_ulonglong)

    c_off_t = c_size_t

    # void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
    mmap = dll.mmap
    mmap.restype = c_void_p
    mmap.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_off_t]

    # int mprotect(void *addr, size_t len, int prot)
    mprotect = dll.mprotect
    mprotect.restype = c_int
    mprotect.argtypes = [c_void_p, c_size_t, c_int]

    # int munmap(void *addr, size_t len)
    munmap = dll.munmap
    munmap.restype = c_int
    munmap.argtypes = [c_void_p, c_size_t]

    # void *memcpy(void *dest, const void *src, size_t n)
    memcpy = dll.memcpy
    memcpy.restype = c_void_p
    memcpy.argtypes = [c_void_p, c_void_p, c_size_t]

    # void *memset(void *s, int c, size_t n)
    memset = dll.memset
    memset.restype = c_void_p
    memset.argtypes = [c_void_p, c_int, c_size_t]
else:
    raise Exception(
        'The Operating system "{}" is not supporting'.format(os.name))


class SimpleCompiler:
    def __init__(self, target_machine):
        self.target_machine = target_machine

    def compile(self, module):
        pass_manager = PassManager()

        module.mfuncs = {}
        for func in module.funcs.values():
            target_info = self.target_machine.get_target_info(func)
            module.mfuncs[func] = MachineFunction(target_info, func)

        # Setup code generation stream
        objformat = self.target_machine.triple.objformat
        if objformat == ObjectFormatType.COFF:
            obj_file_info = COFFObjectFileInfo()
        elif objformat == ObjectFormatType.ELF:
            obj_file_info = X64ELFTargetObjectFile()

        ctx = MCContext(obj_file_info)
        obj_file_info.ctx = ctx

        regalloc = LinearScanRegisterAllocation()

        pass_manager.passes.extend([
            UnreachableBlockElim(),
            Mem2Reg(),
            SimplifyCFG(),
            InstructionSelection(),
            UnreachableBBElim(),
            PeepholeOptimize(),
            TwoAddressInst(),
            DeadMachineCodeElim(),
            LiveIntervals(),
            regalloc,
            regalloc,
            VirtualRegisterRewriter(),
            # MachineCopyProp(),
            ExpandPseudosPostRA(),
            PrologEpilogInsertion(),
        ])

        with BytesIO() as code_output:
            self.target_machine.add_mc_emit_passes(
                pass_manager, ctx, code_output, False)

            pass_manager.run(module)
            obj_data = code_output.getvalue()

        return obj_data

    def __call__(self, module):
        return self.compile(module)


from rachetwrench.jit.mem_mgr import MemoryManager, AllocationPurpose


class Win64MemoryManager(MemoryManager):
    def __init__(self):
        self.memories = []

    def allocate_mapped_mem(self, size, alignment, purpose):
        import ctypes

        if size == 0:
            return None

        MEM_COMMIT = 0x1000
        MEM_RESERVE = 0x2000
        MEM_RESET = 0x00080000
        MEM_RESET_UNDO = 0x1000000
        MEM_LARGE_PAGES = 0x20000000
        MEM_PHYSICAL = 0x00100000
        MEM_WRITE_WATCH = 0x00200000

        PAGE_EXECUTE = 0x10
        PAGE_EXECUTE_READ = 0x20
        PAGE_EXECUTE_READWRITE = 0x40
        PAGE_EXECUTE_WRITECOPY = 0x80
        PAGE_NOACCESS = 0x01
        PAGE_READONLY = 0x02
        PAGE_READWRITE = 0x04
        PAGE_WRITECOPY = 0x08

        if purpose == AllocationPurpose.Code:
            protect = PAGE_EXECUTE_READWRITE
        elif purpose == AllocationPurpose.ROData:
            protect = PAGE_READWRITE
        else:
            assert(purpose == AllocationPurpose.RWData)
            protect = PAGE_READWRITE

        ptr = VirtualAlloc(LPVOID(0),
                           SIZE_T(size),
                           DWORD(MEM_COMMIT | MEM_RESERVE),
                           DWORD(protect))

        assert(ptr % alignment == 0)

        self.memories.append(ptr)

        return ptr

    def copy_to_mem(self, ptr, buffer):
        import ctypes

        buf = (ctypes.c_char * len(buffer)).from_buffer(bytearray(buffer))
        size = len(buffer)

        RtlMoveMemory(LPVOID(ptr), buf, SIZE_T(size))

    def copy_from_mem(self, ptr, buffer):
        import ctypes

        buf = (ctypes.c_char * len(buffer)).from_buffer(bytearray(buffer))
        size = len(buffer)

        RtlMoveMemory(buf, LPVOID(ptr), SIZE_T(size))

    def fill_to_mem(self, ptr, size, value):
        RtlFillMemory(LPVOID(ptr), SIZE_T(size), INT(value))

    def allocate_data(self, size, alignment, is_readonly):
        if is_readonly:
            return self.allocate_mapped_mem(size, alignment, AllocationPurpose.ROData)

        return self.allocate_mapped_mem(size, alignment, AllocationPurpose.RWData)

    def allocate_code(self, size, alignment):
        return self.allocate_mapped_mem(size, alignment, AllocationPurpose.Code)

    def enable_code_protect(self, ptr, size):
        PAGE_EXECUTE_READ = 0x20
        old_protect = DWORD()
        result = VirtualProtect(LPVOID(ptr), SIZE_T(
            size), DWORD(PAGE_EXECUTE_READ), old_protect)
        return result

    def disable_code_protect(self, ptr, size):
        PAGE_EXECUTE_READWRITE = 0x40
        old_protect = DWORD()
        result = VirtualProtect(LPVOID(ptr), SIZE_T(
            size), DWORD(PAGE_EXECUTE_READWRITE), old_protect)
        return result


class PosixMemoryManager:
    def __init__(self):
        self.memories = []

    def allocate_mapped_mem(self, size, alignment, purpose):
        import ctypes

        if size == 0:
            return None

        # Protections are chosen from these bits, or-ed together

        PROT_NONE = 0x00  # no permissions
        PROT_READ = 0x01  # pages can be read
        PROT_WRITE = 0x02  # pages can be written
        PROT_EXEC = 0x04  # pages can be executed

        # Mapping type
        MAP_FILE = 0x0000  # mapped from a file or device
        MAP_ANON = 0x1000  # allocated from memory, swap space

        # Sharing types; choose one
        MAP_SHARED = 0x0001  # share changes
        MAP_PRIVATE = 0x0002  # changes are private

        # Other flags
        MAP_FIXED = 0x0010  # map addr must be exactly as requested

        # Advice to madvise
        MADV_NORMAL = 0  # no further special treatment
        MADV_RANDOM = 1  # expect random page references
        MADV_SEQUENTIAL = 2  # expect sequential page references
        MADV_WILLNEED = 3  # will need these pages
        MADV_DONTNEED = 4  # dont need these pages

        if purpose == AllocationPurpose.Code:
            protect = PROT_READ | PROT_WRITE | PROT_EXEC
        elif purpose == AllocationPurpose.ROData:
            protect = PROT_READ | PROT_WRITE
        else:
            assert(purpose == AllocationPurpose.RWData)
            protect = PROT_READ | PROT_WRITE

        from mmap import MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_WRITE, PROT_READ

        ptr = mmap(0, size, protect, MAP_PRIVATE | MAP_ANON, -1, 0)

        assert(ptr != 0xFFFFFFFF)

        assert(ptr % alignment == 0)

        self.memories.append(ptr)

        return ptr

    def copy_to_mem(self, ptr, buffer):
        import ctypes

        buf = (ctypes.c_char * len(buffer)).from_buffer(bytearray(buffer))
        size = len(buffer)

        result = memcpy(c_void_p(ptr), buf, c_size_t(size))

        assert(result != 0xFFFFFFFF)

    def copy_from_mem(self, ptr, buffer):
        import ctypes

        assert(isinstance(buffer, bytearray))

        buf = (ctypes.c_char * len(buffer)).from_buffer(buffer)
        size = len(buffer)

        result = memcpy(buf, c_void_p(ptr), c_size_t(size))

        assert(result != 0xFFFFFFFF)

    def fill_to_mem(self, ptr, size, value):
        result = memset(c_void_p(ptr), c_int(value), c_size_t(size))

        assert(result != 0xFFFFFFFF)

    def allocate_data(self, size, alignment, is_readonly):
        if is_readonly:
            return self.allocate_mapped_mem(size, alignment, AllocationPurpose.ROData)

        return self.allocate_mapped_mem(size, alignment, AllocationPurpose.RWData)

    def allocate_code(self, size, alignment):
        return self.allocate_mapped_mem(size, alignment, AllocationPurpose.Code)

    def enable_code_protect(self, ptr, size):
        from mmap import MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_WRITE, PROT_READ

        result = mprotect(ptr, size, PROT_READ | PROT_EXEC)
        assert(result != 0xFFFFFFFF)

        return result

    def disable_code_protect(self, ptr, size):
        from mmap import MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_WRITE, PROT_READ

        result = mprotect(ptr, size, PROT_READ | PROT_WRITE | PROT_EXEC)
        assert(result != 0xFFFFFFFF)

        return result


class DynamicLinker:
    def __init__(self):

        if os.name == 'nt':
            self.mem_mgr = Win64MemoryManager()
        elif os.name == 'posix':
            self.mem_mgr = PosixMemoryManager()
        else:
            raise Exception("Not supporting platform")

    def load(self, obj):
        raise NotImplementedError()


class DynamicLinkerELF(DynamicLinker):
    def __init__(self):
        super().__init__()


class RelocationEntry:
    def __init__(self, section, offset, rel_type, addend):
        assert(isinstance(section, SectionEntry))
        self.section = section
        self.offset = offset
        self.rel_type = rel_type
        self.addend = addend


class SectionEntry:
    def __init__(self, name, addr, size, alloc_size):
        self.name = name
        self.addr = addr
        self.size = size
        self.alloc_size = alloc_size
        self.stub_offset = size


class SymbolTableEntry:
    def __init__(self, section, offset, flags):
        self.section = section
        self.offset = offset
        self.flags = flags


def is_section_readonly(section_ref):
    if section_ref.obj.is_coff:
        from rachetwrench.codegen.coff import IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE
        characteristics = section_ref.item.value.characteristics.value
        return characteristics & (IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE) == (IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ)
    elif section_ref.obj.is_elf:
        from rachetwrench.obj.elf import SHF_WRITE, SHF_EXECINSTR
        return section_ref.flags & (SHF_WRITE | SHF_EXECINSTR) != 0

    raise ValueError()


def is_section_zero_init(section_ref):
    if section_ref.obj.is_coff:
        from rachetwrench.codegen.coff import IMAGE_SCN_CNT_UNINITIALIZED_DATA
        characteristics = section_ref.item.value.characteristics.value
        return characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0
    elif section_ref.obj.is_elf:
        from rachetwrench.obj.elf import SHT_NOBITS
        return section_ref.type == SHT_NOBITS

    raise ValueError()


class RelocationValueRef:
    def __init__(self, section, offset, addend):
        self.section = section
        self.offset = offset
        self.addend = addend
        self.symbol_name = None

    def __eq__(self, other):
        if not isinstance(other, RelocationValueRef):
            return False

        return self.section == other.section and self.offset == other.offset and self.addend == other.addend and self.symbol_name == other.symbol_name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.section, self.offset, self.addend, self.symbol_name))


class DynamicLinkerELF(DynamicLinker):
    def __init__(self, symbol_solver):
        super().__init__()
        self.symbol_solver = symbol_solver
        self.section_mems = {}
        self.relocations = {}
        self.global_symbol_table = {}
        self.external_symbol_relocations = {}
        self.got_section = None
        self.got_entry_count = 0
        self.got_entry_size = 8

    def allocate_got_entries(self, count):
        if not self.got_section:
            self.got_section = SectionEntry(".got", 0, 0, 0)

        offset = self.got_entry_count * self.got_entry_size
        self.got_entry_count += count
        return offset

    def load(self, obj):
        self.arch = obj.arch

        for symbol in obj.symbols:
            flags = symbol.flags

            if flags & SymbolFlags.Undefined != SymbolFlags.Non:
                continue

            sym_type = symbol.ty
            sym_name = symbol.name
            jit_symbol_flags = 0

            if (flags & SymbolFlags.Absolute) and (sym_type != SymbolType.File):
                raise NotImplementedError()
            elif sym_type in [SymbolType.Function, SymbolType.Data, SymbolType.Unknown, SymbolType.Other]:
                section = symbol.section
                if section is None:
                    continue

                offset = symbol.address - section.address

                is_code = section.is_text
                jit_section = self.get_or_emit_section(obj, section, is_code)

                self.global_symbol_table[sym_name] = SymbolTableEntry(
                    jit_section, offset, jit_symbol_flags)

        stubs = {}

        for section in obj.sections:
            for reloc in section.relocations:
                self.process_relocation(
                    section.relocated_section, reloc, obj, stubs)

        for section in obj.sections:
            if section in self.section_mems:
                continue

            is_code = section.is_text

            self.get_or_emit_section(obj, section, is_code)

        self.finalize()

    def read_bytes_as_int(self, obj, offset, size):
        return int.from_bytes(obj[offset:(offset+size)], "little")

    def read_bytes_as_uint(self, obj, offset, size):
        return int.from_bytes(obj[offset:(offset+size)], "little", signed=False)

    def emit_stub_function(self, address):
        import struct

        if self.arch == ArchType.ARM:
            self.mem_mgr.copy_to_mem(address, struct.pack(
                "<I", 0xe51ff004))  # ldr pc, [pc, #-4]

            return address + 4

        if self.arch == ArchType.X86_64:
            self.mem_mgr.copy_to_mem(address, struct.pack("<B", 0xFF))  # jmp
            self.mem_mgr.copy_to_mem(
                address + 1, struct.pack("<B", 0x25))  # rip

            return address

        raise ValueError("Unknown architecture")

    def add_relocation_for_symbol(self, reloc_entry: RelocationEntry, symbol_name):
        is_extern = symbol_name not in self.global_symbol_table

        if is_extern:
            if symbol_name not in self.external_symbol_relocations:
                self.external_symbol_relocations[symbol_name] = []

            self.external_symbol_relocations[symbol_name].append(reloc_entry)
        else:
            symbol_info = self.global_symbol_table[symbol_name]
            section = symbol_info.section

            section_id = self.section_mems[section]

            if section not in self.relocations:
                self.relocations[section_id] = []

            offset = reloc_entry.offset
            rel_type = reloc_entry.rel_type

            # The symbol value is the offset from the belonging section
            self.relocations[section_id].append(RelocationEntry(
                reloc_entry.section, offset, rel_type, symbol_info.offset + reloc_entry.addend))

    def add_relocation_for_section(self, reloc_entry, section_id):
        assert(isinstance(section_id, SectionEntry))

        if section_id not in self.relocations:
            self.relocations[section_id] = []

        self.relocations[section_id].append(reloc_entry)

    def process_simple_relocation(self, target_section, offset, rel_type, value):
        symbol_name = value.symbol_name

        reloc_entry = RelocationEntry(
            self.section_mems[target_section], offset, rel_type, value.addend)
        if not symbol_name:
            section_id = self.section_mems[value.section]

            self.add_relocation_for_section(reloc_entry, section_id)
            return

        self.add_relocation_for_symbol(reloc_entry, symbol_name)

    def process_relocation(self, target_section, reloc, obj, stubs):
        symbol = reloc.symbol
        section = symbol.section

        is_extern = section == None

        rel_type = reloc.ty
        offset = reloc.offset
        addend = reloc.addend

        if symbol.name in self.global_symbol_table:
            global_symbol = self.global_symbol_table[symbol.name]

            value = RelocationValueRef(
                section, global_symbol.offset, global_symbol.offset + addend)
        else:
            value = RelocationValueRef(None, 0, addend)
            value.symbol_name = symbol.name

        if self.arch == ArchType.ARM:
            if rel_type in [R_ARM_PC24, R_ARM_CALL, R_ARM_JUMP24]:
                if value in stubs:
                    stub_offset = stubs[value]
                    addr = self.section_mems[target_section].addr + stub_offset
                    self.resolve_relocation(
                        addr, RelocationEntry(self.section_mems[target_section], offset, rel_type, 0))
                else:
                    section_entry = self.section_mems[target_section]
                    stub_offset = section_entry.stub_offset
                    stubs[value] = stub_offset
                    stub_addr = self.emit_stub_function(
                        section_entry.addr + stub_offset)

                    reloc_entry = RelocationEntry(
                        self.section_mems[target_section], stub_addr - section_entry.addr, R_ARM_ABS32, value.addend)

                    symbol_name = value.symbol_name
                    if symbol_name:
                        self.add_relocation_for_symbol(
                            reloc_entry, symbol_name)
                    else:
                        assert(value.section)
                        section_id = value.section

                        self.add_relocation_for_section(
                            reloc_entry, section_id)

                    addr = self.section_mems[target_section].addr + stub_offset
                    self.resolve_relocation(
                        addr, RelocationEntry(self.section_mems[target_section], offset, rel_type, 0))

                    section_entry.stub_offset += self.max_stub_size
            else:
                addr = self.section_mems[target_section].addr + offset
                if rel_type in [R_ARM_PREL31, R_ARM_TARGET1, R_ARM_ABS32]:
                    buf = bytearray(4)
                    self.mem_mgr.copy_from_mem(addr, buf)
                    addr_val = self.read_bytes_as_uint(buf, 0, 4)
                    value.addend += addr_val
                elif rel_type in [R_ARM_MOVW_ABS_NC, R_ARM_MOVT_ABS]:
                    buf = bytearray(4)
                    self.mem_mgr.copy_from_mem(addr, buf)
                    addr_val = self.read_bytes_as_uint(buf, 0, 4)
                    addr_val = (addr_val & 0xFFF) | (
                        ((addr_val >> 16) & 0xF) << 12)
                    value.addend += addr_val
                else:
                    raise NotImplementedError()

                self.process_simple_relocation(
                    target_section, offset, rel_type, value)

        elif self.arch == ArchType.X86_64:
            if rel_type in [R_X86_64_TLSGD]:
                raise NotImplementedError()
                # self.resolve_relocation(
                #     tls_gd_ptr, RelocationEntry(self.section_mems[target_section], offset, R_X86_64_PC32, -4))
            elif rel_type in [R_X86_64_PLT32]:
                if value.symbol_name:
                    if value in stubs:
                        stub_offset = stubs[value]
                        stub_addr = self.section_mems[target_section].addr + stub_offset
                    else:
                        section_entry = self.section_mems[target_section]
                        stub_addr = section_entry.addr + section_entry.stub_offset
                        stub_offset = stub_addr - section_entry.addr
                        stubs[value] = stub_offset
                        stub_addr = self.emit_stub_function(stub_addr)

                        section_entry.stub_offset += self.max_stub_size

                        got_offset = self.allocate_got_entries(1)

                        got_reloc_entry = RelocationEntry(
                            self.section_mems[target_section], stub_offset + 2, R_X86_64_PC32, got_offset - 4)
                        self.add_relocation_for_section(
                            got_reloc_entry, self.got_section)

                        # Relocate address in GOT entry to the symbol's address.
                        reloc_entry = RelocationEntry(
                            self.got_section, got_offset, R_X86_64_64, 0)

                        symbol_name = value.symbol_name
                        self.add_relocation_for_symbol(
                            reloc_entry, symbol_name)

                    self.resolve_relocation(
                        stub_addr, RelocationEntry(self.section_mems[target_section], offset, R_X86_64_PC32, addend))
                else:
                    raise NotImplementedError()
            else:
                self.process_simple_relocation(
                    target_section, offset, rel_type, value)

        else:
            raise ValueError("Invalid architecture")

    @property
    def max_stub_size(self):
        return 8

    def relocation_needs_stub(self, reloc: RelocationRef):
        return reloc.ty in [R_ARM_PC24, R_ARM_CALL, R_ARM_JUMP24]

    def compute_section_stub_buf_size(self, obj, section):
        stub_size = self.max_stub_size

        stub_buf_size = 0
        for rel_section in obj.sections:
            if rel_section.relocated_section != section:
                continue

            for reloc in rel_section.relocations:
                if self.relocation_needs_stub(reloc):
                    stub_buf_size += stub_size

        return stub_buf_size

    def emit_section(self, obj, section, is_code):
        alignment = section.alignment
        is_readonly = is_section_readonly(section)
        is_zero_init = is_section_zero_init(section)
        is_virtual = section.is_virtual
        name = section.name
        size = section.size
        alloc_size = size

        alloc_size += self.compute_section_stub_buf_size(obj, section)

        if is_virtual or is_zero_init:
            data = bytes()
        else:
            data = section.contents
            assert(len(data) >= size)

        if is_code:
            addr = self.mem_mgr.allocate_code(alloc_size, alignment)
        else:
            addr = self.mem_mgr.allocate_data(
                alloc_size, alignment, is_readonly)

        if is_virtual or is_zero_init:
            self.mem_mgr.fill_to_mem(addr, size, 0)
        else:
            self.mem_mgr.copy_to_mem(addr, data)

        return SectionEntry(name, addr, size, alloc_size)

    def get_or_emit_section(self, obj, section, is_code):
        assert(isinstance(section, SectionRef))

        if section in self.section_mems:
            return self.section_mems[section]

        emitted = self.emit_section(obj, section, is_code)
        self.section_mems[section] = emitted
        return emitted

    def resolve_relocation_arm(self, value, reloc):
        import struct

        target_section_mem = reloc.section
        target = target_section_mem.addr + reloc.offset
        value += reloc.addend
        rel_type = reloc.rel_type

        if rel_type == R_ARM_PREL31:
            value = value - target
            assert((value & ~0x80000000) == value)

            buf = bytearray(4)
            self.mem_mgr.copy_from_mem(target, buf)
            data = self.read_bytes_as_uint(buf, 0, 4)
            data = (data & 0x80000000) | value

            bys = struct.pack("<I", data)
            self.mem_mgr.copy_to_mem(target, bys)
        elif rel_type in [R_ARM_ABS32, R_ARM_TARGET1]:
            assert((value & 0xFFFFFFFF) == value)
            bys = struct.pack("<I", value)
            self.mem_mgr.copy_to_mem(target, bys)
        elif rel_type in [R_ARM_MOVW_ABS_NC, R_ARM_MOVT_ABS]:
            if rel_type == R_ARM_MOVW_ABS_NC:
                value = value & 0xFFFF
            else:
                value = (value >> 16) & 0xFFFF

            buf = bytearray(4)
            self.mem_mgr.copy_from_mem(target, buf)
            data = self.read_bytes_as_uint(buf, 0, 4)
            data = (data & ~0x000F0FFF) | (value & 0xFFF) | (
                ((value >> 12) & 0xF) << 16)

            bys = struct.pack("<I", data)
            self.mem_mgr.copy_to_mem(target, bys)
        elif rel_type in [R_ARM_PC24, R_ARM_CALL, R_ARM_JUMP24]:
            rel_value = value - target - 8
            rel_value = (rel_value & 0x03FFFFFC) >> 2

            assert((rel_value & ~0xFF000000) == rel_value)

            buf = bytearray(4)
            self.mem_mgr.copy_from_mem(target, buf)
            data = self.read_bytes_as_uint(buf, 0, 4)

            assert((data & ~0xFF000000) == 0xFFFFFE)

            data = (data & 0xFF000000) | rel_value

            bys = struct.pack("<I", data)
            self.mem_mgr.copy_to_mem(target, bys)
        else:
            raise ValueError()

    def resolve_relocation_x86_64(self, value, reloc):
        import struct

        target_section_mem = reloc.section
        target = target_section_mem.addr + reloc.offset
        value += reloc.addend
        rel_type = reloc.rel_type

        if rel_type == R_X86_64_PC32:
            value = value - target
            # assert(-0x80000000 <= value and value < 0x80000000)
            value = value & 0xFFFFFFFF

            bys = struct.pack("<I", value)
            self.mem_mgr.copy_to_mem(target, bys)
        elif rel_type == R_X86_64_64:
            bys = struct.pack("<Q", value)
            self.mem_mgr.copy_to_mem(target, bys)
        else:
            raise ValueError()

    def resolve_relocation(self, value, reloc):
        import struct

        if self.arch == ArchType.ARM:
            self.resolve_relocation_arm(value, reloc)
        elif self.arch == ArchType.X86_64:
            self.resolve_relocation_x86_64(value, reloc)
        else:
            raise ValueError()

    def resolve_section_relocations(self, section_id, relocations):
        assert(isinstance(section_id, SectionEntry))

        value = section_id.addr
        self.resolve_relocation_list(value, relocations)

    def resolve_relocation_list(self, value, relocations):
        for reloc in relocations:
            self.resolve_relocation(value, reloc)

    def resolve_relocations(self):
        for section, reloc in self.relocations.items():
            self.resolve_section_relocations(section, reloc)

    def apply_external_symbol_relocations(self, symbol_solver):
        for symbol_name, reloc in self.external_symbol_relocations.items():
            symbol_ptr = symbol_solver.lookup_symbol(symbol_name)
            self.resolve_relocation_list(symbol_ptr, reloc)

    def finalize(self):
        if self.got_section:
            got_section_size = self.got_entry_count * self.got_entry_size

            addr = self.mem_mgr.allocate_data(
                got_section_size, self.got_entry_size, False)

            self.got_section.addr = addr
            self.got_section.size = got_section_size
            self.got_section.alloc_size = got_section_size

            self.mem_mgr.fill_to_mem(addr, got_section_size, 0)

        self.resolve_relocations()

        self.apply_external_symbol_relocations(self.symbol_solver)

        for section, section_mem in self.section_mems.items():
            if section.is_text:
                self.mem_mgr.enable_code_protect(
                    section_mem.addr, section_mem.size)

    def get_symbol(self, name):
        if name in self.global_symbol_table:
            symbol = self.global_symbol_table[name]
            return symbol.section.addr + symbol.offset

        return None


class DynamicLinkerCOFF(DynamicLinker):
    def __init__(self, symbol_solver):
        super().__init__()
        self.symbol_solver = symbol_solver
        self.section_mems = {}
        self.relocations = {}
        self.global_symbol_table = {}
        self.external_symbol_relocations = {}

    def load(self, obj):
        for symbol in obj.symbols:
            flags = symbol.flags

            if flags & SymbolFlags.Undefined != SymbolFlags.Non:
                continue

            sym_type = symbol.ty
            sym_name = symbol.name
            jit_symbol_flags = 0

            if (flags & SymbolFlags.Absolute) and (sym_type != SymbolType.File):
                raise NotImplementedError()
            elif sym_type in [SymbolType.Function, SymbolType.Data, SymbolType.Unknown, SymbolType.Other]:
                section = symbol.section
                if section is None:
                    continue

                offset = symbol.address - section.address

                is_code = section.is_text
                jit_section = self.get_or_emit_section(obj, section, is_code)

                self.global_symbol_table[sym_name] = SymbolTableEntry(
                    jit_section, offset, jit_symbol_flags)

        stubs = {}

        for section in obj.sections:
            for reloc in section.relocations:
                self.process_relocation(section, reloc, obj, stubs)

        for section in obj.sections:
            if section in self.section_mems:
                continue

            is_code = section.is_text

            self.get_or_emit_section(obj, section, is_code)

        self.finalize()

    def read_bytes_as_int(self, obj, offset, size):
        return int.from_bytes(obj[offset:(offset+size)], "little")

    def emit_stub_function(self, address):
        import struct

        self.mem_mgr.copy_to_mem(address, struct.pack("<B", 0xFF))  # jmp
        self.mem_mgr.copy_to_mem(address + 1, struct.pack("<B", 0x25))  # rip
        self.mem_mgr.copy_to_mem(address + 2, struct.pack("<I", 0))  # 0
        # 8 bytes absolute address to the target function

    def generate_relocation_stub(self, target_section, offset, rel_type, addend, stubs):
        section_entry = self.section_mems[target_section]

        entry_to_stub = RelocationEntry(
            section_entry, offset, rel_type, addend)

        if entry_to_stub in stubs:
            stub_offset = stubs[entry_to_stub]
        else:
            stub_offset = section_entry.stub_offset
            stubs[entry_to_stub] = stub_offset
            self.emit_stub_function(section_entry.addr + stub_offset)
            section_entry.stub_offset += self.max_stub_size

        stub_addr = section_entry.addr + stub_offset

        self.resolve_relocation(stub_addr, entry_to_stub)

        addend = 0
        offset = stub_offset + 6
        rel_type = IMAGE_REL_AMD64_ADDR64
        return (offset, rel_type, addend)

    def process_relocation(self, target_section, reloc, obj, stubs):
        symbol = reloc.symbol
        section = symbol.section

        is_extern = section == None

        rel_type = reloc.ty
        offset = reloc.offset
        addend = 0

        obj_target = target_section.item.value.pointer_to_raw_data.value + offset

        if rel_type in [
                IMAGE_REL_AMD64_REL32,
                IMAGE_REL_AMD64_REL32_1,
                IMAGE_REL_AMD64_REL32_2,
                IMAGE_REL_AMD64_REL32_3,
                IMAGE_REL_AMD64_REL32_4,
                IMAGE_REL_AMD64_REL32_5,
                IMAGE_REL_AMD64_ADDR32NB]:
            addend = self.read_bytes_as_int(obj.data, obj_target, 4)
            opcode = self.read_bytes_as_int(obj.data, obj_target - 1, 1)
            if opcode in [0xEB, 0xE9, 0xE8] and is_extern:
                offset, rel_type, addend = self.generate_relocation_stub(
                    target_section, offset, rel_type, addend, stubs)
        elif rel_type == IMAGE_REL_AMD64_ADDR64:
            addend = self.read_bytes_as_int(obj.data, obj_target, 8)

        if is_extern:
            if symbol.name not in self.external_symbol_relocations:
                self.external_symbol_relocations[symbol.name] = []

            self.external_symbol_relocations[symbol.name].append(
                RelocationEntry(self.section_mems[target_section], offset, rel_type, addend))
        else:
            self.get_or_emit_section(obj, section, section.is_text)

            section_id = self.section_mems[section]

            if section_id not in self.relocations:
                self.relocations[section_id] = []

            # The symbol value is the offset from the belonging section
            target_offset = symbol.value
            self.relocations[section_id].append(RelocationEntry(
                self.section_mems[target_section], offset, rel_type, target_offset + addend))

    @property
    def max_stub_size(self):
        return 14

    def relocation_needs_stub(self, reloc):
        return True

    def compute_section_stub_buf_size(self, obj, section):
        stub_size = self.max_stub_size

        stub_buf_size = 0
        for reloc in section.relocations:
            if self.relocation_needs_stub(reloc):
                stub_buf_size += stub_size

        return stub_buf_size

    def emit_section(self, obj, section, is_code):
        alignment = section.alignment
        is_readonly = is_section_readonly(section)
        is_zero_init = is_section_zero_init(section)
        is_virtual = section.is_virtual
        name = section.name
        size = section.size
        alloc_size = size

        alloc_size += self.compute_section_stub_buf_size(obj, section)

        if is_virtual or is_zero_init:
            data = bytes()
        else:
            data = section.contents
            assert(len(data) >= size)

        if is_code:
            addr = self.mem_mgr.allocate_code(alloc_size, alignment)
        else:
            addr = self.mem_mgr.allocate_data(
                alloc_size, alignment, is_readonly)

        if is_virtual or is_zero_init:
            self.mem_mgr.fill_to_mem(addr, size, 0)
        else:
            self.mem_mgr.copy_to_mem(addr, data)

        return SectionEntry(name, addr, size, alloc_size)

    def get_or_emit_section(self, obj, section, is_code):
        import struct

        if section in self.section_mems:
            return self.section_mems[section]

        emitted = self.emit_section(obj, section, is_code)
        self.section_mems[section] = emitted

        if section.name == ".tls$":
            addr = self.mem_mgr.allocate_data(
                8, 8, False)
            self.tls_index_addr = addr
            self.tls_index = TlsAlloc()
            tls_index_bys = struct.pack("<I", self.tls_index)
            self.mem_mgr.copy_to_mem(self.tls_index_addr, tls_index_bys)
            p_tls_data = TlsGetValue(self.tls_index)
            if not p_tls_data:
                if TlsSetValue(self.tls_index, emitted.addr) != 1:
                    raise RuntimeError("Failed to set tls index.")

        return emitted

    def resolve_relocation(self, value, reloc):
        import struct

        target_section_mem = reloc.section
        target = target_section_mem.addr + reloc.offset

        if reloc.rel_type in [
                IMAGE_REL_AMD64_REL32,
                IMAGE_REL_AMD64_REL32_1,
                IMAGE_REL_AMD64_REL32_2,
                IMAGE_REL_AMD64_REL32_3,
                IMAGE_REL_AMD64_REL32_4,
                IMAGE_REL_AMD64_REL32_5]:

            def to_signed(value):
                return (value & ((1 << 31) - 1)) - (value & (1 << 31))

            delta = 4 + (reloc.rel_type - IMAGE_REL_AMD64_REL32)
            addend = to_signed(reloc.addend)
            offset_to_rel_section = value - (target + delta)
            offset_to_rel_symbol = offset_to_rel_section + addend

            bys = struct.pack("<i", offset_to_rel_symbol)
            self.mem_mgr.copy_to_mem(target, bys)
        elif reloc.rel_type == IMAGE_REL_AMD64_ADDR64:
            def to_signed(value):
                return (value & ((1 << 63) - 1)) - (value & (1 << 63))

            addend = to_signed(reloc.addend)
            addr = value + addend
            bys = struct.pack("<Q", addr)
            self.mem_mgr.copy_to_mem(target, bys)
        elif reloc.rel_type == IMAGE_REL_AMD64_SECREL:
            def to_signed(value):
                return (value & ((1 << 31) - 1)) - (value & (1 << 31))

            addend = to_signed(reloc.addend)
            bys = struct.pack("<i", addend)
            self.mem_mgr.copy_to_mem(target, bys)
        else:
            raise ValueError()

    def resolve_section_relocations(self, section, relocations):
        assert(isinstance(section, SectionEntry))
        value = section.addr
        self.resolve_relocation_list(value, relocations)

    def resolve_relocation_list(self, value, relocations):
        for reloc in relocations:
            self.resolve_relocation(value, reloc)

    def resolve_relocations(self):
        for section, reloc in self.relocations.items():
            self.resolve_section_relocations(section, reloc)

    def apply_external_symbol_relocations(self, symbol_solver):
        for symbol_name, reloc in self.external_symbol_relocations.items():
            if symbol_name == "_tls_index":
                self.resolve_relocation_list(self.tls_index_addr, reloc)
                continue

            symbol_ptr = symbol_solver.lookup_symbol(symbol_name)
            self.resolve_relocation_list(symbol_ptr, reloc)

    def finalize(self):
        self.resolve_relocations()

        self.apply_external_symbol_relocations(self.symbol_solver)

        for section, section_mem in self.section_mems.items():
            if section.is_text:
                self.mem_mgr.enable_code_protect(
                    section_mem.addr, section_mem.size)

    def get_symbol(self, name):
        if name in self.global_symbol_table:
            symbol = self.global_symbol_table[name]
            return symbol.section.addr + symbol.offset

        return None


class DynamicLinkerCOFFX86_64(DynamicLinkerCOFF):
    def __init__(self, symbol_solver):
        super().__init__(symbol_solver)


class RuntimeLinkerLoader:
    def __init__(self, symbol_solver):
        self.symbol_solver = symbol_solver
        self.linked_objects = []

    def load(self, obj):
        if obj.is_coff:
            linker = DynamicLinkerCOFFX86_64(self.symbol_solver)
        elif obj.is_elf:
            linker = DynamicLinkerELF(self.symbol_solver)
        else:
            raise ValueError()

        self.linked_objects.append(linker)

        linker.load(obj)

    def emit(self, obj):
        with BytesIO(obj) as code_output:
            objfile = parse_object_file(code_output)

        self.load(objfile)

    def find_symbol(self, name):
        for linked_obj in self.linked_objects:
            sym = linked_obj.get_symbol(name)
            if sym is not None:
                return sym

        return None


class IRCompilerLayer:
    def __init__(self, object_layer, compile_func):
        self.object_layer = object_layer
        self.compile_func = compile_func

    def compile(self, module):
        obj = self.compile_func(module)
        if not obj:
            raise RuntimeError()

        self.object_layer.emit(obj)

    def add_module(self, module):
        obj = self.compile(module)


class SymbolSolver:
    def lookup_symbol(self, symbol_name):
        raise NotImplementedError()


class ExecutionEngine(SymbolSolver):
    def __init__(self, target_machine):
        self.target_machine = target_machine
        self.libs = []
        self.linker = RuntimeLinkerLoader(self)
        compiler = SimpleCompiler(target_machine)
        self.compiler_layer = IRCompilerLayer(self.linker, compiler)

    def load_dll(self, name):
        from ctypes.util import find_library
        import ctypes
        import os

        if os.path.exists(name):
            lib = name
        else:
            lib = find_library(name)

        if not lib:
            raise Exception(f"Can't found the library: {name}.")

        self.libs.append(ctypes.CDLL(lib))

    def lookup_symbol(self, symbol_name):
        import ctypes

        addr = self.linker.find_symbol(symbol_name)
        if addr is not None:
            return addr

        for lib in self.libs:
            try:
                if ctypes.c_void_p.in_dll(lib, symbol_name):
                    func = getattr(lib, symbol_name)
                    func_ptr = ctypes.cast(func, ctypes.c_void_p).value
                    return func_ptr
            except:
                pass

        raise KeyError("Couldn't find the symbol.")

    def add_module(self, module):
        self.compiler_layer.add_module(module)

    def get_global_value_address(self, name):
        return self.get_symbol_address(name)

    def get_function_address(self, name):
        return self.get_symbol_address(name)

    def mangle(self, name):
        return name

    def get_symbol_address(self, name):
        return self.lookup_symbol(self.mangle(name))

    def get_pointer_to_function(self, func):
        return self.get_symbol_address(func.name)
