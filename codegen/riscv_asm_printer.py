#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *
from codegen.mc import *
from codegen.asm_emitter import *
from codegen.assembler import *
from ir.values import *
from codegen.riscv_gen import RISCVMachineOps
import zlib
import io
from codegen.riscv_def import *


def get_bb_symbol(bb: BasicBlock, ctx: MCContext):
    bb_num = bb.number
    func_num = list(bb.func.func_info.func.module.funcs.values()).index(
        bb.func.func_info.func)
    prefix = f"{get_private_global_prefix()}BB"
    return ctx.get_or_create_symbol(f"{prefix}{str(func_num)}_{str(bb_num)}")


def get_global_symbol(g: GlobalValue, ctx: MCContext):
    name = f"{get_global_prefix()}{g.name}"
    return ctx.get_or_create_symbol(name)


def get_pic_label(label_id, func_number, ctx: MCContext):
    return ctx.get_or_create_symbol(f"PC{func_number}_{label_id}")


class RISCVMCExprVarKind(Enum):
    Non = auto()
    Lo = auto()
    Hi = auto()
    PCRelLo = auto()
    PCRelHi = auto()
    TLSGDHi = auto()
    Call = auto()


class RISCVMCExpr(MCTargetExpr):
    def __init__(self, kind, expr):
        super().__init__()

        self.kind = kind
        self.expr = expr

    @property
    def pcrel_hi_fixup(self):
        auipc_sym = self.expr.symbol
        fragment = auipc_sym.fragment

        if not fragment:
            return None

        for fixup in fragment.fixups:
            if fixup.kind not in [RISCVFixupKind.RISCV_PCREL_HI20, RISCVFixupKind.RISCV_TLS_GD_HI20]:
                continue
            if fixup.offset == auipc_sym.offset:
                return fixup

        return None

    def evaluate_expr_as_relocatable(self, layout, fixup: MCFixup):
        result = evaluate_expr_as_relocatable(
            self.expr, layout.asm, layout, fixup)
        if not result:
            return None

        if fixup.kind in [RISCVFixupKind.RISCV_PCREL_LO12_I, RISCVFixupKind.RISCV_PCREL_LO12_S]:
            pcrel_hi_fixup = self.pcrel_hi_fixup

            result_hi = evaluate_expr_as_relocatable(
                pcrel_hi_fixup.value.expr, layout.asm, layout, pcrel_hi_fixup)
            if not result_hi:
                return None

            if result_hi.symbol1.symbol.section == self.expr.symbol.section:
                auipc_offset = layout.get_symbol_offset(result.symbol1.symbol)

                return MCValue(fixup.offset - auipc_offset, result_hi.symbol1, None)

        return result


class RISCVMCInstLower:
    def __init__(self, ctx: MCContext, func, asm_printer):
        self.ctx = ctx
        self.func = func
        self.asm_printer = asm_printer

    def get_symbol(self, operand: MachineOperand):
        if isinstance(operand, MOBasicBlock):
            return get_bb_symbol(operand.mbb, self.ctx)

        elif isinstance(operand, MOGlobalAddress):
            return get_global_symbol(operand.value, self.ctx)

        elif isinstance(operand, MOExternalSymbol):
            return self.ctx.get_or_create_symbol(operand.symbol)

        raise NotImplementedError()

    def lower_symbol_operand(self, operand: MachineOperand, symbol: MCSymbol):
        from codegen.riscv_gen import RISCVOperandFlag

        expr = MCSymbolRefExpr(symbol)

        if operand.target_flags & RISCVOperandFlag.LO.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.Lo, expr)
        elif operand.target_flags & RISCVOperandFlag.HI.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.Hi, expr)
        elif operand.target_flags & RISCVOperandFlag.PCREL_LO.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.PCRelLo, expr)
        elif operand.target_flags & RISCVOperandFlag.PCREL_HI.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.PCRelHi, expr)
        elif operand.target_flags & RISCVOperandFlag.TLS_GD_HI.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.TLSGDHi, expr)
        elif operand.target_flags & RISCVOperandFlag.CALL.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.Call, expr)

        if not operand.is_jti and not operand.is_mbb and operand.offset != 0:
            expr = MCBinaryExpr(MCBinaryOpcode.Add, expr,
                                MCConstantExpr(operand.offset))

        return MCOperandExpr(expr)

    @property
    def target_info(self):
        return self.func.target_info

    def lower_operand(self, inst, operand):
        if isinstance(operand, MOReg):
            if operand.is_implicit:
                return None
            assert(operand.subreg is None)
            return MCOperandReg(operand.reg)
        elif isinstance(operand, MOImm):
            return MCOperandImm(operand.val)
        elif isinstance(operand, (MOBasicBlock, MOGlobalAddress, MOExternalSymbol)):
            symbol = self.get_symbol(operand)
            return self.lower_symbol_operand(operand, symbol)
        elif isinstance(operand, MOConstantPoolIndex):
            symbol = self.asm_printer.get_cp_index_symbol(operand.index)
            return self.lower_symbol_operand(operand, symbol)

        raise NotImplementedError()

    def lower(self, inst):
        mc_inst = MCInst(inst.opcode)
        for operand in inst.operands:
            mc_op = self.lower_operand(inst, operand)
            if mc_op is not None:
                mc_inst.add_operand(mc_op)
        return mc_inst


class RISCVAsmInfo(MCAsmInfo):
    def __init__(self):
        super().__init__()
        self._has_dot_type_dot_size_directive = True

    def get_8bit_data_directive(self):
        return ".byte"

    def get_16bit_data_directive(self):
        return ".value"

    def get_32bit_data_directive(self):
        return ".long"

    def get_64bit_data_directive(self):
        return ".quad"

    @property
    def has_dot_type_dot_size_directive(self):
        return self._has_dot_type_dot_size_directive


class AsmPrinter(MachineFunctionPass):
    def __init__(self, stream: MCStream):
        super().__init__()

        self.stream = stream
        self.ctx = stream.context
        self.asm_info = self.ctx.asm_info


class RISCVAsmPrinter(AsmPrinter):
    def __init__(self, stream: MCStream):
        super().__init__(stream)

    def emit_linkage(self, value):
        symbol = get_global_symbol(value, self.ctx)
        if value.linkage == GlobalLinkage.Global:
            self.stream.emit_symbol_attrib(symbol, MCSymbolAttribute.Global)

    def get_global_value_section_kind(self, value):
        if value.is_thread_local:
            return SectionKind.ThreadData
        return SectionKind.Data

    def get_cp_section_kind(self, entry: MachineConstantPoolEntry):
        ty = entry.value.ty

        size = self.module.data_layout.get_type_alloc_size(ty)

        if size == 4:
            return SectionKind.MergeableConst4
        elif size == 8:
            return SectionKind.MergeableConst8
        elif size == 16:
            return SectionKind.MergeableConst16
        elif size == 32:
            return SectionKind.MergeableConst32
        else:
            return SectionKind.ReadOnly

    @property
    def target_info(self):
        return self.mfunc.target_info

    def get_cp_index_symbol(self, index):
        from codegen.coff import MCSectionCOFF

        if self.target_info.triple.env == Environment.MSVC:
            cp_entry = self.mfunc.constant_pool.constants[index]
            if not cp_entry.is_machine_cp_entry:
                kind = self.get_cp_section_kind(cp_entry)
                value = cp_entry.value
                align = cp_entry.alignment

                sec = self.ctx.obj_file_info.get_section_for_const(
                    kind, value, align)

                if isinstance(sec, MCSectionCOFF):
                    sym = sec.comdat_sym
                    self.stream.emit_symbol_attrib(
                        sym, MCSymbolAttribute.Global)
                    return sym

        func = self.mfunc.func_info.func
        func_num = list(func.module.funcs.values()).index(func)
        name = f"{get_private_global_prefix()}CPI{func_num}_{index}"
        return self.ctx.get_or_create_symbol(name)

    def emit_constant_pool(self):
        cp = self.mfunc.constant_pool
        if len(cp.constants) == 0:
            return

        cp_sections = OrderedDict()
        for idx, entry in enumerate(cp.constants):
            align = entry.alignment
            kind = self.get_cp_section_kind(entry)

            section = self.ctx.obj_file_info.get_section_for_const(
                kind, entry.value, align)

            if section not in cp_sections:
                cp_indices = []
                cp_sections[section] = cp_indices
            else:
                cp_indices = cp_sections[section]

            cp_indices.append(idx)

        for section, cp_indices in cp_sections.items():
            offset = 0
            self.stream.switch_section(section)
            for cp_index in cp_indices:
                symbol = self.get_cp_index_symbol(cp_index)
                self.stream.emit_label(symbol)

                cp_entry = cp.constants[cp_index]
                align = cp_entry.alignment

                aligned_offset = int(int((offset + align - 1) / align) * align)

                self.stream.emit_zeros(aligned_offset - offset)
                offset = aligned_offset

                data_layout = self.module.data_layout
                value_size = data_layout.get_type_alloc_size(cp_entry.value.ty)
                self.emit_global_constant(data_layout, cp_entry.value)

                offset += value_size

    def emit_function_header(self, func):
        self.emit_constant_pool()

        self.stream.switch_section(self.ctx.obj_file_info.text_section)
        self.emit_linkage(func.func_info.func)

        func_symbol = get_global_symbol(func.func_info.func, self.ctx)

        if self.asm_info.has_dot_type_dot_size_directive:
            self.stream.emit_symbol_attrib(
                func_symbol, MCSymbolAttribute.ELF_TypeFunction)

        if isinstance(func_symbol, MCSymbolCOFF):
            from codegen.coff import IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_DTYPE_FUNCTION, SCT_COMPLEX_TYPE_SHIFT

            self.stream.emit_coff_symbol_storage_class(
                func_symbol, IMAGE_SYM_CLASS_EXTERNAL)
            self.stream.emit_coff_symbol_type(
                func_symbol, IMAGE_SYM_DTYPE_FUNCTION << SCT_COMPLEX_TYPE_SHIFT)

        self.stream.emit_label(func_symbol)

        self.current_func_sym = func_symbol

    def emit_function_body_start(self, func):
        pass

    def emit_function_body_end(self, func):
        pass

    def emit_basicblock_start(self, bb):
        self.stream.emit_label(get_bb_symbol(bb, self.ctx))

    def emit_basicblock_end(self, bb):
        pass

    def get_constant_pool_symbol(self, index):
        data_layout = self.func.module.data_layout
        func_num = self.func.module.funcs.index(self.func)
        name = f"CPI{func_num}_{index}"
        return self.ctx.get_or_create_symbol(name)

    def emit_instruction(self, inst: MachineInstruction):
        data_layout = self.func.module.data_layout

        mc_inst_lower = RISCVMCInstLower(self.ctx, inst.mbb.func, self)
        mc_inst = mc_inst_lower.lower(inst)

        for operand in mc_inst.operands:
            if operand.is_expr:
                expr = operand.expr
                if expr.ty == MCExprType.Target:
                    expr = expr.expr

                if expr.ty == MCExprType.SymbolRef:
                    if hasattr(self.stream, "assembler"):
                        self.stream.assembler.register_symbol(expr.symbol)

        self.emit_mc_inst(mc_inst)

    def emit_mc_inst(self, inst):
        self.stream.emit_instruction(inst)

    def emit_function_body(self, func):
        self.emit_function_header(func)
        self.emit_function_body_start(func)

        for bb in func.bbs:
            self.emit_basicblock_start(bb)

            for inst in bb.insts:
                self.emit_instruction(inst)

            self.emit_basicblock_end(bb)

        self.emit_function_body_end(func)

    def process_machine_function(self, mfunc: MachineFunction):
        self.mfunc = mfunc
        self.target_lowering = mfunc.target_info.get_lowering()
        self.target_inst_info = mfunc.target_info.get_inst_info()

        current_func_beg_sym = self.ctx.create_temp_symbol("func_beg")
        self.stream.emit_label(current_func_beg_sym)

        self.emit_function_body(mfunc)

        current_func_end_sym = self.ctx.create_temp_symbol("func_end")
        self.stream.emit_label(current_func_end_sym)

        size_expr = MCBinaryExpr(MCBinaryOpcode.Sub,
                                 MCSymbolRefExpr(current_func_end_sym), MCSymbolRefExpr(current_func_beg_sym))

        if self.asm_info.has_dot_type_dot_size_directive:
            self.stream.emit_elf_size(self.current_func_sym, size_expr)

    def initialize(self):
        self.emit_start_of_asm_file()
        self.stream.init_sections()

    def emit_visibility(self, symbol, visibility, is_definition):
        attr = GlobalVisibility.Default

        attr = None
        if visibility == GlobalVisibility.Hidden:
            if is_definition:
                attr = self.asm_info.hidden_visibility_attr
            else:
                attr = self.asm_info.hidden_decl_visibility_attr
        elif visibility == GlobalVisibility.Protected:
            attr = self.asm_info.protected_visibility_attr

        if attr is not None:
            self.stream.emit_symbol_attrib(symbol, attr)

    def finalize(self):
        for variable in self.module.globals.values():
            self.emit_global_variable(variable)

        for func in self.module.funcs.values():
            if not func.is_declaration:
                continue

            vis = func.visibility
            symbol = get_global_symbol(func, self.ctx)
            self.emit_visibility(symbol, vis, False)

        self.stream.finalize()

    def emit_global_constant_fp(self, constant):
        import struct

        size = self.module.data_layout.get_type_alloc_size(constant.ty)
        if size == 4:
            packed = struct.pack('f', constant.value)
        elif size == 8:
            packed = struct.pack('d', constant.value)
        else:
            raise ValueError("The type of constant is invalid.")

        self.stream.emit_bytes(packed)

    def emit_global_constant_vector(self, data_layout, constant, offset):
        from codegen.types import is_integer_ty

        size, align = data_layout.get_type_size_in_bits(constant.ty.elem_ty)
        if is_integer_ty(constant.ty):
            for value in constant.values:
                self.stream.emit_int_value(value, int(size / 8))
        else:
            for value in constant.values:
                self.emit_global_constant(data_layout, value)

    def emit_global_constant_array(self, data_layout, constant, offset):
        for value in constant.values:
            self.emit_global_constant(data_layout, value)

    def emit_global_constant_struct(self, data_layout: DataLayout, constant: ConstantStruct, offset):
        struct_size = data_layout.get_type_alloc_size(constant.ty)
        for i, field in enumerate(constant.values):
            field_size = data_layout.get_type_alloc_size(field.ty)
            field_offset = data_layout.get_elem_offset(constant.ty, i)
            if i == len(constant.values) - 1:
                next_field_offset = struct_size
            else:
                next_field_offset = data_layout.get_elem_offset(
                    constant.ty, i + 1)

            pad_size = (next_field_offset - field_offset) - field_size

            self.emit_global_constant(data_layout, field)
            self.stream.emit_zeros(pad_size)

    def emit_machine_cp_value(self, data_layout, value):
        from codegen.riscv_gen import RISCVConstantPoolKind, RISCVConstantPoolModifier
        if value.kind == RISCVConstantPoolKind.Value:
            symbol = get_global_symbol(value.value, self.ctx)
        else:
            raise ValueError("Unrecognized constant pool kind.")

        def get_variant_kind_from_modifier(modifier):
            if modifier == RISCVConstantPoolModifier.Non:
                return MCVariantKind.Non
            elif modifier == RISCVConstantPoolModifier.TLSGlobalDesc:
                return MCVariantKind.TLSGD

            raise ValueError("Unrecognized modifier type.")

        size = data_layout.get_type_alloc_size(value.ty)

        expr = MCSymbolRefExpr(
            symbol, get_variant_kind_from_modifier(value.modifier))

        def get_function_number(func):
            return func.module.funcs.index(func)

        if value.pc_offset != 0:
            pc_label = get_pic_label(
                value.label_id, get_function_number(self.func), self.ctx)

            pcrel_expr = MCSymbolRefExpr(pc_label)
            pcrel_expr = MCBinaryExpr(MCBinaryOpcode.Add,
                                      pcrel_expr, MCConstantExpr(value.pc_offset))

            if value.relative:
                dotsym = self.ctx.create_temp_symbol("tmp")
                self.stream.emit_label(dotsym)
                dotexpr = MCSymbolRefExpr(dotsym)
                pcrel_expr = MCBinaryExpr(
                    MCBinaryOpcode.Sub, pcrel_expr, dotexpr)

            expr = MCBinaryExpr(MCBinaryOpcode.Sub, expr, pcrel_expr)

        self.stream.emit_value(expr, size)

    def emit_global_constant(self, data_layout, constant, offset=0):
        size, align = data_layout.get_type_size_in_bits(constant.ty)

        if isinstance(constant, ConstantInt):
            self.stream.emit_int_value(constant.value, int(size / 8))
        elif isinstance(constant, ConstantFP):
            self.emit_global_constant_fp(constant)
        elif isinstance(constant, ConstantVector):
            self.emit_global_constant_vector(data_layout, constant, offset)
        elif isinstance(constant, ConstantArray):
            self.emit_global_constant_array(data_layout, constant, offset)
        elif isinstance(constant, ConstantStruct):
            self.emit_global_constant_struct(data_layout, constant, offset)
        elif isinstance(constant, GlobalValue):
            self.stream.emit_int_value(0, int(size / 8))
        else:
            raise ValueError("Invalid constant type")

    def emit_start_of_asm_file(self):
        self.stream.emit_syntax_directive()

    def emit_alignment(self, align):
        self.stream.emit_value_to_alignment(align)

    def emit_global_variable(self, variable: GlobalValue):
        size, align = self.module.data_layout.get_type_size_in_bits(
            variable.vty)

        kind = self.get_global_value_section_kind(variable)
        if kind == SectionKind.Data:
            self.stream.switch_section(self.ctx.obj_file_info.data_section)
        elif kind == SectionKind.ThreadData:
            self.stream.switch_section(self.ctx.obj_file_info.tls_data_section)
        else:
            raise ValueError("Invalid section kind for the global variable.")

        self.emit_alignment(int(align / 8))

        symbol = get_global_symbol(variable, self.ctx)
        self.stream.emit_label(symbol)

        if self.asm_info.has_dot_type_dot_size_directive:
            self.stream.emit_symbol_attrib(
                symbol, MCSymbolAttribute.ELF_TypeObject)

        self.emit_linkage(variable)

        initializer = variable.initializer
        if initializer is not None:
            data_layout = self.module.data_layout
            self.emit_global_constant(data_layout, initializer)
        else:
            self.stream.emit_zeros(size)

        if self.asm_info.has_dot_type_dot_size_directive:
            self.stream.emit_elf_size(symbol, MCConstantExpr(int(size / 8)))


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
    elif kind in [
            RISCVFixupKind.RISCV_COND_BRANCH, RISCVFixupKind.RISCV_UNCOND_BRANCH,
            RISCVFixupKind.RISCV_UNCOND_BL, RISCVFixupKind.RISCV_PCREL_10, RISCVFixupKind.RISCV_LDST_PCREL_12]:
        return 3
    elif kind in [
            RISCVFixupKind.RISCV_MOVW_LO16, RISCVFixupKind.RISCV_MOVT_HI16]:
        return 4

    raise ValueError("kind")


def is_int_range(value, bits):
    return -(1 << (bits - 1)) <= value and value < (1 << (bits - 1))


class RISCVAsmBackend(MCAsmBackend):
    def __init__(self):
        super().__init__()

    def may_need_relaxation(self, inst: MCInst):
        return False

    def relax_instruction(self, inst: MCInst):
        pass

    def get_fixup_kind_info(self, kind):
        table = {
            RISCVFixupKind.RISCV_HI20: MCFixupKindInfo(
                "RISCV_HI20", 12, 20, MCFixupKindInfoFlag.Non),
            RISCVFixupKind.RISCV_LO12_I: MCFixupKindInfo(
                "RISCV_LO12_I", 20, 12, MCFixupKindInfoFlag.Non),
            RISCVFixupKind.RISCV_LO12_S: MCFixupKindInfo(
                "RISCV_LO12_S", 0, 32, MCFixupKindInfoFlag.Non),
            RISCVFixupKind.RISCV_PCREL_HI20: MCFixupKindInfo(
                "RISCV_PCREL_HI20", 12, 20, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_PCREL_LO12_I: MCFixupKindInfo(
                "RISCV_PCREL_LO12_I", 20, 12, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_PCREL_LO12_S: MCFixupKindInfo(
                "RISCV_PCREL_LO12_S", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_TLS_GD_HI20: MCFixupKindInfo(
                "RISCV_TLS_GD_HI20", 12, 20, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_BRANCH: MCFixupKindInfo(
                "RISCV_BRANCH", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_CALL: MCFixupKindInfo(
                "RISCV_CALL", 0, 64, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_JAL: MCFixupKindInfo(
                "RISCV_JAL", 12, 20, MCFixupKindInfoFlag.IsPCRel)
        }

        if kind in table:
            return table[kind]

        return super().get_fixup_kind_info(kind)

    def is_fixup_kind_pcrel(self, fixup):
        kind = fixup.kind
        return self.get_fixup_kind_info(kind).flags & MCFixupKindInfoFlag.IsPCRel == MCFixupKindInfoFlag.IsPCRel

    def adjust_fixup_value(self, fixup: MCFixup, fixed_value: int):
        kind = fixup.kind

        if kind in [RISCVFixupKind.RISCV_PCREL_HI20, RISCVFixupKind.RISCV_HI20]:
            return ((fixed_value + 0x800) >> 12) & 0xfffff

        if kind in [RISCVFixupKind.RISCV_PCREL_LO12_I, RISCVFixupKind.RISCV_LO12_I]:
            return fixed_value & 0xfff

        if kind in [RISCVFixupKind.RISCV_JAL]:
            assert(is_int_range(fixed_value, 21))
            assert((fixed_value & 1) == 0)

            sign = (fixed_value >> 20) & 0x1
            hi8 = (fixed_value >> 12) & 0xff
            mid1 = (fixed_value >> 11) & 0x1
            lo10 = (fixed_value >> 1) & 0x3ff

            return (sign << 19) | (lo10 << 9) | (mid1 << 8) | hi8

        if kind in [RISCVFixupKind.RISCV_BRANCH]:
            assert(is_int_range(fixed_value, 13))
            assert((fixed_value & 1) == 0)

            sign = (fixed_value >> 12) & 0x1
            hi1 = (fixed_value >> 11) & 0x1
            mid6 = (fixed_value >> 5) & 0x3f
            lo4 = (fixed_value >> 1) & 0xf

            return (sign << 31) | (mid6 << 25) | (lo4 << 8) | (hi1 << 7)

        if kind in [RISCVFixupKind.RISCV_CALL]:
            hi = (fixed_value + 0x800) & 0xfffff000
            lo = fixed_value & 0xfff

            return hi | ((lo << 20) << 32)

        raise NotImplementedError()

    def apply_fixup(self, fixup: MCFixup, fixed_value: int, contents):
        size_in_bits = self.get_fixup_kind_info(fixup.kind).size
        offset_in_bits = self.get_fixup_kind_info(fixup.kind).offset
        size = int((offset_in_bits + size_in_bits + 7) / 8)

        if fixed_value == 0:
            return

        offset = fixup.offset
        fixed_value = self.adjust_fixup_value(fixup, fixed_value)

        def to_bytes(value, order):
            import struct

            if order == "little":
                return struct.pack("<Q", value)
            else:
                assert(order == "big")
                return struct.pack(">Q", value)

        assert(offset < len(contents))
        assert(offset + size <= len(contents))

        fixed_value = fixed_value << offset_in_bits

        order = 'little'
        bys = list(to_bytes(fixed_value, order))

        for idx in range(size):
            contents[offset + idx] |= bys[idx]

    def should_force_relocation(self, asm, fixup: MCFixup, target):
        if fixup.kind == RISCVFixupKind.RISCV_TLS_GD_HI20:
            return True

        if fixup.kind in [RISCVFixupKind.RISCV_PCREL_LO12_I, RISCVFixupKind.RISCV_PCREL_LO12_S]:
            pcrel_hi_fixup = fixup.value.pcrel_hi_fixup

            assert(pcrel_hi_fixup.kind in [
                   RISCVFixupKind.RISCV_PCREL_HI20, RISCVFixupKind.RISCV_TLS_GD_HI20])
            if pcrel_hi_fixup.value.expr.symbol.section != fixup.value.expr.symbol.section:
                return True

        return False


class RISCVTSFlags(IntFlag):
    Pseudo = 0


class RISCVFixupKind(Enum):
    RISCV_HI20 = auto()
    RISCV_LO12_I = auto()
    RISCV_LO12_S = auto()
    RISCV_JAL = auto()
    RISCV_BRANCH = auto()
    RISCV_RVC_JUMP = auto()
    RISCV_RVC_BRANCH = auto()
    RISCV_PCREL_HI20 = auto()
    RISCV_PCREL_LO12_I = auto()
    RISCV_PCREL_LO12_S = auto()
    RISCV_TLS_GD_HI20 = auto()
    RISCV_CALL = auto()


def get_reg_code(operand):
    assert(isinstance(operand, MCOperandReg))

    return operand.reg.spec.encoding


def get_imm_common_code(inst, operand, fixups):

    assert(operand.is_expr)

    expr = operand.expr
    ty = expr.ty

    if ty == MCExprType.Target:
        if expr.kind == RISCVMCExprVarKind.PCRelHi:
            fixup_kind = RISCVFixupKind.RISCV_PCREL_HI20
        elif expr.kind == RISCVMCExprVarKind.PCRelLo:
            if inst.opcode in [RISCVMachineOps.ADDI]:
                fixup_kind = RISCVFixupKind.RISCV_PCREL_LO12_I
            else:
                raise NotImplementedError()
        elif expr.kind == RISCVMCExprVarKind.Hi:
            fixup_kind = RISCVFixupKind.RISCV_HI20
        elif expr.kind == RISCVMCExprVarKind.Lo:
            if inst.opcode in [RISCVMachineOps.ADDI]:
                fixup_kind = RISCVFixupKind.RISCV_LO12_I
            else:
                raise NotImplementedError()
        elif expr.kind == RISCVMCExprVarKind.TLSGDHi:
            fixup_kind = RISCVFixupKind.RISCV_TLS_GD_HI20
        elif expr.kind == RISCVMCExprVarKind.Call:
            fixup_kind = RISCVFixupKind.RISCV_CALL
        else:
            raise NotImplementedError()
    elif ty == MCExprType.SymbolRef and expr.kind == MCVariantKind.Non:
        if inst.opcode == RISCVMachineOps.JAL:
            fixup_kind = RISCVFixupKind.RISCV_JAL
        elif inst.opcode in [RISCVMachineOps.BNE]:
            fixup_kind = RISCVFixupKind.RISCV_BRANCH
        else:
            raise NotImplementedError()

    fixups.append(MCFixup(0, expr, fixup_kind))

    return 0


def get_imm12_code(inst, operand, fixups):
    if operand.is_imm:
        assert(operand.imm < ((1 << 12) - 1))
        return operand.imm

    return get_imm_common_code(inst, operand, fixups)


def get_imm20_code(inst, operand, fixups):
    if operand.is_imm:
        assert(operand.imm < ((1 << 20) - 1))
        return operand.imm

    return get_imm_common_code(inst, operand, fixups)


def write_bits(value, bits, offset, count):
    mask = ((0x1 << count) - 1) << offset
    return (value & ~mask) | ((bits << offset) & mask)


def read_bits(value, offset, count):
    mask = (0x1 << count) - 1
    return (value >> offset) & mask


def get_rv_inst_r(inst: MCInst, fixups):
    OPCODE_TABLE = {
        RISCVMachineOps.ADD: (0b0110011, 0b000, 0b0000000),
        RISCVMachineOps.SUB: (0b0110011, 0b000, 0b0100000),
        RISCVMachineOps.SLL: (0b0110011, 0b001, 0b0000000),
        RISCVMachineOps.SLT: (0b0110011, 0b010, 0b0000000),
        RISCVMachineOps.SLTU: (0b0110011, 0b011, 0b0000000),
        RISCVMachineOps.XOR: (0b0110011, 0b100, 0b0000000),
        RISCVMachineOps.SRL: (0b0110011, 0b101, 0b0000000),
        RISCVMachineOps.SRA: (0b0110011, 0b101, 0b0100000),
        RISCVMachineOps.OR: (0b0110011, 0b110, 0b0000000),
        RISCVMachineOps.AND: (0b0110011, 0b111, 0b0000000),

        RISCVMachineOps.SLLW: (0b0111011, 0b001, 0b0000000),
        RISCVMachineOps.SRLW: (0b0111011, 0b101, 0b0000000),
        RISCVMachineOps.SRAW: (0b0111011, 0b101, 0b0100000),

        RISCVMachineOps.FADD_S: (0b1010011, 0b000, 0b0000000),
        RISCVMachineOps.FSUB_S: (0b1010011, 0b000, 0b0000100),
        RISCVMachineOps.FMUL_S: (0b1010011, 0b000, 0b0001000),
        RISCVMachineOps.FDIV_S: (0b1010011, 0b000, 0b0001100),
        RISCVMachineOps.FSGNJ_S: (0b1010011, 0b000, 0b0010000),

        RISCVMachineOps.FEQ_S: (0b1010011, 0b010, 0b1010000),
        RISCVMachineOps.FLT_S: (0b1010011, 0b001, 0b1010000),
        RISCVMachineOps.FLE_S: (0b1010011, 0b000, 0b1010000),
    }

    opcode, funct3, funct7 = OPCODE_TABLE[inst.opcode]

    rd = get_reg_code(inst.operands[0])
    rs1 = get_reg_code(inst.operands[1])
    rs2 = get_reg_code(inst.operands[2])

    code = 0
    code = write_bits(code, opcode, 0, 7)
    code = write_bits(code, rd, 7, 5)
    code = write_bits(code, funct3, 12, 3)
    code = write_bits(code, rs1, 15, 5)
    code = write_bits(code, rs2, 20, 5)
    code = write_bits(code, funct7, 25, 7)

    return code


def get_rv_inst_i(inst: MCInst, fixups):
    OPCODE_TABLE = {
        RISCVMachineOps.ADDI: (0b0010011, 0b000),
        RISCVMachineOps.SLTI: (0b0010011, 0b010),
        RISCVMachineOps.SLTIU: (0b0010011, 0b011),
        RISCVMachineOps.XORI: (0b0010011, 0b100),
        RISCVMachineOps.ORI: (0b0010011, 0b110),
        RISCVMachineOps.ANDI: (0b0010011, 0b111),
        RISCVMachineOps.LW: (0b0000011, 0b010),
        RISCVMachineOps.FLW: (0b0000111, 0b010),
        RISCVMachineOps.LD: (0b0000011, 0b011),
        RISCVMachineOps.FLD: (0b0000111, 0b011),
        RISCVMachineOps.JALR: (0b1100111, 0b000),
    }

    opcode, funct3 = OPCODE_TABLE[inst.opcode]

    rd = get_reg_code(inst.operands[0])
    rs1 = get_reg_code(inst.operands[1])
    imm = get_imm12_code(inst, inst.operands[2], fixups)

    code = 0
    code = write_bits(code, opcode, 0, 7)
    code = write_bits(code, rd, 7, 5)
    code = write_bits(code, funct3, 12, 3)
    code = write_bits(code, rs1, 15, 5)
    code = write_bits(code, imm, 20, 12)

    return code


def get_rv_inst_s(inst: MCInst, fixups):
    OPCODE_TABLE = {
        RISCVMachineOps.SW: (0b0100011, 0b010),
        RISCVMachineOps.FSW: (0b0100111, 0b010),
        RISCVMachineOps.SD: (0b0100011, 0b011),
        RISCVMachineOps.FSD: (0b0100111, 0b011),
    }

    opcode, funct3 = OPCODE_TABLE[inst.opcode]

    rs2 = get_reg_code(inst.operands[0])
    rs1 = get_reg_code(inst.operands[1])
    imm = get_imm12_code(inst, inst.operands[2], fixups)

    code = 0
    code = write_bits(code, opcode, 0, 7)
    code = write_bits(code, read_bits(imm, 0, 5), 7, 5)
    code = write_bits(code, funct3, 12, 3)
    code = write_bits(code, rs1, 15, 5)
    code = write_bits(code, rs2, 20, 5)
    code = write_bits(code, read_bits(imm, 5, 7), 25, 7)

    return code


def get_rv_inst_b(inst: MCInst, fixups):
    OPCODE_TABLE = {
        RISCVMachineOps.BEQ: (0b1100011, 0b000),
        RISCVMachineOps.BNE: (0b1100011, 0b001),
        RISCVMachineOps.BLT: (0b1100011, 0b100),
        RISCVMachineOps.BGE: (0b1100011, 0b101),
        RISCVMachineOps.BLTU: (0b1100011, 0b110),
        RISCVMachineOps.BGEU: (0b1100011, 0b111),
    }

    opcode, funct3 = OPCODE_TABLE[inst.opcode]

    rs1 = get_reg_code(inst.operands[0])
    rs2 = get_reg_code(inst.operands[1])
    imm = get_imm12_code(inst, inst.operands[2], fixups)

    code = 0
    code = write_bits(code, opcode, 0, 7)
    code = write_bits(code, read_bits(imm, 0, 5), 7, 5)
    code = write_bits(code, funct3, 12, 3)
    code = write_bits(code, rs1, 15, 5)
    code = write_bits(code, rs2, 20, 5)
    code = write_bits(code, read_bits(imm, 5, 7), 25, 7)

    return code


def get_rv_inst_u(inst: MCInst, fixups):
    OPCODE_TABLE = {
        RISCVMachineOps.LUI: (0b0110111),
        RISCVMachineOps.AUIPC: (0b0010111),
    }

    opcode = OPCODE_TABLE[inst.opcode]

    rd = get_reg_code(inst.operands[0])
    imm = get_imm20_code(inst, inst.operands[1], fixups)

    code = 0
    code = write_bits(code, opcode, 0, 7)
    code = write_bits(code, rd, 7, 5)
    code = write_bits(code, imm, 12, 20)

    return code


def get_rv_inst_j(inst: MCInst, fixups):

    if inst.opcode == RISCVMachineOps.JAL:
        opcode = 0b1101111

    rd = get_reg_code(inst.operands[0])
    imm = get_imm20_code(inst, inst.operands[1], fixups)
    imm = read_bits(imm, 12, 8) | (read_bits(imm, 11, 1) << 8) | \
        (read_bits(imm, 1, 10) << 9) | (read_bits(imm, 20, 1) << 19)

    code = 0
    code = write_bits(code, opcode, 0, 7)
    code = write_bits(code, rd, 7, 5)
    code = write_bits(code, imm, 12, 20)

    return code


def get_inst_binary_code(inst: MCInst, fixups):
    opcode = inst.opcode
    num_operands = len(inst.operands)

    if opcode in [
            RISCVMachineOps.ADD, RISCVMachineOps.SUB, RISCVMachineOps.SLT, RISCVMachineOps.SLTU, RISCVMachineOps.AND,
            RISCVMachineOps.OR, RISCVMachineOps.SRL, RISCVMachineOps.SRA, RISCVMachineOps.SLL, RISCVMachineOps.XOR,
            RISCVMachineOps.SLLW, RISCVMachineOps.SRAW, RISCVMachineOps.SRLW,
            RISCVMachineOps.FADD_S, RISCVMachineOps.FSUB_S, RISCVMachineOps.FMUL_S, RISCVMachineOps.FDIV_S,
            RISCVMachineOps.FLE_S, RISCVMachineOps.FLT_S, RISCVMachineOps.FSGNJ_S]:
        return get_rv_inst_r(inst, fixups)

    if opcode in [
            RISCVMachineOps.ADDI, RISCVMachineOps.XORI, RISCVMachineOps.SLTI, RISCVMachineOps.SLTIU,
            RISCVMachineOps.JALR,
            RISCVMachineOps.LW, RISCVMachineOps.LD, RISCVMachineOps.FLW, RISCVMachineOps.FLD]:
        return get_rv_inst_i(inst, fixups)

    if opcode in [RISCVMachineOps.SW, RISCVMachineOps.SD, RISCVMachineOps.FSW, RISCVMachineOps.FSD]:
        return get_rv_inst_s(inst, fixups)

    if opcode in [RISCVMachineOps.LUI, RISCVMachineOps.AUIPC]:
        return get_rv_inst_u(inst, fixups)

    if opcode in [RISCVMachineOps.JAL]:
        return get_rv_inst_j(inst, fixups)

    if opcode in [RISCVMachineOps.BNE]:
        return get_rv_inst_b(inst, fixups)

    raise NotImplementedError()


class RISCVCodeEmitter(MCCodeEmitter):
    def __init__(self, context: MCContext):
        super().__init__()
        self.context = context

    def emit_byte(self, value, output):
        output.write(value.to_bytes(1, byteorder="little", signed=False))

    def emit_constant(self, value, size, output):
        output.write(value.to_bytes(size, byteorder="little", signed=False))

    def expand_func_call(self, inst: MCInst, fixups, output):
        func = inst.operands[0]
        ra = MachineRegister(X1)

        func_expr = func.expr

        mcinst = MCInst(RISCVMachineOps.AUIPC)
        mcinst.add_operand(MCOperandReg(ra))

        mcinst.add_operand(MCOperandExpr(func_expr))

        code = get_inst_binary_code(mcinst, fixups)
        self.emit_constant(code, 4, output)

        mcinst = MCInst(RISCVMachineOps.JALR)
        mcinst.add_operand(MCOperandReg(ra))
        mcinst.add_operand(MCOperandReg(ra))
        mcinst.add_operand(MCOperandImm(0))

        code = get_inst_binary_code(mcinst, fixups)
        self.emit_constant(code, 4, output)

    def encode_instruction(self, inst: MCInst, fixups, output):
        opcode = inst.opcode
        num_operands = len(inst.operands)

        if opcode == RISCVMachineOps.PseudoCALL:
            self.expand_func_call(inst, fixups, output)
            return

        code = get_inst_binary_code(inst, fixups)

        self.emit_constant(code, 4, output)
