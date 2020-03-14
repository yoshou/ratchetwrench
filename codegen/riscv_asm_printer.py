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
    func_num = bb.func.func_info.func.module.funcs.index(
        bb.func.func_info.func)
    prefix = ".LBB"
    return ctx.get_or_create_symbol(f"{prefix}{str(func_num)}_{str(bb_num)}")


def get_global_symbol(g: GlobalValue, ctx: MCContext):
    # name = "_" + g.name
    name = g.name
    return ctx.get_or_create_symbol(name)


def get_pic_label(label_id, func_number, ctx: MCContext):
    return ctx.get_or_create_symbol(f"PC{func_number}_{label_id}")


class RISCVMCExprVarKind(Enum):
    Non = auto()
    Lo = auto()
    Hi = auto()


class RISCVMCExpr(MCTargetExpr):
    def __init__(self, kind, expr):
        super().__init__()

        self.kind = kind
        self.expr = expr


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

        if operand.target_flags & RISCVOperandFlag.MO_LO16.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.Lo, expr)
        elif operand.target_flags & RISCVOperandFlag.MO_HI16.value:
            expr = RISCVMCExpr(RISCVMCExprVarKind.Hi, expr)

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
        func_num = func.module.funcs.index(func)
        name = f"CPI{func_num}_{index}"
        return self.ctx.get_or_create_symbol(name)

    def emit_constant_pool(self):
        pass

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

        if inst.opcode == RISCVMachineOps.CPEntry:
            label_id = inst.operands[0].val
            cp_idx = inst.operands[1].index

            self.stream.emit_label(self.get_constant_pool_symbol(cp_idx))

            cp_entry = self.mfunc.constant_pool.constants[cp_idx]

            if cp_entry.is_machine_cp_entry:
                self.emit_machine_cp_value(data_layout, cp_entry.value)
            else:
                self.emit_global_constant(data_layout, cp_entry.value)
            return

        if inst.opcode == RISCVMachineOps.PIC_ADD:
            pic_label_id = inst.operands[2].val
            func_number = self.func.module.funcs.index(self.func)
            self.stream.emit_label(get_pic_label(
                pic_label_id, func_number, self.ctx))

            mc_inst = MCInst(RISCVMachineOps.ADDrr)
            mc_inst.add_operand(MCOperandReg(inst.operands[0].reg))
            mc_inst.add_operand(MCOperandReg(MachineRegister(PC)))
            mc_inst.add_operand(MCOperandReg(inst.operands[1].reg))
            self.emit_mc_inst(mc_inst)
            return

        mc_inst_lower = RISCVMCInstLower(self.ctx, inst.mbb.func, self)
        mc_inst = mc_inst_lower.lower(inst)

        for operand in mc_inst.operands:
            if operand.is_expr:
                if operand.expr.ty == MCExprType.SymbolRef:
                    if hasattr(self.stream, "assembler"):
                        self.stream.assembler.register_symbol(
                            operand.expr.symbol)

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
        for variable in self.module.global_variables:
            self.emit_global_variable(variable)

        for func in self.module.funcs:
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


class RISCVAsmBackend(MCAsmBackend):
    def __init__(self):
        super().__init__()

    def may_need_relaxation(self, inst: MCInst):
        return False

    def relax_instruction(self, inst: MCInst):
        pass

    def get_fixup_kind_info(self, kind):
        table = {
            RISCVFixupKind.RISCV_COND_BRANCH: MCFixupKindInfo(
                "RISCV_COND_BRANCH", 0, 24, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_UNCOND_BRANCH: MCFixupKindInfo(
                "RISCV_UNCOND_BRANCH", 0, 24, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_UNCOND_BL: MCFixupKindInfo(
                "RISCV_UNCOND_BL", 0, 24, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_PCREL_10: MCFixupKindInfo(
                "RISCV_PCREL_10", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_LDST_PCREL_12: MCFixupKindInfo(
                "RISCV_LDST_PCREL_12", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            RISCVFixupKind.RISCV_MOVT_HI16: MCFixupKindInfo(
                "RISCV_MOVT_HI16", 0, 20, MCFixupKindInfoFlag.Non),
            RISCVFixupKind.RISCV_MOVW_LO16: MCFixupKindInfo(
                "RISCV_MOVW_LO16", 0, 20, MCFixupKindInfoFlag.Non)

        }

        if kind in table:
            return table[kind]

        return super().get_fixup_kind_info(kind)

    def is_fixup_kind_pcrel(self, fixup):
        kind = fixup.kind
        return self.get_fixup_kind_info(kind).flags & MCFixupKindInfoFlag.IsPCRel == MCFixupKindInfoFlag.IsPCRel

    def adjust_fixup_value(self, fixup: MCFixup, fixed_value: int):
        kind = fixup.kind

        if kind in [RISCVFixupKind.RISCV_COND_BRANCH, RISCVFixupKind.RISCV_UNCOND_BRANCH, RISCVFixupKind.RISCV_UNCOND_BL]:
            return 0xffffff & ((fixed_value - 8) >> 2)

        if kind in [RISCVFixupKind.RISCV_PCREL_10]:
            fixed_value = fixed_value - 8
            is_add = 1
            if fixed_value < 0:
                is_add = 0
                fixed_value = -fixed_value

            fixed_value = fixed_value >> 2

            assert(fixed_value < 256)

            fixed_value = fixed_value | (is_add << 23)
            return fixed_value

        if kind in [RISCVFixupKind.RISCV_LDST_PCREL_12]:
            fixed_value = fixed_value - 8
            is_add = 1
            if fixed_value < 0:
                is_add = 0
                fixed_value = -fixed_value

            fixed_value = get_mod_imm(fixed_value)

            assert(fixed_value >= 0)

            fixed_value = fixed_value | (is_add << 23)
            return fixed_value

        if kind in [RISCVFixupKind.RISCV_MOVT_HI16]:
            fixed_value = fixed_value >> 16
        if kind in [RISCVFixupKind.RISCV_MOVT_HI16, RISCVFixupKind.RISCV_MOVW_LO16]:
            hi4 = (fixed_value >> 12) & 0xF
            lo12 = fixed_value & 0xFFF

            return (hi4 << 16) | lo12

        if kind in [MCFixupKind.Data_1, MCFixupKind.Data_2, MCFixupKind.Data_4]:
            return fixed_value

        raise NotImplementedError()

    def apply_fixup(self, fixup: MCFixup, fixed_value: int, contents):
        size = get_fixup_size_by_kind(fixup.kind)
        offset = fixup.offset
        fixed_value = self.adjust_fixup_value(fixup, fixed_value)

        if fixed_value == 0:
            return

        def to_bytes(value, order):
            import struct

            if order == "little":
                return struct.pack("<Q", value)
            else:
                assert(order == "big")
                return struct.pack(">Q", value)

        assert(offset < len(contents))
        assert(offset + size <= len(contents))

        order = 'little'
        bys = list(to_bytes(fixed_value, order))

        for idx in range(size):
            contents[offset + idx] |= bys[idx]

    def should_force_relocation(self, asm, fixup: MCFixup, target):
        if fixup.kind == RISCVFixupKind.RISCV_UNCOND_BL:
            return True

        return False


class RISCVTSFlags(IntFlag):
    Pseudo = 0


class RISCVFixupKind(Enum):
    RISCV_COND_BRANCH = auto()
    RISCV_UNCOND_BRANCH = auto()
    RISCV_UNCOND_BL = auto()
    RISCV_PCREL_10 = auto()
    RISCV_LDST_PCREL_12 = auto()
    RISCV_MOVW_LO16 = auto()
    RISCV_MOVT_HI16 = auto()


def write_bits(value, bits, offset, count):
    mask = ((0x1 << count) - 1) << offset
    return (value & ~mask) | ((bits << offset) & mask)


def get_binop_opcode(inst: MCInst):
    opcode = inst.opcode
    if opcode in [RISCVMachineOps.MOVr, RISCVMachineOps.MOVCCi, RISCVMachineOps.MOVCCr]:
        return 0b1101
    if opcode in [RISCVMachineOps.ANDri, RISCVMachineOps.ANDrr, RISCVMachineOps.ANDrsi]:
        return 0b0000
    if opcode in [RISCVMachineOps.XORri, RISCVMachineOps.XORrr, RISCVMachineOps.XORrsi]:
        return 0b0001
    if opcode in [RISCVMachineOps.ORri, RISCVMachineOps.ORrr, RISCVMachineOps.ORrsi]:
        return 0b1100
    if opcode in [RISCVMachineOps.ADDri, RISCVMachineOps.ADDrr, RISCVMachineOps.ADDrsi]:
        return 0b0100
    if opcode in [RISCVMachineOps.SUBri, RISCVMachineOps.SUBrr, RISCVMachineOps.SUBrsi]:
        return 0b0010
    if opcode in [RISCVMachineOps.CMPri, RISCVMachineOps.CMPrr, RISCVMachineOps.CMPrsi]:
        return 0b1010

    raise NotImplementedError()


def get_binop_inst_code(inst: MCInst, cond, imm_op, set_cond, rn, rd):
    opc = get_binop_opcode(inst)

    if opc == 0b1101:
        rn = rd

    code = get_inst_code(inst, rn, rd)

    code = write_bits(code, cond, 28, 4)

    code = write_bits(code, 0b0, 26, 2)
    code = write_bits(code, imm_op, 25, 1)
    code = write_bits(code, opc, 21, 4)

    if opc == 0b1101:
        code = write_bits(code, 0, 16, 4)

    if opc == 0b1010:
        code = write_bits(code, 0, 12, 4)
        code = write_bits(code, 1, 20, 1)

    if imm_op == 0:
        code = write_bits(code, get_reg_code(inst.operands[-1]), 0, 4)
    else:
        assert(imm_op == 1)
        imm = get_modimm_code(inst.operands[-1])
        code = write_bits(code, imm, 0, 12)

    return code


def get_inst_code(inst: MCInst, rn, rd):
    code = 0
    code = write_bits(code, 0b1110, 28, 4)
    code = write_bits(code, get_reg_code(rd), 12, 4)
    code = write_bits(code, get_reg_code(rn), 16, 4)
    code = write_bits(code, 1, 23, 1)

    return code


def get_reg_code(operand):
    return operand.reg.spec.encoding


def get_imm16_code(operand):
    return read_bits(operand.imm, 0, 16)


def get_imm_code(operand):
    simm = operand.imm
    is_add = 1
    if simm < 0:
        is_add = 0
        simm = -simm

    return (is_add, simm)


def get_shift_opcode(operand):
    opc = operand.imm & 0b111

    if opc == 0x2:  # lsl
        return 0x0
    if opc == 0x3:  # lsr
        return 0x2
    if opc == 0x1:  # asr
        return 0x4
    if opc == 0x4:  # ror
        return 0x6

    raise NotImplementedError()


def get_shift_offset(operand):
    return operand.imm >> 3


def get_so_imm_code(inst, idx):
    operands = inst.operands

    value = get_reg_code(operands[idx])
    shift_op = get_shift_opcode(operands[idx + 1])
    offset = get_shift_offset(operands[idx + 1])

    return value | (shift_op << 4) | (offset << 7)


def get_modimm_code(operand):
    imm = operand.imm
    return get_mod_imm(imm)


def get_ldst_inst_code(inst: MCInst, is_load, is_byte, imm_op, fixups):
    code = 0
    code = write_bits(code, 0b1110, 28, 4)  # always
    code = write_bits(code, get_reg_code(inst.operands[0]), 12, 4)
    code = write_bits(code, 1, 23, 1)

    code = write_bits(code, 0b01, 26, 2)
    code = write_bits(code, 1, 24, 1)  # Pre
    code = write_bits(code, is_byte, 22, 1)  # Byte
    code = write_bits(code, 0, 21, 1)  # Write-back
    code = write_bits(code, is_load, 20, 1)  # Load

    if imm_op == 0:
        raise NotImplementedError()
        code = write_bits(code, get_reg_code(inst.operands[2]), 0, 4)
    else:
        assert(imm_op == 1)
        reg, is_add, imm = get_addr_mode_imm12_code(inst, 1, fixups)
        code = write_bits(code, reg, 16, 4)
        code = write_bits(code, read_bits(imm, 0, 12), 0, 12)
        code = write_bits(code, is_add, 23, 1)

    return code


def get_indexed_ldst_inst_code(inst: MCInst, is_load, is_byte, is_pre, imm_op, rn, rd):
    code = get_inst_code(inst, rn, rd)

    code = write_bits(code, 0b01, 26, 2)
    code = write_bits(code, is_pre, 24, 1)
    code = write_bits(code, is_byte, 22, 1)
    code = write_bits(code, is_pre, 21, 1)
    code = write_bits(code, is_load, 20, 1)

    if imm_op == 0:
        code = write_bits(code, get_reg_code(inst.operands[2]), 0, 4)
    else:
        assert(imm_op == 1)
        is_add, imm = get_imm_code(inst.operands[2])
        code = write_bits(code, read_bits(imm, 0, 12), 0, 12)
        code = write_bits(code, is_add, 23, 1)

    return code


def get_addr_mode_imm12_code(inst, operands, idx):
    reg_op = operands[idx]
    imm_op = operands[idx + 1]

    reg = get_reg_code(reg_op)
    is_add, imm = get_imm_code(imm_op)

    return (reg, is_add, read_bits(imm, 0, 12))


def get_indexed_st_inst_code(inst: MCInst, is_byte, is_pre, imm_op, rn, rd, fixups):
    code = get_indexed_ldst_inst_code(inst, 0, is_byte, is_pre, imm_op, rn, rd)

    reg, is_add, imm = get_addr_mode_imm12_code(inst, 1, fixups)

    code = write_bits(code, 0b0, 25, 1)
    code = write_bits(code, is_add, 23, 1)
    code = write_bits(code, reg, 16, 4)
    code = write_bits(code, imm, 0, 12)

    return code


def get_indexed_ld_inst_code(inst: MCInst, is_byte, is_pre, imm_op, rn, rd, fixups):
    code = get_indexed_ldst_inst_code(inst, 1, is_byte, is_pre, imm_op, rn, rd)

    reg, is_add, imm = get_addr_mode_imm12_code(inst, 1, fixups)

    code = write_bits(code, 0b0, 25, 1)
    code = write_bits(code, is_add, 23, 1)
    code = write_bits(code, reg, 16, 4)
    code = write_bits(code, imm, 0, 12)

    return code


def read_bits(value, offset, count):
    mask = (0x1 << count) - 1
    return (value >> offset) & mask


def get_hilo16_imm_code(inst: MCInst, idx, fixups):
    operand = inst.operands[idx]
    if operand.is_imm:
        return get_imm16_code(operand)

    assert(operand.is_expr)

    expr = operand.expr
    assert(expr.ty == MCExprType.Target)

    subexpr = expr.expr

    if isinstance(subexpr, MCConstantExpr):
        raise NotImplementedError()

    assert(isinstance(subexpr, MCExpr))
    if expr.kind == RISCVMCExprVarKind.Lo:
        fixup_kind = RISCVFixupKind.RISCV_MOVW_LO16
    elif expr.kind == RISCVMCExprVarKind.Hi:
        fixup_kind = RISCVFixupKind.RISCV_MOVT_HI16
    else:
        assert(0)

    fixups.append(MCFixup(0, subexpr, fixup_kind))

    return 0


def get_mov_immediate_inst_code(inst: MCInst, opcode, idx, fixups):
    code = 0

    code = write_bits(code, 0b1110, 28, 4)  # cond
    code = write_bits(code, opcode, 21, 4)  # opcod
    code = write_bits(code, 0b00, 26, 2)  #
    code = write_bits(code, get_reg_code(inst.operands[idx]), 12, 4)

    imm = get_hilo16_imm_code(inst, idx+1, fixups)

    code = write_bits(code, read_bits(imm, 0, 12), 0, 12)
    code = write_bits(code, read_bits(imm, 12, 4), 16, 4)

    code = write_bits(code, 0b0, 20, 1)
    code = write_bits(code, 0b1, 25, 1)

    return code


def get_branch_target_code(inst: MCInst, idx, fixup_kind, fixups):
    operand = inst.operands[idx]

    if operand.is_expr:
        expr = operand.expr
        fixups.append(MCFixup(0, expr, fixup_kind))
        return 0

    return operand.imm >> 2


def get_addr_mode5_code(inst: MCInst, idx, fixups):
    operands = inst.operands

    op1 = operands[idx]

    if op1.is_expr:
        reg = PC.encoding
        is_add, imm = 0, 0

        expr = op1.expr
        fixups.append(MCFixup(0, expr, RISCVFixupKind.RISCV_PCREL_10))
    else:
        imm_op = operands[idx + 1]

        reg = get_reg_code(op1)
        is_add, imm = get_imm_code(imm_op)

    return (reg, is_add, read_bits(imm, 0, 8))


def get_addr_mode_imm12_code(inst: MCInst, idx, fixups):
    operands = inst.operands

    op1 = operands[idx]

    if op1.is_expr:
        reg = PC.encoding
        is_add, imm = 0, 0

        expr = op1.expr
        fixups.append(MCFixup(0, expr, RISCVFixupKind.RISCV_LDST_PCREL_12))
    else:
        imm_op = operands[idx + 1]

        reg = get_reg_code(op1)
        is_add, imm = get_imm_code(imm_op)

    return (reg, is_add, read_bits(imm, 0, 12))


def get_inst_binary_code(inst: MCInst, fixups):
    opcode = inst.opcode
    num_operands = len(inst.operands)

    if opcode == RISCVMachineOps.STRi12:
        return get_ldst_inst_code(inst, 0, 0, 1, fixups)

    if opcode == RISCVMachineOps.LDRi12:
        return get_ldst_inst_code(inst, 1, 0, 1, fixups)

    if opcode == RISCVMachineOps.MOVi:
        rd = inst.operands[0]

        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # cond
        code = write_bits(code, 0b1101, 21, 4)  # opcod
        code = write_bits(code, 0b00, 26, 2)  #
        code = write_bits(code, get_reg_code(rd), 12, 4)

        imm = get_imm16_code(inst.operands[-1])

        code = write_bits(code, read_bits(imm, 0, 12), 0, 12)
        code = write_bits(code, 0b0000, 16, 4)

        code = write_bits(code, 0b1, 25, 1)

        return code

    if opcode == RISCVMachineOps.MOVi16:
        return get_mov_immediate_inst_code(inst, 0b1000, 0, fixups)

    if opcode == RISCVMachineOps.MOVTi16:
        return get_mov_immediate_inst_code(inst, 0b1010, 1, fixups)

    if opcode == RISCVMachineOps.MOVsi:
        rd = inst.operands[0]

        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # cond
        code = write_bits(code, 0b1101, 21, 4)  # opcod
        code = write_bits(code, 0b00, 26, 2)  #
        code = write_bits(code, get_reg_code(rd), 12, 4)

        imm = get_so_imm_code(inst, 1)

        code = write_bits(code, read_bits(imm, 0, 4), 0, 4)
        code = write_bits(code, 0b0, 4, 1)
        code = write_bits(code, read_bits(imm, 5, 7), 5, 7)

        code = write_bits(code, 0b0000, 16, 4)

        code = write_bits(code, 0b0, 25, 1)

        return code

    if opcode == RISCVMachineOps.MOVPCLR:
        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b0001101000001111000000001110, 0, 28)
        return code

    if opcode == RISCVMachineOps.STR_PRE_IMM:
        return get_indexed_st_inst_code(inst, 0, 1, 1, inst.operands[1], inst.operands[0], fixups)

    if opcode == RISCVMachineOps.LDR_POST_IMM:
        return get_indexed_ld_inst_code(inst, 0, 0, 1, inst.operands[1], inst.operands[0], fixups)

    if opcode in [RISCVMachineOps.MOVr, RISCVMachineOps.ADDrr, RISCVMachineOps.ANDrr, RISCVMachineOps.ORrr, RISCVMachineOps.SUBrr, RISCVMachineOps.XORrr]:
        return get_binop_inst_code(
            inst, 0b1110, 0, 0, inst.operands[1], inst.operands[0])

    if opcode in [RISCVMachineOps.CMPrr]:
        return get_binop_inst_code(
            inst, 0b1110, 0, 0, inst.operands[0], inst.operands[0])

    if opcode in [RISCVMachineOps.ADDri, RISCVMachineOps.ANDri, RISCVMachineOps.ORri, RISCVMachineOps.SUBri, RISCVMachineOps.XORri]:
        return get_binop_inst_code(
            inst, 0b1110, 1, 0, inst.operands[1], inst.operands[0])

    if opcode in [RISCVMachineOps.CMPri]:
        return get_binop_inst_code(
            inst, 0b1110, 1, 0, inst.operands[0], inst.operands[0])

    # vfp (between RISCV core register and single-precision register)
    if opcode in [RISCVMachineOps.VMOVSR]:
        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b11100000, 20, 8)  # opc1
        code = write_bits(code, 0b1010, 8, 4)
        code = write_bits(code, 0b1, 4, 1)

        code = write_bits(code, 0b0, 5, 2)  # opc2

        code = write_bits(code, 0b0000, 0, 4)

        sn = inst.operands[0]
        rt = inst.operands[1]

        sn_reg = get_reg_code(sn)

        code = write_bits(code, read_bits(sn_reg, 1, 4), 16, 4)  # Vd
        code = write_bits(code, read_bits(sn_reg, 0, 1), 7, 1)  # D

        code = write_bits(code, get_reg_code(rt), 12, 4)  # Rt

        return code

    if opcode in [RISCVMachineOps.VLDRS]:
        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b1101, 24, 4)  # opc1
        code = write_bits(code, 0b01, 20, 2)  # opc2
        code = write_bits(code, 0b101, 9, 3)
        code = write_bits(code, 0b0, 8, 1)  # single-presicion

        rn_reg, is_add, imm = get_addr_mode5_code(inst, 1, fixups)

        sd_reg = get_reg_code(inst.operands[0])

        code = write_bits(code, is_add, 23, 1)  # U (add)

        code = write_bits(code, read_bits(sd_reg, 1, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 0, 1), 22, 1)  # D

        code = write_bits(code, rn_reg, 16, 4)  # Rn

        code = write_bits(code, read_bits(imm, 0, 8), 0, 8)  # imm

        return code

    if opcode in [RISCVMachineOps.VLDRD]:
        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b1101, 24, 4)  # opc1
        code = write_bits(code, 0b01, 20, 2)  # opc2
        code = write_bits(code, 0b101, 9, 3)
        code = write_bits(code, 0b1, 8, 1)  # single-presicion

        rn_reg, is_add, imm = get_addr_mode5_code(inst, 1, fixups)

        sd_reg = get_reg_code(inst.operands[0])

        code = write_bits(code, is_add, 23, 1)  # U (add)

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, rn_reg, 16, 4)  # Rn

        code = write_bits(code, read_bits(imm, 0, 8), 0, 8)  # imm

        return code

    if opcode in [RISCVMachineOps.VSTRS]:
        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b1101, 24, 4)  # opc1
        code = write_bits(code, 0b00, 20, 2)  # opc2
        code = write_bits(code, 0b101, 9, 3)
        code = write_bits(code, 0b0, 8, 1)  # single-presicion

        rn_reg, is_add, imm = get_addr_mode5_code(inst, 1, fixups)

        sd_reg = get_reg_code(inst.operands[0])

        code = write_bits(code, is_add, 23, 1)  # U (add)

        code = write_bits(code, read_bits(sd_reg, 1, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 0, 1), 22, 1)  # D

        code = write_bits(code, rn_reg, 16, 4)  # Rn

        code = write_bits(code, read_bits(imm, 0, 8), 0, 8)  # imm

        return code

    if opcode in [RISCVMachineOps.VSTRD]:
        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b1101, 24, 4)  # opc1
        code = write_bits(code, 0b00, 20, 2)  # opc2
        code = write_bits(code, 0b101, 9, 3)
        code = write_bits(code, 0b1, 8, 1)  # single-presicion

        rn_reg, is_add, imm = get_addr_mode5_code(inst, 1, fixups)

        sd_reg = get_reg_code(inst.operands[0])

        code = write_bits(code, is_add, 23, 1)  # U (add)

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, rn_reg, 16, 4)  # Rn

        code = write_bits(code, read_bits(imm, 0, 8), 0, 8)  # imm

        return code

    if opcode in [RISCVMachineOps.VADDS, RISCVMachineOps.VSUBS, RISCVMachineOps.VMULS, RISCVMachineOps.VDIVS]:
        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # always

        if opcode in [RISCVMachineOps.VADDS, RISCVMachineOps.VSUBS, RISCVMachineOps.VMULS]:
            code = write_bits(code, 0b11100, 23, 5)  # opc1
        elif opcode in [RISCVMachineOps.VDIVS]:
            code = write_bits(code, 0b11101, 23, 5)  # opc1

        if opcode in [RISCVMachineOps.VADDS]:
            code = write_bits(code, 0b11, 20, 2)  # opc2
        elif opcode in [RISCVMachineOps.VSUBS]:
            code = write_bits(code, 0b11, 20, 2)
        elif opcode in [RISCVMachineOps.VMULS]:
            code = write_bits(code, 0b10, 20, 2)
        elif opcode in [RISCVMachineOps.VDIVS]:
            code = write_bits(code, 0b00, 20, 2)

        code = write_bits(code, 0b101, 9, 3)
        code = write_bits(code, 0b0, 8, 1)  # single-presicion

        code = write_bits(code, 0b0, 4, 1)

        if opcode in [RISCVMachineOps.VADDS]:
            code = write_bits(code, 0b0, 6, 1)
        elif opcode in [RISCVMachineOps.VMULS]:
            code = write_bits(code, 0b0, 6, 1)
        elif opcode in [RISCVMachineOps.VSUBS]:
            code = write_bits(code, 0b1, 6, 1)

        sd = inst.operands[0]
        sn = inst.operands[1]
        sm = inst.operands[2]

        sn_reg = get_reg_code(sn)
        sd_reg = get_reg_code(sd)
        sm_reg = get_reg_code(sm)

        code = write_bits(code, read_bits(sd_reg, 1, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 0, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sn_reg, 1, 4), 16, 4)  # Vn
        code = write_bits(code, read_bits(sn_reg, 0, 1), 7, 1)  # N

        code = write_bits(code, read_bits(sm_reg, 1, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 0, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.VCMPS]:
        code = 0

        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b11101, 23, 5)  # opc1
        code = write_bits(code, 0b11, 20, 2)  # opc2
        code = write_bits(code, 0b101, 9, 3)
        code = write_bits(code, 0b0, 8, 1)  # dp_operation = (sz == ‘1’)

        code = write_bits(code, 0b0, 4, 1)
        code = write_bits(code, 0b1, 6, 1)

        sd_reg = get_reg_code(inst.operands[0])
        sm_reg = get_reg_code(inst.operands[1])

        code = write_bits(code, read_bits(sd_reg, 1, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 0, 1), 22, 1)  # D

        code = write_bits(code, 0b0100, 16, 4)  # Vn
        code = write_bits(code, 0b0, 7, 1)  # quiet_nan_exc = (E == ‘1’)

        code = write_bits(code, read_bits(sm_reg, 1, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 0, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.Bcc]:
        target = get_branch_target_code(
            inst, 0, RISCVFixupKind.RISCV_COND_BRANCH, fixups)

        code = 0
        code = write_bits(code, inst.operands[1].imm, 28, 4)
        code = write_bits(code, 0b1010, 24, 4)
        code = write_bits(code, target, 0, 24)
        return code

    if opcode in [RISCVMachineOps.B]:
        target = get_branch_target_code(
            inst, 0, RISCVFixupKind.RISCV_UNCOND_BRANCH, fixups)

        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b1010, 24, 4)
        code = write_bits(code, target, 0, 24)

        return code

    if opcode in [RISCVMachineOps.BL]:
        target = get_branch_target_code(
            inst, 0, RISCVFixupKind.RISCV_UNCOND_BL, fixups)

        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b1011, 24, 4)
        code = write_bits(code, target, 0, 24)

        return code

    if opcode in [RISCVMachineOps.VADDfq]:
        code = 0

        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b00100, 23, 5)  # opc1
        code = write_bits(code, 0b0, 21, 1)
        code = write_bits(code, 0b0, 20, 1)  # sz
        code = write_bits(code, 0b110, 9, 3)
        code = write_bits(code, 0b1, 8, 1)  # single-presicion

        code = write_bits(code, 0b0, 4, 1)
        code = write_bits(code, 0b1, 6, 1)  # Q

        sd_reg = get_reg_code(inst.operands[0])
        sn_reg = get_reg_code(inst.operands[1])
        sm_reg = get_reg_code(inst.operands[2])

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sn_reg, 0, 4), 16, 4)  # Vn
        code = write_bits(code, read_bits(sn_reg, 4, 1), 7, 1)  # N

        code = write_bits(code, read_bits(sm_reg, 0, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 4, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.VORRq]:
        code = 0

        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b00100, 23, 5)  # opc1
        code = write_bits(code, 0b1, 21, 1)
        code = write_bits(code, 0b0, 20, 1)
        code = write_bits(code, 0b000, 9, 3)
        code = write_bits(code, 0b1, 8, 1)  # single-presicion

        code = write_bits(code, 0b1, 4, 1)
        code = write_bits(code, 0b1, 6, 1)  # Q

        sd_reg = get_reg_code(inst.operands[0])
        sn_reg = get_reg_code(inst.operands[1])
        sm_reg = get_reg_code(inst.operands[2])

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sn_reg, 0, 4), 16, 4)  # Vn
        code = write_bits(code, read_bits(sn_reg, 4, 1), 7, 1)  # N

        code = write_bits(code, read_bits(sm_reg, 0, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 4, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.VST1q64]:
        code = 0

        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b01000, 23, 5)  # opc1
        code = write_bits(code, 0b0, 21, 1)
        code = write_bits(code, 0b0, 20, 1)  # sz
        code = write_bits(code, 0b1010, 8, 4)  # type == '2' (regs = 2)

        code = write_bits(code, 0b0, 4, 1)

        align_amt = count_trailing_zeros(inst.operands[2].imm)

        if align_amt == 0:
            align = 0
        else:
            assert(align_amt >= 1 and align_amt < 4)
            align_amt = align_amt - 1
            assert((inst.operands[2].imm & (2 << align_amt))
                   == inst.operands[2].imm)

        code = write_bits(code, 3, 6, 2)  # size = log2(nbytes / 8)

        # alignment = if align == ‘00’ then 1 else 4 << UInt(align);
        code = write_bits(code, align_amt, 4, 2)

        sd_reg = get_reg_code(inst.operands[0])
        rn_reg = get_reg_code(inst.operands[1])
        rm_reg = 0b1111

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(rn_reg, 0, 4), 16, 4)  # Rn

        code = write_bits(code, read_bits(rm_reg, 0, 4), 0, 4)  # Rm

        return code

    if opcode in [RISCVMachineOps.VLD1q64]:
        code = 0

        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b01000, 23, 5)  # opc1
        code = write_bits(code, 0b1, 21, 1)
        code = write_bits(code, 0b0, 20, 1)  # sz
        code = write_bits(code, 0b1010, 8, 4)  # type == '2' (regs = 2)

        code = write_bits(code, 0b0, 4, 1)

        align_amt = count_trailing_zeros(inst.operands[2].imm)

        if align_amt == 0:
            align = 0
        else:
            assert(align_amt >= 1 and align_amt < 4)
            align_amt = align_amt - 1
            assert((inst.operands[2].imm & (2 << align_amt))
                   == inst.operands[2].imm)

        code = write_bits(code, 3, 6, 2)  # size = log2(nbytes / 8)

        # alignment = if align == ‘00’ then 1 else 4 << UInt(align);
        code = write_bits(code, align_amt, 4, 2)

        sd_reg = get_reg_code(inst.operands[0])
        rn_reg = get_reg_code(inst.operands[1])
        rm_reg = 0b1111

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(rn_reg, 0, 4), 16, 4)  # Rn

        code = write_bits(code, read_bits(rm_reg, 0, 4), 0, 4)  # Rm

        return code

    if opcode in [RISCVMachineOps.FMSTAT]:
        code = 0
        code = write_bits(code, 0b1110, 28, 4)
        code = write_bits(code, 0b111011110001, 16, 12)
        code = write_bits(code, 0b101000010000, 0, 12)
        code = write_bits(code, 0b1111, 12, 4)
        return code

    if opcode in [RISCVMachineOps.MOVCCr, RISCVMachineOps.MOVCCi]:
        cond = inst.operands[3].imm
        rn, rd = inst.operands[1], inst.operands[0]
        imm_op = 1 if inst.operands[2].is_imm else 0

        opc = get_binop_opcode(inst)

        if opc == 0b1101:
            rn = rd

        code = get_inst_code(inst, rn, rd)

        code = write_bits(code, cond, 28, 4)

        code = write_bits(code, 0b0, 26, 2)
        code = write_bits(code, imm_op, 25, 1)
        code = write_bits(code, opc, 21, 4)

        if opc == 0b1101:
            code = write_bits(code, 0, 16, 4)

        if opc == 0b1010:
            code = write_bits(code, 0, 12, 4)
            code = write_bits(code, 1, 20, 1)

        if imm_op == 0:
            code = write_bits(code, get_reg_code(inst.operands[2]), 0, 4)
        else:
            assert(imm_op == 1)
            imm = inst.operands[2].imm
            code = write_bits(code, imm, 0, 12)

        return code

    if opcode in [RISCVMachineOps.VDUPLN32q]:
        lane = (inst.operands[2].imm << 4) | 0b100

        code = 0
        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b00111, 23, 5)  # opc1

        code = write_bits(code, 0b110, 9, 3)
        code = write_bits(code, 0b1, 21, 1)
        code = write_bits(code, 0b1, 20, 1)
        code = write_bits(code, 0b0, 8, 1)
        code = write_bits(code, 0b0, 7, 1)

        code = write_bits(code, lane, 16, 4)  # ln

        code = write_bits(code, 0b0, 4, 1)
        code = write_bits(code, 0b1, 6, 1)  # Q

        sd_reg = get_reg_code(inst.operands[0])
        sm_reg = get_reg_code(inst.operands[1]) >> 1

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sm_reg, 0, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 4, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.VMULfq]:
        code = 0

        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b00110, 23, 5)  # opc1
        code = write_bits(code, 0b0, 21, 1)
        code = write_bits(code, 0b0, 20, 1)  # sz
        code = write_bits(code, 0b110, 9, 3)
        code = write_bits(code, 0b1, 8, 1)  # single-presicion

        code = write_bits(code, 0b1, 4, 1)
        code = write_bits(code, 0b1, 6, 1)  # Q

        sd_reg = get_reg_code(inst.operands[0])
        sn_reg = get_reg_code(inst.operands[1])
        sm_reg = get_reg_code(inst.operands[2])

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sn_reg, 0, 4), 16, 4)  # Vn
        code = write_bits(code, read_bits(sn_reg, 4, 1), 7, 1)  # N

        code = write_bits(code, read_bits(sm_reg, 0, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 4, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.VSUBfq]:
        code = 0

        code = write_bits(code, 0b1111, 28, 4)  # neon
        code = write_bits(code, 0b00100, 23, 5)  # opc1
        code = write_bits(code, 0b1, 21, 1)
        code = write_bits(code, 0b0, 20, 1)  # sz
        code = write_bits(code, 0b110, 9, 3)
        code = write_bits(code, 0b1, 8, 1)  # single-presicion

        code = write_bits(code, 0b0, 4, 1)
        code = write_bits(code, 0b1, 6, 1)  # Q

        sd_reg = get_reg_code(inst.operands[0])
        sn_reg = get_reg_code(inst.operands[1])
        sm_reg = get_reg_code(inst.operands[2])

        code = write_bits(code, read_bits(sd_reg, 0, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 4, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sn_reg, 0, 4), 16, 4)  # Vn
        code = write_bits(code, read_bits(sn_reg, 4, 1), 7, 1)  # N

        code = write_bits(code, read_bits(sm_reg, 0, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 4, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.VMOVS]:
        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b11101, 23, 5)  # opc1
        code = write_bits(code, 0b11, 20, 2)
        code = write_bits(code, 0b0000, 16, 4)
        code = write_bits(code, 0b0, 4, 1)
        code = write_bits(code, 0b0, 8, 1)  # single-presicion

        code = write_bits(code, 0b01, 6, 2)

        code = write_bits(code, 0b101, 9, 3)

        sd_reg = get_reg_code(inst.operands[0])
        sm_reg = get_reg_code(inst.operands[1])

        code = write_bits(code, read_bits(sd_reg, 1, 4), 12, 4)  # Vd
        code = write_bits(code, read_bits(sd_reg, 0, 1), 22, 1)  # D

        code = write_bits(code, read_bits(sm_reg, 1, 4), 0, 4)  # Vm
        code = write_bits(code, read_bits(sm_reg, 0, 1), 5, 1)  # M

        return code

    if opcode in [RISCVMachineOps.STMDB_UPD]:
        rd = inst.operands[0]

        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b10, 23, 2)  # op
        code = write_bits(code, 0b0, 22, 1)
        code = write_bits(code, 0b1, 21, 1)
        code = write_bits(code, 0b1, 27, 1)

        code = write_bits(code, get_reg_code(rd), 16, 4)

        bits = 0
        for operand in inst.operands[2:]:
            reg = get_reg_code(operand)

            bits = bits | (1 << reg)

        code = write_bits(code, bits, 0, 16)

        return code

    if opcode in [RISCVMachineOps.LDMIA_UPD]:
        rd = inst.operands[0]

        code = 0
        code = write_bits(code, 0b1110, 28, 4)  # always
        code = write_bits(code, 0b01, 23, 2)  # op
        code = write_bits(code, 0b0, 22, 1)
        code = write_bits(code, 0b1, 21, 1)
        code = write_bits(code, 0b1, 27, 1)
        code = write_bits(code, 0b1, 20, 1)

        code = write_bits(code, get_reg_code(rd), 16, 4)

        bits = 0
        for operand in inst.operands[2:]:
            reg = get_reg_code(operand)

            bits = bits | (1 << reg)

        code = write_bits(code, bits, 0, 16)

        return code

    raise NotImplementedError()


class RISCVCodeEmitter(MCCodeEmitter):
    def __init__(self, context: MCContext):
        super().__init__()
        self.context = context

    def emit_byte(self, value, output):
        output.write(value.to_bytes(1, byteorder="little", signed=False))

    def emit_constant(self, value, size, output):
        output.write(value.to_bytes(size, byteorder="little", signed=False))

    def encode_instruction(self, inst: MCInst, fixups, output):
        opcode = inst.opcode
        num_operands = len(inst.operands)

        code = get_inst_binary_code(inst, fixups)

        self.emit_constant(code, 4, output)
