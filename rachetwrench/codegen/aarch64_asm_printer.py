#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rachetwrench.codegen.mir import *
from rachetwrench.codegen.spec import *
from rachetwrench.codegen.passes import *
from rachetwrench.codegen.mc import *
from rachetwrench.codegen.asm_emitter import *
from rachetwrench.codegen.assembler import *
from rachetwrench.ir.values import *
from rachetwrench.codegen.aarch64_gen import AArch64MachineOps
import zlib
import io
from rachetwrench.codegen.aarch64_def import *


def get_bb_symbol(bb: BasicBlock, ctx: MCContext):
    bb_num = bb.number
    func_num = list(bb.func.func_info.func.module.funcs.values()).index(
        bb.func.func_info.func)
    prefix = ".LBB"
    return ctx.get_or_create_symbol(f"{prefix}{str(func_num)}_{str(bb_num)}")


def get_global_symbol(g: GlobalValue, ctx: MCContext):
    # name = "_" + g.name
    name = g.name
    return ctx.get_or_create_symbol(name)


def get_pic_label(label_id, func_number, ctx: MCContext):
    return ctx.get_or_create_symbol(f"PC{func_number}_{label_id}")


class AArch64MCExprVarKind(IntFlag):
    Non = auto()
    Page = auto()
    PageOff = auto()
    Nc = auto()

    ABS = auto()
    GOTTPREL = auto()
    TPREL = auto()
    DTPREL = auto()
    TLSDESC = auto()


class AArch64MCExpr(MCTargetExpr):
    def __init__(self, kind, expr):
        super().__init__()

        self.kind = kind
        self.expr = expr

    def evaluate_expr_as_relocatable(self, layout, fixup):

        result = evaluate_expr_as_relocatable(
            self.expr, layout.asm, layout, fixup)

        if not result:
            return result

        return MCValue(result.value, result.symbol1, result.symbol2, self.kind)


class AArch64MCInstLower:
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
        from rachetwrench.codegen.aarch64_gen import AArch64OperandFlag

        expr = MCSymbolRefExpr(symbol)

        ref_flags = AArch64MCExprVarKind.Non

        if operand.target_flags & AArch64OperandFlag.MO_PAGE.value:
            ref_flags |= AArch64MCExprVarKind.Page

        if operand.target_flags & AArch64OperandFlag.MO_NC.value:
            ref_flags |= AArch64MCExprVarKind.Nc

        if operand.target_flags & AArch64OperandFlag.MO_TLS.value:
            model = operand.value.thread_local

            if model == ThreadLocalMode.GeneralDynamicTLSModel:
                ref_flags |= AArch64MCExprVarKind.TLSDESC
            else:
                raise NotImplementedError()
        else:
            ref_flags |= AArch64MCExprVarKind.ABS

        expr = AArch64MCExpr(ref_flags, expr)

        if not operand.is_jti and not operand.is_mbb and operand.offset != 0:
            expr = MCBinaryExpr(MCBinaryOpcode.Add, expr,
                                MCConstantExpr(operand.offset))

        return MCOperandExpr(expr)

    @property
    def target_info(self):
        return self.func.target_info

    def lower_operand(self, operand):
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
        elif isinstance(operand, MORegisterMask):
            return None

        raise NotImplementedError()

    def lower(self, inst):
        mc_inst = MCInst(inst.opcode)
        for operand in inst.operands:
            mc_op = self.lower_operand(operand)
            if mc_op is not None:
                mc_inst.add_operand(mc_op)
        return mc_inst


class AArch64AsmInfo(MCAsmInfo):
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


class AArch64AsmPrinter(AsmPrinter):
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
        from rachetwrench.codegen.coff import MCSectionCOFF

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
        name = f"CPI{func_num}_{index}"
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
            from rachetwrench.codegen.coff import IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_DTYPE_FUNCTION, SCT_COMPLEX_TYPE_SHIFT

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

        if inst.opcode == TargetDagOps.IMPLICIT_DEF:
            return

        # if inst.opcode == AArch64MachineOps.CPEntry:
        #     label_id = inst.operands[0].val
        #     cp_idx = inst.operands[1].index

        #     self.stream.emit_label(self.get_constant_pool_symbol(cp_idx))

        #     cp_entry = self.mfunc.constant_pool.constants[cp_idx]

        #     if cp_entry.is_machine_cp_entry:
        #         self.emit_machine_cp_value(data_layout, cp_entry.value)
        #     else:
        #         self.emit_global_constant(data_layout, cp_entry.value)
        #     return

        # if inst.opcode == AArch64MachineOps.PIC_ADD:
        #     pic_label_id = inst.operands[2].val
        #     func_number = self.func.module.funcs.index(self.func)
        #     self.stream.emit_label(get_pic_label(
        #         pic_label_id, func_number, self.ctx))

        #     mc_inst = MCInst(AArch64MachineOps.ADDrr)
        #     mc_inst.add_operand(MCOperandReg(inst.operands[0].reg))
        #     mc_inst.add_operand(MCOperandReg(MachineRegister(PC)))
        #     mc_inst.add_operand(MCOperandReg(inst.operands[1].reg))
        #     self.emit_mc_inst(mc_inst)
        #     return

        from rachetwrench.codegen.aarch64_gen import AArch64OperandFlag

        mc_inst_lower = AArch64MCInstLower(self.ctx, inst.mbb.func, self)

        if inst.opcode == AArch64MachineOps.TLSDESC_CALLSEQ:
            sym_op = inst.operands[0]

            tlsdesc_lo12 = MOGlobalAddress(
                sym_op.value, sym_op.offset, AArch64OperandFlag.MO_TLS | AArch64OperandFlag.MO_PAGEOFF)

            tlsdesc = MOGlobalAddress(
                sym_op.value, sym_op.offset, AArch64OperandFlag.MO_TLS | AArch64OperandFlag.MO_PAGE)

            sym = mc_inst_lower.lower_operand(sym_op)
            sym_tlsdesc_lo12 = mc_inst_lower.lower_operand(tlsdesc_lo12)
            sym_tlsdesc = mc_inst_lower.lower_operand(tlsdesc)

            # adrp  x0, :tlsdesc:var
            adrp_inst = MCInst(AArch64MachineOps.ADRP)
            adrp_inst.add_operand(MCOperandReg(MachineRegister(X0)))
            adrp_inst.add_operand(sym_tlsdesc)

            self.emit_mc_inst(adrp_inst)

            # ldr   x1, [x0, #:tlsdesc_lo12:var]
            ldr_inst = MCInst(AArch64MachineOps.LDRXui)
            ldr_inst.add_operand(MCOperandReg(MachineRegister(X1)))
            ldr_inst.add_operand(MCOperandReg(MachineRegister(X0)))
            ldr_inst.add_operand(sym_tlsdesc_lo12)
            ldr_inst.add_operand(MCOperandImm(0))

            self.emit_mc_inst(ldr_inst)

            def get_shift_value(imm):
                return imm & 0x3f

            # add   x0, x0, #:tlsdesc_lo12:var
            add_inst = MCInst(AArch64MachineOps.ADDXri)
            add_inst.add_operand(MCOperandReg(MachineRegister(X0)))
            add_inst.add_operand(MCOperandReg(MachineRegister(X0)))
            add_inst.add_operand(sym_tlsdesc_lo12)
            add_inst.add_operand(MCOperandImm(get_shift_value(0)))

            self.emit_mc_inst(add_inst)

            # .tlsdesccall var
            tlsdesccall_inst = MCInst(AArch64MachineOps.TLSDESCCALL)
            tlsdesccall_inst.add_operand(sym)

            self.emit_mc_inst(tlsdesccall_inst)

            # blr   x1
            blr_inst = MCInst(AArch64MachineOps.BLR)
            blr_inst.add_operand(MCOperandReg(MachineRegister(X1)))

            self.emit_mc_inst(blr_inst)

            return

        if inst.opcode == AArch64MachineOps.FMOVS0:
            mc_inst = MCInst(AArch64MachineOps.FMOVWSr)
            mc_inst.add_operand(MCOperandReg(inst.operands[0].reg))
            mc_inst.add_operand(MCOperandReg(MachineRegister(WZR)))

            self.emit_mc_inst(mc_inst)

            return

        mc_inst = mc_inst_lower.lower(inst)

        for operand in mc_inst.operands:
            if operand.is_expr:
                expr = operand.expr
                if expr.ty == MCExprType.Target:
                    expr = expr.expr

                if expr.ty == MCExprType.SymbolRef:
                    if hasattr(self.stream, "assembler"):
                        self.stream.assembler.register_symbol(
                            expr.symbol)

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
        from rachetwrench.codegen.types import is_integer_ty

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
        from rachetwrench.codegen.aarch64_gen import AArch64ConstantPoolKind, AArch64ConstantPoolModifier
        if value.kind == AArch64ConstantPoolKind.Value:
            symbol = get_global_symbol(value.value, self.ctx)
        else:
            raise ValueError("Unrecognized constant pool kind.")

        def get_variant_kind_from_modifier(modifier):
            if modifier == AArch64ConstantPoolModifier.Non:
                return MCVariantKind.Non
            elif modifier == AArch64ConstantPoolModifier.TLSGlobalDesc:
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
        elif isinstance(constant, TruncInst):
            if isinstance(constant.rs, ConstantInt):
                self.stream.emit_int_value(constant.rs.value, int(size / 8))
            else:
                raise ValueError("Invalid constant type")
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

        initializer = variable.initializer
        if not initializer:
            return

        symbol = get_global_symbol(variable, self.ctx)
        self.stream.emit_label(symbol)

        if self.asm_info.has_dot_type_dot_size_directive:
            self.stream.emit_symbol_attrib(
                symbol, MCSymbolAttribute.ELF_TypeObject)

        self.emit_linkage(variable)

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
            AArch64FixupKind.AArch64_ADD_IMM12]:
        return 3
    elif kind in [
            AArch64FixupKind.AArch64_PCREL_ADRP_IMM21, AArch64FixupKind.AArch64_PCREL_BRANCH26, AArch64FixupKind.AArch64_PCREL_BRANCH19]:
        return 4

    raise ValueError("kind")


class AArch64AsmBackend(MCAsmBackend):
    def __init__(self):
        super().__init__()

    def may_need_relaxation(self, inst: MCInst):
        return False

    def relax_instruction(self, inst: MCInst):
        pass

    def get_fixup_kind_info(self, kind):
        table = {
            AArch64FixupKind.AArch64_ADD_IMM12: MCFixupKindInfo(
                "AArch64_ADD_IMM12", 10, 12, MCFixupKindInfoFlag.Non),
            AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED1: MCFixupKindInfo(
                "AArch64_LDST_IMM12_UNSCALED1", 10, 12, MCFixupKindInfoFlag.Non),
            AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED2: MCFixupKindInfo(
                "AArch64_LDST_IMM12_UNSCALED2", 10, 12, MCFixupKindInfoFlag.Non),
            AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED4: MCFixupKindInfo(
                "AArch64_LDST_IMM12_UNSCALED4", 10, 12, MCFixupKindInfoFlag.Non),
            AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED8: MCFixupKindInfo(
                "AArch64_LDST_IMM12_UNSCALED8", 10, 12, MCFixupKindInfoFlag.Non),
            AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED16: MCFixupKindInfo(
                "AArch64_LDST_IMM12_UNSCALED16", 10, 12, MCFixupKindInfoFlag.Non),
            AArch64FixupKind.AArch64_PCREL_ADR_IMM21: MCFixupKindInfo(
                "AArch64_PCREL_ADR_IMM21", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            AArch64FixupKind.AArch64_PCREL_ADRP_IMM21: MCFixupKindInfo(
                "AArch64_PCREL_ADRP_IMM21", 0, 32, MCFixupKindInfoFlag.IsPCRel),
            AArch64FixupKind.AArch64_PCREL_CALL26: MCFixupKindInfo(
                "AArch64_PCREL_CALL26", 0, 26, MCFixupKindInfoFlag.IsPCRel),
            AArch64FixupKind.AArch64_PCREL_BRANCH14: MCFixupKindInfo(
                "AArch64_PCREL_BRANCH14", 5, 14, MCFixupKindInfoFlag.IsPCRel),
            AArch64FixupKind.AArch64_PCREL_BRANCH19: MCFixupKindInfo(
                "AArch64_PCREL_BRANCH19", 5, 19, MCFixupKindInfoFlag.IsPCRel),
            AArch64FixupKind.AArch64_PCREL_BRANCH26: MCFixupKindInfo(
                "AArch64_PCREL_BRANCH26", 0, 26, MCFixupKindInfoFlag.IsPCRel),
            AArch64FixupKind.AArch64_TLSDESC_CALL: MCFixupKindInfo(
                "AArch64_TLSDESC_CALL", 0, 0, MCFixupKindInfoFlag.Non)
        }

        if kind in table:
            return table[kind]

        return super().get_fixup_kind_info(kind)

    def is_fixup_kind_pcrel(self, fixup):
        kind = fixup.kind
        return self.get_fixup_kind_info(kind).flags & MCFixupKindInfoFlag.IsPCRel == MCFixupKindInfoFlag.IsPCRel

    def adjust_fixup_value(self, fixup: MCFixup, fixed_value: int):
        kind = fixup.kind
        signed_value = fixed_value

        def adr_imm_bits(value):
            lo2 = value & 0x3
            hi19 = (value & 0x1ffffc) >> 2
            return (hi19 << 5) | (lo2 << 29)

        if kind in [AArch64FixupKind.AArch64_PCREL_ADRP_IMM21]:
            return adr_imm_bits((fixed_value & 0x1fffff000) >> 12)

        if kind in [AArch64FixupKind.AArch64_ADD_IMM12]:
            assert(fixed_value < 0x1000)

            return fixed_value & 0xfff

        if kind in [AArch64FixupKind.AArch64_PCREL_BRANCH26, AArch64FixupKind.AArch64_PCREL_CALL26]:
            assert(signed_value <= 134217727 and signed_value >= -134217728)
            assert(fixed_value & 0x3 == 0)

            return (fixed_value >> 2) & 0x3ffffff

        if kind in [AArch64FixupKind.AArch64_PCREL_BRANCH19]:
            assert(signed_value <= 2097151 and signed_value >= -2097152)
            assert(fixed_value & 0x3 == 0)

            return (fixed_value >> 2) & 0x7ffff

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
        # if fixup.kind == AArch64FixupKind.AArch64_UNCOND_BL:
        #     return True

        return False


class AArch64TSFlags(IntFlag):
    Pseudo = 0


class AArch64FixupKind(Enum):
    AArch64_PCREL_ADR_IMM21 = auto()
    AArch64_PCREL_ADRP_IMM21 = auto()
    AArch64_ADD_IMM12 = auto()
    AArch64_LDST_IMM12_UNSCALED1 = auto()
    AArch64_LDST_IMM12_UNSCALED2 = auto()
    AArch64_LDST_IMM12_UNSCALED4 = auto()
    AArch64_LDST_IMM12_UNSCALED8 = auto()
    AArch64_LDST_IMM12_UNSCALED16 = auto()
    AArch64_PCREL_CALL26 = auto()
    AArch64_PCREL_BRANCH14 = auto()
    AArch64_PCREL_BRANCH19 = auto()
    AArch64_PCREL_BRANCH26 = auto()
    AArch64_TLSDESC_CALL = auto()


def get_reg_code(operand):
    return operand.reg.spec.encoding


def get_sysreg_code(operand):
    assert(operand.is_imm)
    idx = operand.imm
    sysreg = list(AArch64SysReg)[idx].value

    code = 0
    code = write_bits(code, sysreg.op0, 14, 2)
    code = write_bits(code, sysreg.op1, 11, 3)
    code = write_bits(code, sysreg.crn, 7, 4)
    code = write_bits(code, sysreg.crm, 3, 4)
    code = write_bits(code, sysreg.op2, 0, 3)

    return code


def write_bits(value, bits, offset, count):
    mask = ((0x1 << count) - 1) << offset
    return (value & ~mask) | ((bits << offset) & mask)


def read_bits(value, offset, count):
    mask = (0x1 << count) - 1
    return (value >> offset) & mask


def get_ldst_unscaled_inst_code(inst: MCInst, sz, v, opc, fixups):
    code = 0
    code = write_bits(code, sz, 30, 2)
    code = write_bits(code, 0b111, 27, 3)
    code = write_bits(code, v, 26, 1)
    code = write_bits(code, 0b00, 24, 2)
    code = write_bits(code, opc, 22, 2)
    code = write_bits(code, 0b0, 21, 0)
    code = write_bits(code, inst.operands[2].imm, 12, 9)
    code = write_bits(code, 0b00, 10, 2)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_ldst_uimm12_value(inst: MCInst, idx, fixups, fixup_kind):
    if inst.operands[idx].is_imm:
        return inst.operands[idx].imm

    assert(inst.operands[idx].is_expr)

    expr = inst.operands[idx].expr
    fixups.append(MCFixup(0, expr, fixup_kind))

    return 0


def get_ldst_ui_inst_code(inst: MCInst, sz, v, opc, fixups, fixup_kind):
    code = 0
    code = write_bits(code, sz, 30, 2)
    code = write_bits(code, 0b111, 27, 3)
    code = write_bits(code, v, 26, 1)
    code = write_bits(code, 0b01, 24, 2)
    code = write_bits(code, opc, 22, 2)
    code = write_bits(code, get_ldst_uimm12_value(
        inst, 2, fixups, fixup_kind), 10, 12)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_ldst_pair_offset_inst_code(inst: MCInst, opc, v, is_load, fixups):
    code = 0
    code = write_bits(code, opc, 30, 2)
    code = write_bits(code, 0b101, 27, 3)
    code = write_bits(code, v, 26, 1)
    code = write_bits(code, 0b010, 23, 3)
    code = write_bits(code, is_load, 22, 1)
    code = write_bits(code, inst.operands[3].imm >> 3, 15, 7)
    code = write_bits(code, get_reg_code(inst.operands[1]), 10, 5)
    code = write_bits(code, get_reg_code(inst.operands[2]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_ldst_pair_pre_idx_inst_code(inst: MCInst, opc, v, is_load, fixups):
    code = 0
    code = write_bits(code, opc, 30, 2)
    code = write_bits(code, 0b101, 27, 3)
    code = write_bits(code, v, 26, 1)
    code = write_bits(code, 0b011, 23, 3)
    code = write_bits(code, is_load, 22, 1)
    code = write_bits(code, inst.operands[3].imm >> 3, 15, 7)
    code = write_bits(code, get_reg_code(inst.operands[1]), 10, 5)
    code = write_bits(code, get_reg_code(inst.operands[2]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_ldst_pair_post_idx_inst_code(inst: MCInst, opc, v, is_load, fixups):
    code = 0
    code = write_bits(code, opc, 30, 2)
    code = write_bits(code, 0b101, 27, 3)
    code = write_bits(code, v, 26, 1)
    code = write_bits(code, 0b001, 23, 3)
    code = write_bits(code, is_load, 22, 1)
    code = write_bits(code, inst.operands[3].imm >> 3, 15, 7)
    code = write_bits(code, get_reg_code(inst.operands[1]), 10, 5)
    code = write_bits(code, get_reg_code(inst.operands[2]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_addsub_imm(inst: MCInst, is_sub, set_flags, fixups):
    code = 0

    code = write_bits(code, is_sub, 30, 1)
    code = write_bits(code, set_flags, 29, 1)
    code = write_bits(code, 0b10001, 24, 5)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_addsub_imm_value(inst: MCInst, idx, fixups):
    if inst.operands[idx].is_imm:
        return inst.operands[idx].imm

    assert(inst.operands[idx].is_expr)

    expr = inst.operands[idx].expr

    fixup_kind = AArch64FixupKind.AArch64_ADD_IMM12

    fixups.append(MCFixup(0, expr, fixup_kind))

    return 0


def get_addsub_imm_shift(inst: MCInst, is_sub, set_flags, fixups):
    code = get_addsub_imm(inst, is_sub, set_flags, fixups)

    imm = get_addsub_imm_value(inst, 2, fixups)

    # code = write_bits(code, read_bits(imm, 12, 2), 22, 2)
    code = write_bits(code, read_bits(imm, 0, 12), 10, 12)

    return code


def get_addsubs_reg(inst: MCInst, is_sub, set_flags, fixups):
    code = 0

    # shift = inst.operands[3].imm
    shift = 0

    code = write_bits(code, is_sub, 30, 1)
    code = write_bits(code, set_flags, 29, 1)
    code = write_bits(code, 0b01011, 24, 5)
    code = write_bits(code, read_bits(shift, 6, 2), 22, 2)
    code = write_bits(code, 0, 21, 1)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, read_bits(shift, 0, 6), 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_div(inst: MCInst, is_signed, fixups):
    code = get_two_operand(inst, 0b0010, fixups)
    code = write_bits(code, is_signed, 10, 1)

    return code


def mul_accum(inst: MCInst, is_sub, op, fixups):
    code = 0

    code = write_bits(code, 0b0011011, 24, 7)
    code = write_bits(code, op, 21, 3)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, is_sub, 15, 1)
    code = write_bits(code, get_reg_code(inst.operands[3]), 10, 5)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_logical_imm(inst: MCInst, opc, fixups):
    code = 0

    code = write_bits(code, opc, 29, 2)
    code = write_bits(code, 0b100100, 23, 6)

    imm = inst.operands[2].imm

    code = write_bits(code, read_bits(imm, 12, 1), 22, 1)
    code = write_bits(code, read_bits(imm, 6, 6), 16, 6)
    code = write_bits(code, read_bits(imm, 0, 6), 10, 6)

    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_logical_sreg(inst: MCInst, opc, n, fixups):
    code = 0

    # shift = inst.operands[3].imm
    shift = 0

    code = write_bits(code, opc, 29, 2)
    code = write_bits(code, 0b01010, 24, 5)
    code = write_bits(code, read_bits(shift, 6, 2), 22, 2)
    code = write_bits(code, n, 21, 1)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, read_bits(shift, 0, 6), 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_insert_immediate(inst: MCInst, opc, fixups):
    code = 0

    code = write_bits(code, opc, 29, 2)
    code = write_bits(code, 0b100101, 23, 6)

    imm = inst.operands[2].imm
    shift = inst.operands[3].imm

    code = write_bits(code, read_bits(shift, 4, 2), 21, 2)
    code = write_bits(code, imm, 5, 16)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_move_immediate(inst: MCInst, opc, fixups):
    code = 0

    code = write_bits(code, opc, 29, 2)
    code = write_bits(code, 0b100101, 23, 6)

    imm = inst.operands[1].imm
    shift = inst.operands[2].imm

    code = write_bits(code, read_bits(shift, 4, 2), 21, 2)
    code = write_bits(code, imm, 5, 16)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_bitfield_imm(inst: MCInst, opc, fixups):
    code = 0

    code = write_bits(code, opc, 29, 2)
    code = write_bits(code, 0b100110, 23, 6)

    immr = inst.operands[2].imm
    imms = inst.operands[3].imm

    code = write_bits(code, immr, 16, 6)
    code = write_bits(code, imms, 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_adr_label_value(inst: MCInst, idx, fixups):
    if inst.operands[idx].is_imm:
        return inst.operands[idx].imm

    assert(inst.operands[idx].is_expr)

    expr = inst.operands[idx].expr

    if inst.opcode == AArch64MachineOps.ADR:
        fixup_kind = AArch64FixupKind.AArch64_PCREL_ADR_IMM21
    else:
        fixup_kind = AArch64FixupKind.AArch64_PCREL_ADRP_IMM21

    fixups.append(MCFixup(0, expr, fixup_kind))

    return 0


def get_adri_inst_code(inst: MCInst, page, fixups):
    code = 0

    label = get_adr_label_value(inst, 1, fixups)

    code = write_bits(code, page, 31, 1)
    code = write_bits(code, read_bits(label, 0, 2), 29, 2)
    code = write_bits(code, 0b10000, 24, 5)
    code = write_bits(code, read_bits(label, 2, 19), 5, 19)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_branch_cond_target_value(inst: MCInst, idx, fixups):
    if inst.operands[idx].is_imm:
        return inst.operands[idx].imm

    assert(inst.operands[idx].is_expr)

    expr = inst.operands[idx].expr

    fixups.append(MCFixup(0, expr, AArch64FixupKind.AArch64_PCREL_BRANCH19))

    return 0


def get_branch_target_value(inst: MCInst, idx, fixups):
    if inst.operands[idx].is_imm:
        return inst.operands[idx].imm

    assert(inst.operands[idx].is_expr)

    expr = inst.operands[idx].expr

    if inst.opcode == AArch64MachineOps.BL:
        fixup_kind = AArch64FixupKind.AArch64_PCREL_CALL26
    else:
        fixup_kind = AArch64FixupKind.AArch64_PCREL_BRANCH26

    fixups.append(MCFixup(0, expr, fixup_kind))

    return 0


def get_branch_cond(inst: MCInst, fixups):
    code = 0

    target = get_branch_cond_target_value(inst, 0, fixups)
    cond = inst.operands[1].imm

    code = write_bits(code, 0b01010100, 24, 8)
    code = write_bits(code, target, 5, 19)
    code = write_bits(code, 0, 4, 1)
    code = write_bits(code, cond, 0, 4)

    return code


def get_branch_imm(inst: MCInst, op, fixups):
    code = 0

    addr = get_branch_target_value(inst, 0, fixups)

    code = write_bits(code, op, 31, 1)
    code = write_bits(code, 0b00101, 26, 5)
    code = write_bits(code, addr, 0, 26)

    return code


def get_call_imm(inst: MCInst, op, fixups):
    return get_branch_imm(inst, op, fixups)


def get_branch_reg(inst: MCInst, opc, fixups):
    code = 0

    code = write_bits(code, 0b1101011, 25, 7)
    code = write_bits(code, opc, 21, 4)
    code = write_bits(code, 0b11111, 16, 5)
    code = write_bits(code, 0b000000, 10, 6)
    code = write_bits(code, 0b00000, 0, 5)

    return code


def get_single_operand_fp_data(inst: MCInst, opcode, fixups):
    code = 0

    code = write_bits(code, 0b00011110, 24, 8)
    code = write_bits(code, 0b1, 21, 1)
    code = write_bits(code, opcode, 15, 6)
    code = write_bits(code, 0b10000, 10, 5)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_unscaled_conversion(inst: MCInst, rmode, opcode, fixups):
    code = 0

    code = write_bits(code, 0b0011110, 24, 7)
    code = write_bits(code, 0b1, 21, 1)
    code = write_bits(code, rmode, 19, 2)
    code = write_bits(code, opcode, 16, 3)
    code = write_bits(code, 0b000000, 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_two_operand_fp_data(inst: MCInst, opcode, fixups):
    code = 0

    code = write_bits(code, 0b00011110, 24, 8)
    code = write_bits(code, 0b1, 21, 1)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, opcode, 12, 4)
    code = write_bits(code, 0b10, 10, 2)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_fp_comparison(inst: MCInst, signal_all_nans, fixups):
    code = 0

    code = write_bits(code, 0b00011110, 24, 8)
    code = write_bits(code, 0b1, 21, 1)
    code = write_bits(code, get_reg_code(inst.operands[1]), 16, 5)
    code = write_bits(code, 0b001000, 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[0]), 5, 5)
    code = write_bits(code, signal_all_nans, 4, 1)
    code = write_bits(code, 0b0000, 0, 4)

    return code


def get_cond_select_op(inst: MCInst, op, op2, fixups):
    code = 0

    cond = inst.operands[3].imm

    code = write_bits(code, op, 30, 1)
    code = write_bits(code, 0b011010100, 21, 9)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, cond, 12, 4)
    code = write_bits(code, op2, 10, 2)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_simd_three_vec(inst: MCInst, q, u, size, opcode, fixups):
    code = 0

    code = write_bits(code, 0, 31, 1)
    code = write_bits(code, q, 30, 1)
    code = write_bits(code, u, 29, 1)
    code = write_bits(code, 0b01110, 24, 5)
    code = write_bits(code, size, 21, 3)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, opcode, 11, 5)
    code = write_bits(code, 1, 10, 1)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_simd_dup(inst: MCInst, q, op, fixups):
    code = 0

    code = write_bits(code, 0, 31, 1)
    code = write_bits(code, q, 30, 1)
    code = write_bits(code, op, 29, 1)
    code = write_bits(code, 0b01110000, 21, 8)
    code = write_bits(code, 0, 15, 1)
    code = write_bits(code, 1, 10, 1)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_simd_ins(inst: MCInst, q, op, fixups):
    code = 0

    code = write_bits(code, 0, 31, 1)
    code = write_bits(code, q, 30, 1)
    code = write_bits(code, op, 29, 1)
    code = write_bits(code, 0b01110000, 21, 8)
    code = write_bits(code, 0, 15, 1)
    code = write_bits(code, 1, 10, 1)
    code = write_bits(code, get_reg_code(inst.operands[3]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_simd_dup_from_elem(inst: MCInst, q, fixups):
    code = get_simd_dup(inst, q, 0, fixups)
    code = write_bits(code, 0b0000, 11, 4)
    return code


def get_simd_ins_from_elem(inst: MCInst, q, fixups):
    return get_simd_ins(inst, 1, 1, fixups)


def get_system(inst: MCInst, l, fixups):
    code = 0
    code = write_bits(code, 0b1101010100, 22, 10)
    code = write_bits(code, l, 21, 1)

    return code


def fp_conversion(inst: MCInst, ty, op, fixups):
    code = 0

    code = write_bits(code, 0b00011110, 24, 8)
    code = write_bits(code, ty, 22, 2)
    code = write_bits(code, 0b10001, 17, 5)
    code = write_bits(code, op, 15, 2)
    code = write_bits(code, 0b10000, 10, 5)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def fp_to_integer_unscaled(inst: MCInst, rmode, op, ty, fixups):
    code = 0

    code = write_bits(code, 0b00, 29, 2)
    code = write_bits(code, 0b11110, 24, 5)
    code = write_bits(code, ty, 22, 2)
    code = write_bits(code, 1, 21, 1)
    code = write_bits(code, rmode, 19, 2)
    code = write_bits(code, op, 16, 3)
    code = write_bits(code, 0, 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def integer_to_fp_unscaled(inst: MCInst, is_unsigned, ty, fixups):
    code = 0

    code = write_bits(code, 0b0011110, 24, 7)
    code = write_bits(code, ty, 22, 2)
    code = write_bits(code, 0b10001, 17, 5)
    code = write_bits(code, is_unsigned, 16, 1)
    code = write_bits(code, 0b000000, 10, 6)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def get_two_operand(inst: MCInst, op, fixups):
    code = 0

    code = write_bits(code, 0b0011010110, 21, 10)
    code = write_bits(code, get_reg_code(inst.operands[2]), 16, 5)
    code = write_bits(code, 0b00, 14, 2)
    code = write_bits(code, op, 10, 4)
    code = write_bits(code, get_reg_code(inst.operands[1]), 5, 5)
    code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)

    return code


def shift(inst: MCInst, shift_ty, fixups):
    code = get_two_operand(inst, 0b1000, fixups)
    code = write_bits(code, shift_ty, 10, 2)

    return code


def get_inst_binary_code(inst: MCInst, fixups):
    opcode = inst.opcode
    num_operands = len(inst.operands)

    if opcode == AArch64MachineOps.STURWi:
        return get_ldst_unscaled_inst_code(inst, 0b10, 0, 0b00, fixups)

    if opcode == AArch64MachineOps.STURXi:
        return get_ldst_unscaled_inst_code(inst, 0b11, 0, 0b00, fixups)

    if opcode == AArch64MachineOps.STURHi:
        return get_ldst_unscaled_inst_code(inst, 0b01, 1, 0b00, fixups)

    if opcode == AArch64MachineOps.STURSi:
        return get_ldst_unscaled_inst_code(inst, 0b10, 1, 0b00, fixups)

    if opcode == AArch64MachineOps.STURDi:
        return get_ldst_unscaled_inst_code(inst, 0b11, 1, 0b00, fixups)

    if opcode == AArch64MachineOps.STURQi:
        return get_ldst_unscaled_inst_code(inst, 0b00, 1, 0b10, fixups)

    if opcode == AArch64MachineOps.STRBBui:
        return get_ldst_ui_inst_code(inst, 0b00, 0, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED1)

    if opcode == AArch64MachineOps.STRHHui:
        return get_ldst_ui_inst_code(inst, 0b01, 0, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED2)

    if opcode == AArch64MachineOps.STRWui:
        return get_ldst_ui_inst_code(inst, 0b10, 0, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED4)

    if opcode == AArch64MachineOps.STRXui:
        return get_ldst_ui_inst_code(inst, 0b11, 0, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED8)

    if opcode == AArch64MachineOps.STRBui:
        return get_ldst_ui_inst_code(inst, 0b00, 1, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED1)

    if opcode == AArch64MachineOps.STRHui:
        return get_ldst_ui_inst_code(inst, 0b01, 1, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED2)

    if opcode == AArch64MachineOps.STRSui:
        return get_ldst_ui_inst_code(inst, 0b10, 1, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED4)

    if opcode == AArch64MachineOps.STRDui:
        return get_ldst_ui_inst_code(inst, 0b11, 1, 0b00, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED8)

    if opcode == AArch64MachineOps.STRQui:
        return get_ldst_ui_inst_code(inst, 0b00, 1, 0b10, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED16)

    if opcode == AArch64MachineOps.LDURWi:
        return get_ldst_unscaled_inst_code(inst, 0b10, 0, 0b01, fixups)

    if opcode == AArch64MachineOps.LDURXi:
        return get_ldst_unscaled_inst_code(inst, 0b11, 0, 0b01, fixups)

    if opcode == AArch64MachineOps.LDURHi:
        return get_ldst_unscaled_inst_code(inst, 0b01, 1, 0b01, fixups)

    if opcode == AArch64MachineOps.LDURSi:
        return get_ldst_unscaled_inst_code(inst, 0b10, 1, 0b01, fixups)

    if opcode == AArch64MachineOps.LDURDi:
        return get_ldst_unscaled_inst_code(inst, 0b11, 1, 0b01, fixups)

    if opcode == AArch64MachineOps.LDURQi:
        return get_ldst_unscaled_inst_code(inst, 0b00, 1, 0b11, fixups)

    if opcode == AArch64MachineOps.LDRBBui:
        return get_ldst_ui_inst_code(inst, 0b00, 0, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED1)

    if opcode == AArch64MachineOps.LDRHHui:
        return get_ldst_ui_inst_code(inst, 0b01, 0, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED2)

    if opcode == AArch64MachineOps.LDRWui:
        return get_ldst_ui_inst_code(inst, 0b10, 0, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED4)

    if opcode == AArch64MachineOps.LDRXui:
        return get_ldst_ui_inst_code(inst, 0b11, 0, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED8)

    if opcode == AArch64MachineOps.LDRBui:
        return get_ldst_ui_inst_code(inst, 0b00, 1, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED1)

    if opcode == AArch64MachineOps.LDRHui:
        return get_ldst_ui_inst_code(inst, 0b01, 1, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED2)

    if opcode == AArch64MachineOps.LDRSui:
        return get_ldst_ui_inst_code(inst, 0b10, 1, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED4)

    if opcode == AArch64MachineOps.LDRDui:
        return get_ldst_ui_inst_code(inst, 0b11, 1, 0b01, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED8)

    if opcode == AArch64MachineOps.LDRQui:
        return get_ldst_ui_inst_code(inst, 0b00, 1, 0b11, fixups, AArch64FixupKind.AArch64_LDST_IMM12_UNSCALED16)

    if opcode == AArch64MachineOps.STPWi:
        return get_ldst_pair_offset_inst_code(inst, 0b00, 0, 0, fixups)

    if opcode == AArch64MachineOps.STPXi:
        return get_ldst_pair_offset_inst_code(inst, 0b10, 0, 0, fixups)

    if opcode == AArch64MachineOps.STPDi:
        return get_ldst_pair_offset_inst_code(inst, 0b01, 1, 0, fixups)

    if opcode == AArch64MachineOps.STPXprei:
        return get_ldst_pair_pre_idx_inst_code(inst, 0b10, 0, 0, fixups)

    if opcode == AArch64MachineOps.STPXposti:
        return get_ldst_pair_post_idx_inst_code(inst, 0b10, 0, 0, fixups)

    if opcode == AArch64MachineOps.LDPWi:
        return get_ldst_pair_offset_inst_code(inst, 0b00, 0, 1, fixups)

    if opcode == AArch64MachineOps.LDPXi:
        return get_ldst_pair_offset_inst_code(inst, 0b10, 0, 1, fixups)

    if opcode == AArch64MachineOps.LDPSi:
        return get_ldst_pair_offset_inst_code(inst, 0b00, 1, 1, fixups)

    if opcode == AArch64MachineOps.LDPDi:
        return get_ldst_pair_offset_inst_code(inst, 0b01, 1, 1, fixups)

    if opcode == AArch64MachineOps.LDPQi:
        return get_ldst_pair_offset_inst_code(inst, 0b10, 1, 1, fixups)

    if opcode == AArch64MachineOps.LDPXprei:
        return get_ldst_pair_pre_idx_inst_code(inst, 0b10, 0, 1, fixups)

    if opcode == AArch64MachineOps.LDPXposti:
        return get_ldst_pair_post_idx_inst_code(inst, 0b10, 0, 1, fixups)

    if opcode == AArch64MachineOps.ADDWri:
        code = get_addsub_imm_shift(inst, 0, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.ADDXri:
        code = get_addsub_imm_shift(inst, 0, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SUBWri:
        code = get_addsub_imm_shift(inst, 1, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SUBXri:
        code = get_addsub_imm_shift(inst, 1, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.MADDWrrr:
        code = mul_accum(inst, 0, 0b000, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.MADDXrrr:
        code = mul_accum(inst, 0, 0b000, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.MSUBWrrr:
        code = mul_accum(inst, 1, 0b000, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.MSUBXrrr:
        code = mul_accum(inst, 1, 0b000, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ADDWrs:
        code = get_addsubs_reg(inst, 0, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.ADDXrs:
        code = get_addsubs_reg(inst, 0, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SUBWrs:
        code = get_addsubs_reg(inst, 1, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SUBXrs:
        code = get_addsubs_reg(inst, 1, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ADDSWrs:
        code = get_addsubs_reg(inst, 0, 1, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.ADDSXrs:
        code = get_addsubs_reg(inst, 0, 1, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SUBSWrs:
        code = get_addsubs_reg(inst, 1, 1, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SUBSXrs:
        code = get_addsubs_reg(inst, 1, 1, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.UDIVWrr:
        code = get_div(inst, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.UDIVXrr:
        code = get_div(inst, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SDIVWrr:
        code = get_div(inst, 1, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SDIVXrr:
        code = get_div(inst, 1, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ORRWri:
        code = get_logical_imm(inst, 0b01, fixups)
        code = write_bits(code, 0, 31, 1)
        code = write_bits(code, 0, 22, 1)
        return code

    if opcode == AArch64MachineOps.ORRWrr:
        code = get_logical_sreg(inst, 0b01, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.ORRXrr:
        code = get_logical_sreg(inst, 0b01, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ANDWrr:
        code = get_logical_sreg(inst, 0b00, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.ANDXrr:
        code = get_logical_sreg(inst, 0b00, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.EORWrr:
        code = get_logical_sreg(inst, 0b10, 0, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.EORXrr:
        code = get_logical_sreg(inst, 0b10, 0, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.MOVKWi:
        code = get_insert_immediate(inst, 0b11, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.MOVKXi:
        code = get_insert_immediate(inst, 0b11, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SBFMWri:
        code = get_bitfield_imm(inst, 0b00, fixups)
        code = write_bits(code, 0, 31, 1)
        code = write_bits(code, 0, 22, 1)
        code = write_bits(code, 0, 21, 1)
        code = write_bits(code, 0, 15, 1)
        return code

    if opcode == AArch64MachineOps.SBFMXri:
        code = get_bitfield_imm(inst, 0b00, fixups)
        code = write_bits(code, 1, 31, 1)
        code = write_bits(code, 1, 22, 1)
        return code

    if opcode == AArch64MachineOps.UBFMWri:
        code = get_bitfield_imm(inst, 0b10, fixups)
        code = write_bits(code, 0, 31, 1)
        code = write_bits(code, 0, 22, 1)
        code = write_bits(code, 0, 21, 1)
        code = write_bits(code, 0, 15, 1)
        return code

    if opcode == AArch64MachineOps.UBFMXri:
        code = get_bitfield_imm(inst, 0b10, fixups)
        code = write_bits(code, 1, 31, 1)
        code = write_bits(code, 1, 22, 1)
        return code

    if opcode == AArch64MachineOps.MOVZWi:
        code = get_move_immediate(inst, 0b10, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.MOVZXi:
        code = get_move_immediate(inst, 0b10, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ADR:
        return get_adri_inst_code(inst, 0, fixups)

    if opcode == AArch64MachineOps.ADRP:
        return get_adri_inst_code(inst, 1, fixups)

    if opcode == AArch64MachineOps.Bcc:
        return get_branch_cond(inst, fixups)

    if opcode == AArch64MachineOps.B:
        return get_branch_imm(inst, 0, fixups)

    if opcode == AArch64MachineOps.BL:
        return get_call_imm(inst, 1, fixups)

    if opcode == AArch64MachineOps.BLR:
        code = get_branch_reg(inst, 0b0001, fixups)
        code = write_bits(code, get_reg_code(inst.operands[0]), 5, 5)
        return code

    if opcode == AArch64MachineOps.RET:
        code = get_branch_reg(inst, 0b0010, fixups)
        code = write_bits(code, get_reg_code(inst.operands[0]), 5, 5)
        return code

    if opcode == AArch64MachineOps.FMOVHr:
        code = get_single_operand_fp_data(inst, 0b000000, fixups)
        code = write_bits(code, 0b11, 22, 2)
        return code

    if opcode == AArch64MachineOps.FMOVSr:
        code = get_single_operand_fp_data(inst, 0b000000, fixups)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FMOVDr:
        code = get_single_operand_fp_data(inst, 0b000000, fixups)
        code = write_bits(code, 0b01, 22, 2)
        return code

    if opcode == AArch64MachineOps.FMOVWSr:
        code = get_unscaled_conversion(inst, 0b00, 0b111, fixups)
        code = write_bits(code, 0, 31, 1)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FADDHrr:
        code = get_two_operand_fp_data(inst, 0b0010, fixups)
        code = write_bits(code, 0b11, 22, 2)
        return code

    if opcode == AArch64MachineOps.FADDSrr:
        code = get_two_operand_fp_data(inst, 0b0010, fixups)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FADDDrr:
        code = get_two_operand_fp_data(inst, 0b0010, fixups)
        code = write_bits(code, 0b01, 22, 2)
        return code

    if opcode == AArch64MachineOps.FSUBHrr:
        code = get_two_operand_fp_data(inst, 0b0011, fixups)
        code = write_bits(code, 0b11, 22, 2)
        return code

    if opcode == AArch64MachineOps.FSUBSrr:
        code = get_two_operand_fp_data(inst, 0b0011, fixups)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FSUBDrr:
        code = get_two_operand_fp_data(inst, 0b0011, fixups)
        code = write_bits(code, 0b01, 22, 2)
        return code

    if opcode == AArch64MachineOps.FMULHrr:
        code = get_two_operand_fp_data(inst, 0b0000, fixups)
        code = write_bits(code, 0b11, 22, 2)
        return code

    if opcode == AArch64MachineOps.FMULSrr:
        code = get_two_operand_fp_data(inst, 0b0000, fixups)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FMULDrr:
        code = get_two_operand_fp_data(inst, 0b0000, fixups)
        code = write_bits(code, 0b01, 22, 2)
        return code

    if opcode == AArch64MachineOps.FDIVHrr:
        code = get_two_operand_fp_data(inst, 0b0001, fixups)
        code = write_bits(code, 0b11, 22, 2)
        return code

    if opcode == AArch64MachineOps.FDIVSrr:
        code = get_two_operand_fp_data(inst, 0b0001, fixups)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FDIVDrr:
        code = get_two_operand_fp_data(inst, 0b0001, fixups)
        code = write_bits(code, 0b01, 22, 2)
        return code

    if opcode == AArch64MachineOps.FCMPHrr:
        code = get_fp_comparison(inst, 0, fixups)
        code = write_bits(code, 0b11, 22, 2)
        return code

    if opcode == AArch64MachineOps.FCMPSrr:
        code = get_fp_comparison(inst, 0, fixups)
        code = write_bits(code, 0b00, 22, 2)
        return code

    if opcode == AArch64MachineOps.FCMPDrr:
        code = get_fp_comparison(inst, 0, fixups)
        code = write_bits(code, 0b01, 22, 2)
        return code

    if opcode == AArch64MachineOps.CSINCWr:
        code = get_cond_select_op(inst, 0, 0b01, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.CSINCXr:
        code = get_cond_select_op(inst, 0, 0b01, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ORRv16i8:
        code = get_simd_three_vec(inst, 1, 0, 0b101, 0b00011, fixups)
        return code

    if opcode == AArch64MachineOps.FADDv4f32:
        code = get_simd_three_vec(inst, 1, 0, 0b001, 0b11010, fixups)
        return code

    if opcode == AArch64MachineOps.FSUBv4f32:
        code = get_simd_three_vec(inst, 1, 0, 0b101, 0b11010, fixups)
        return code

    if opcode == AArch64MachineOps.FMULv4f32:
        code = get_simd_three_vec(inst, 1, 1, 0b001, 0b11011, fixups)
        return code

    if opcode == AArch64MachineOps.FDIVv4f32:
        code = get_simd_three_vec(inst, 1, 1, 0b001, 0b11111, fixups)
        return code

    # if opcode == AArch64MachineOps.DUPv2i32lane:
    #     code = get_simd_dup_from_elem(inst, 0, fixups)
        # idx = inst.operands[1].imm
        # code = write_bits(code, idx, 18, 3)
        # code = write_bits(code, 0b10, 16, 2)
    #     return code

    if opcode == AArch64MachineOps.DUPv4i32lane:
        code = get_simd_dup_from_elem(inst, 1, fixups)
        idx = inst.operands[2].imm
        code = write_bits(code, idx, 19, 2)
        code = write_bits(code, 0b100, 16, 3)
        return code

    if opcode == AArch64MachineOps.INSvi32lane:
        code = get_simd_ins_from_elem(inst, 1, fixups)
        idx = inst.operands[2].imm
        idx2 = inst.operands[4].imm
        code = write_bits(code, idx, 19, 2)
        code = write_bits(code, 0b100, 16, 3)
        code = write_bits(code, idx2, 13, 2)
        return code

    if opcode == AArch64MachineOps.MRS:
        code = get_system(inst, 1, fixups)
        code = write_bits(code, get_sysreg_code(inst.operands[1]), 5, 16)
        code = write_bits(code, get_reg_code(inst.operands[0]), 0, 5)
        return code

    if opcode == AArch64MachineOps.FCVTSHr:
        code = fp_conversion(inst, 0b11, 0b00, fixups)
        return code

    if opcode == AArch64MachineOps.FCVTDHr:
        code = fp_conversion(inst, 0b11, 0b01, fixups)
        return code

    if opcode == AArch64MachineOps.FCVTHSr:
        code = fp_conversion(inst, 0b00, 0b11, fixups)
        return code

    if opcode == AArch64MachineOps.FCVTDSr:
        code = fp_conversion(inst, 0b00, 0b01, fixups)
        return code

    if opcode == AArch64MachineOps.FCVTHDr:
        code = fp_conversion(inst, 0b01, 0b11, fixups)
        return code

    if opcode == AArch64MachineOps.FCVTSDr:
        code = fp_conversion(inst, 0b01, 0b00, fixups)
        return code

    if opcode == AArch64MachineOps.FCVTZSUWHr:
        code = fp_to_integer_unscaled(inst, 0b11, 0b000, 0b11, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.FCVTZSUXHr:
        code = fp_to_integer_unscaled(inst, 0b11, 0b000, 0b11, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.FCVTZSUWSr:
        code = fp_to_integer_unscaled(inst, 0b11, 0b000, 0b00, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.FCVTZSUXSr:
        code = fp_to_integer_unscaled(inst, 0b11, 0b000, 0b00, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.FCVTZSUWDr:
        code = fp_to_integer_unscaled(inst, 0b11, 0b000, 0b01, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.FCVTZSUXDr:
        code = fp_to_integer_unscaled(inst, 0b11, 0b000, 0b01, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SCVTFUWHr:
        code = integer_to_fp_unscaled(inst, 0, 0b11, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SCVTFUWSr:
        code = integer_to_fp_unscaled(inst, 0, 0b00, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SCVTFUWDr:
        code = integer_to_fp_unscaled(inst, 0, 0b01, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.SCVTFUXHr:
        code = integer_to_fp_unscaled(inst, 0, 0b11, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SCVTFUXSr:
        code = integer_to_fp_unscaled(inst, 0, 0b00, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.SCVTFUXDr:
        code = integer_to_fp_unscaled(inst, 0, 0b01, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.UCVTFUWHr:
        code = integer_to_fp_unscaled(inst, 1, 0b11, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.UCVTFUWSr:
        code = integer_to_fp_unscaled(inst, 1, 0b00, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.UCVTFUWDr:
        code = integer_to_fp_unscaled(inst, 1, 0b01, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.UCVTFUXHr:
        code = integer_to_fp_unscaled(inst, 1, 0b11, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.UCVTFUXSr:
        code = integer_to_fp_unscaled(inst, 1, 0b00, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.UCVTFUXDr:
        code = integer_to_fp_unscaled(inst, 1, 0b01, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.ASRVWrr:
        code = shift(inst, 0b10, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.ASRVXrr:
        code = shift(inst, 0b10, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.LSLVWrr:
        code = shift(inst, 0b00, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.LSLVXrr:
        code = shift(inst, 0b00, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    if opcode == AArch64MachineOps.LSRVWrr:
        code = shift(inst, 0b01, fixups)
        code = write_bits(code, 0, 31, 1)
        return code

    if opcode == AArch64MachineOps.LSRVXrr:
        code = shift(inst, 0b01, fixups)
        code = write_bits(code, 1, 31, 1)
        return code

    raise NotImplementedError()


class AArch64CodeEmitter(MCCodeEmitter):
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

        if inst.opcode == AArch64MachineOps.TLSDESCCALL:
            fixup_kind = AArch64FixupKind.AArch64_TLSDESC_CALL
            expr = inst.operands[0].expr
            fixups.append(MCFixup(0, expr, fixup_kind))

            return

        code = get_inst_binary_code(inst, fixups)

        self.emit_constant(code, 4, output)
