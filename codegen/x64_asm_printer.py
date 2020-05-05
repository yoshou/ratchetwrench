#!/usr/bin/env python
# -*- coding: utf-8 -*-

from codegen.mir import *
from codegen.spec import *
from codegen.passes import *
from codegen.mc import *
from codegen.asm_emitter import *
from codegen.assembler import *
from ir.values import *
from codegen.x64_gen import X64MachineOps, RIP, EIP
import zlib
import io


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


class X64MCInstLower:
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
        from codegen.x64_gen import X64OperandFlag

        ref_kind = MCVariantKind.Non
        if operand.target_flags & X64OperandFlag.SECREL:
            ref_kind = MCVariantKind.SECREL

        expr = MCSymbolRefExpr(symbol, ref_kind)

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


class X64IntelInstPrinter(MCInstPrinter):
    def __init__(self):
        super().__init__()

    def print_inst(self, inst: MCInst, output):
        opcodes = {
            X64MachineOps.MOV32mi: "mov",
            X64MachineOps.MOV32mr: "mov",
            X64MachineOps.MOV32rm: "mov",
            X64MachineOps.MOV32rr: "mov",

            X64MachineOps.MOV64mi: "mov",
            X64MachineOps.MOV64mr: "mov",
            X64MachineOps.MOV64rm: "mov",
            X64MachineOps.MOV64rr: "mov",

            X64MachineOps.LEA32r: "lea",
            X64MachineOps.LEA64r: "lea",

            X64MachineOps.PUSH32r: "push",
            X64MachineOps.PUSH64r: "push",
            X64MachineOps.POP32r: "pop",
            X64MachineOps.POP64r: "pop",

            X64MachineOps.ADD32mi: "add",
            X64MachineOps.ADD32ri: "add",
            X64MachineOps.ADD32rm: "add",
            X64MachineOps.ADD32rr: "add",

            X64MachineOps.ADD64mi: "add",
            X64MachineOps.ADD64ri: "add",
            X64MachineOps.ADD64rm: "add",
            X64MachineOps.ADD64rr: "add",

            X64MachineOps.SUB32mi: "sub",
            X64MachineOps.SUB32ri: "sub",
            X64MachineOps.SUB32rm: "sub",
            X64MachineOps.SUB32rr: "sub",

            X64MachineOps.SUB64mi: "sub",
            X64MachineOps.SUB64ri: "sub",
            X64MachineOps.SUB64rm: "sub",
            X64MachineOps.SUB64rr: "sub",

            X64MachineOps.AND32mi: "and",
            X64MachineOps.AND32ri: "and",
            X64MachineOps.AND32rm: "and",
            X64MachineOps.AND32rr: "and",

            X64MachineOps.OR32mi: "or",
            X64MachineOps.OR32ri: "or",
            X64MachineOps.OR32rm: "or",
            X64MachineOps.OR32rr: "or",

            X64MachineOps.XOR32mi: "xor",
            X64MachineOps.XOR32ri: "xor",
            X64MachineOps.XOR32rm: "xor",
            X64MachineOps.XOR32rr: "xor",

            X64MachineOps.SHR32rCL: "shr",
            X64MachineOps.SHR32ri: "shr",
            X64MachineOps.SHL32rCL: "shl",
            X64MachineOps.SHL32ri: "shl",

            X64MachineOps.CMP32ri: "cmp",
            X64MachineOps.CMP32rm: "cmp",
            X64MachineOps.CMP32rr: "cmp",

            # SSE
            X64MachineOps.MOVSSmi: "movss",
            X64MachineOps.MOVSSmr: "movss",
            X64MachineOps.MOVSSrm: "movss",
            X64MachineOps.MOVSSrr: "movss",

            X64MachineOps.JCC_1: "j",
            X64MachineOps.JCC_2: "j",
            X64MachineOps.JCC_4: "j",
            X64MachineOps.JMP_1: "jmp",
            X64MachineOps.JMP_2: "jmp",
            X64MachineOps.JMP_4: "jmp",
            X64MachineOps.CALLpcrel32: "call",
            X64MachineOps.RET: "ret",
        }

        output.write(opcodes[inst.opcode])

        if inst.opcode in [X64MachineOps.JCC_1, X64MachineOps.JCC_2, X64MachineOps.JCC_4]:
            self.print_condcode(inst, 1, output)

        if inst.opcode == X64MachineOps.RET:
            return

        output.write("\t")

        # First operand
        if inst.opcode == X64MachineOps.JCC_1:
            self.print_pc_rel_imm(inst, 0, output)
            return
        if inst.opcode == X64MachineOps.JMP_1:
            self.print_pc_rel_imm(inst, 0, output)
            return
        if inst.opcode == X64MachineOps.CALLpcrel32:
            self.print_pc_rel_imm(inst, 0, output)
            return

        if inst.opcode in [X64MachineOps.PUSH32r, X64MachineOps.PUSH64r]:
            self.print_operand(inst, 0, output)
            return

        if inst.opcode in [X64MachineOps.POP32r, X64MachineOps.POP64r]:
            self.print_operand(inst, 0, output)
            return

        if inst.opcode in [X64MachineOps.MOV32rm, X64MachineOps.MOV64rm]:
            self.print_operand(inst, 0, output)
            output.write(", ")
            if inst.opcode == X64MachineOps.MOV32rm:
                self.print_dwordmem(inst, 1, output)
            elif inst.opcode == X64MachineOps.MOV64rm:
                self.print_qwordmem(inst, 1, output)
            return

        if inst.opcode in [X64MachineOps.MOV32rr, X64MachineOps.MOV64rr]:
            self.print_operand(inst, 0, output)
            output.write(", ")
            self.print_operand(inst, 1, output)
            return

        if inst.opcode in [X64MachineOps.MOV32mi, X64MachineOps.MOV64mi]:
            if inst.opcode == X64MachineOps.MOV32mi:
                self.print_dwordmem(inst, 0, output)
            elif inst.opcode == X64MachineOps.MOV64mi:
                self.print_qwordmem(inst, 0, output)
            output.write(", ")
            self.print_operand(inst, 5, output)
            return

        if inst.opcode in [X64MachineOps.MOV32mr, X64MachineOps.MOV64mr]:
            if inst.opcode == X64MachineOps.MOV32mr:
                self.print_dwordmem(inst, 0, output)
            elif inst.opcode == X64MachineOps.MOV64mr:
                self.print_qwordmem(inst, 0, output)
            output.write(", ")
            self.print_operand(inst, 5, output)
            return

        if inst.opcode in [X64MachineOps.LEA32r, X64MachineOps.LEA64r]:
            self.print_operand(inst, 0, output)
            output.write(", ")
            if inst.opcode == X64MachineOps.LEA32r:
                self.print_dwordmem(inst, 1, output)
            elif inst.opcode == X64MachineOps.LEA64r:
                self.print_qwordmem(inst, 1, output)
            return

        if inst.opcode in [X64MachineOps.SUB32mi, X64MachineOps.ADD32mi, X64MachineOps.AND32mi, X64MachineOps.OR32mi, X64MachineOps.XOR32mi]:
            self.print_dwordmem(inst, 0, output)
            output.write(", ")
            self.print_operand(inst, 5, output)
            return

        if inst.opcode in [X64MachineOps.SUB64mi, X64MachineOps.ADD64mi]:
            self.print_qwordmem(inst, 0, output)
            output.write(", ")
            self.print_operand(inst, 5, output)
            return

        if inst.opcode in [X64MachineOps.SUB32ri, X64MachineOps.SUB64ri, X64MachineOps.ADD32ri, X64MachineOps.ADD64ri, X64MachineOps.AND32ri, X64MachineOps.OR32ri, X64MachineOps.XOR32ri, X64MachineOps.SUB32rr, X64MachineOps.ADD32rr, X64MachineOps.AND32rr, X64MachineOps.OR32rr, X64MachineOps.XOR32rr]:
            self.print_operand(inst, 1, output)
            output.write(", ")
            self.print_operand(inst, 2, output)
            return

        if inst.opcode in [X64MachineOps.CMP32ri, X64MachineOps.CMP32rr]:
            self.print_operand(inst, 0, output)
            output.write(", ")
            self.print_operand(inst, 1, output)
            return

        if inst.opcode in [X64MachineOps.CMP32rm]:
            self.print_operand(inst, 0, output)
            output.write(", ")
            self.print_dwordmem(inst, 1, output)
            return

        if inst.opcode in [X64MachineOps.SHL32rCL, X64MachineOps.SHL32ri, X64MachineOps.SHR32rCL, X64MachineOps.SHR32ri, X64MachineOps.SAR32rCL, X64MachineOps.SAR32ri]:
            self.print_operand(inst, 1, output)
            output.write(", ")
            self.print_operand(inst, 2, output)
            return

        if inst.opcode in [X64MachineOps.MOVSSrm]:
            self.print_operand(inst, 1, output)
            output.write(", ")
            self.print_dwordmem(inst, 1, output)
            return

        raise NotImplementedError()

    def print_mem(self, inst, op, output):
        base = inst.operands[op]
        scale = inst.operands[op + 1]
        index = inst.operands[op + 2]
        disp = inst.operands[op + 3]
        segment = inst.operands[op + 4]

        if segment.reg.spec.name != "noreg":
            self.print_operand(inst, op + 4, output)
            output.write(":")

        output.write("[")

        self.print_operand(inst, op, output)

        if not disp.is_imm:
            output.write(" + ")
            self.print_operand(inst, op + 3, output)
        elif disp.imm != 0:
            disp_val = disp.imm
            if disp_val < 0:
                output.write(" - ")
                disp_val = -disp_val
            else:
                output.write(" + ")
            output.write(str(disp_val))

        if index.reg.spec.name != "noreg":
            output.write(" + (")
            self.print_operand(inst, op + 1, output)
            output.write(" * ")
            self.print_operand(inst, op + 2, output)
            output.write(")")

        output.write("]")

    def print_dwordmem(self, inst, op, output):
        output.write("dword ptr ")
        self.print_mem(inst, op, output)

    def print_qwordmem(self, inst, op, output):
        output.write("qword ptr ")
        self.print_mem(inst, op, output)

    def print_operand(self, inst, op, output):
        if isinstance(inst.operands[op], MCOperandReg):
            output.write(str(inst.operands[op].reg))
            return
        elif isinstance(inst.operands[op], MCOperandImm):
            output.write(str(inst.operands[op].imm))
            return
        elif isinstance(inst.operands[op], MCOperandExpr):
            self.print_expr(inst.operands[op].expr, output)
            return

        raise NotImplementedError()

    def print_expr(self, expr, output):
        if isinstance(expr, MCSymbolRefExpr):
            output.write(expr.symbol.name)
            return

        raise NotImplementedError()

    def print_pc_rel_imm(self, inst, op, output):
        operand = inst.operands[op]
        if isinstance(operand, MCOperandImm):
            raise NotImplementedError()
        else:
            assert(isinstance(operand, MCOperandExpr))

            self.print_expr(operand.expr, output)

    def print_condcode(self, inst, op, output):
        TABLE = {
            0: "o",
            1: "no",
            2: "b",
            3: "ae",
            4: "e",
            5: "ne",
            6: "be",
            7: "a",
            8: "s",
            9: "ns",
            10: "p",
            11: "np",
            12: "l",
            13: "ge",
            14: "le",
            15: "g",
        }

        imm = inst.operands[op].imm

        output.write(TABLE[imm])


class X64AsmInfo(MCAsmInfo):
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


class X64AsmPrinter(AsmPrinter):
    def __init__(self, stream: MCStream):
        super().__init__(stream)

    def emit_linkage(self, value):
        symbol = get_global_symbol(value, self.ctx)
        if value.linkage == GlobalLinkage.Global:
            self.stream.emit_symbol_attrib(symbol, MCSymbolAttribute.Global)

        if value.linkage == GlobalLinkage.External:
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

    def get_global_section_kind(self, value):
        if isinstance(value, Function):
            return SectionKind.Text

        raise NotImplementedError()

    def emit_function_header(self, func):
        self.emit_constant_pool()

        section_kind = self.get_global_section_kind(func.func_info.func)
        self.stream.switch_section(
            self.ctx.obj_file_info.select_section_for_global(section_kind, func.func_info.func, self.ctx))
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

    def emit_instruction(self, inst):
        from codegen.x64_def import RDI, RIP, NOREG
        mc_inst_lower = X64MCInstLower(self.ctx, inst.mbb.func, self)

        if inst.opcode == X64MachineOps.TLSADDR64:
            var_kind = MCVariantKind.TLSGD

            operand = inst.operands[0]
            gv = mc_inst_lower.get_symbol(operand)
            gv_symbol_ref = MCSymbolRefExpr(gv, var_kind)

            lea_inst = MCInst(X64MachineOps.LEA64r)
            lea_inst.add_operand(MCOperandReg(MachineRegister(RDI)))
            lea_inst.add_operand(MCOperandReg(MachineRegister(RIP)))
            lea_inst.add_operand(MCOperandImm(1))
            lea_inst.add_operand(MCOperandReg(MachineRegister(NOREG)))
            lea_inst.add_operand(MCOperandExpr(gv_symbol_ref))
            lea_inst.add_operand(MCOperandReg(MachineRegister(NOREG)))

            self.emit_mc_inst(MCInst(X64MachineOps.DATA16_PREFIX))

            self.emit_mc_inst(lea_inst)

            tls_get_addr = self.ctx.get_or_create_symbol("__tls_get_addr")
            tls_symbol_ref = MCSymbolRefExpr(tls_get_addr, MCVariantKind.PLT)

            mc_inst = MCInst(X64MachineOps.CALLpcrel32)
            mc_inst.add_operand(MCOperandExpr(tls_symbol_ref))

            self.emit_mc_inst(MCInst(X64MachineOps.DATA16_PREFIX))
            self.emit_mc_inst(MCInst(X64MachineOps.DATA16_PREFIX))
            self.emit_mc_inst(MCInst(X64MachineOps.REX64_PREFIX))

            self.emit_mc_inst(mc_inst)

            self.stream.assembler.register_symbol(gv)
            self.stream.assembler.register_symbol(tls_get_addr)
            return

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

        self.emit_visibility(symbol, variable.visibility,
                             not variable.is_declaration)

        initializer = variable.initializer
        if not initializer:
            return

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


def get_relaxed_opcode_br(inst):
    opcode = inst.opcode

    if opcode == X64MachineOps.JCC_1:
        return X64MachineOps.JCC_4
    if opcode == X64MachineOps.JMP_1:
        return X64MachineOps.JMP_4

    return opcode


def get_relaxed_opcode_arith(inst):
    opcode = inst.opcode

    return opcode


def get_relaxed_opcode(inst):
    relaxed_op = get_relaxed_opcode_arith(inst)
    if relaxed_op != inst.opcode:
        return relaxed_op

    return get_relaxed_opcode_br(inst)


def get_fixup_size_by_kind_x64(kind: MCFixupKind):
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
    elif kind in [X64FixupKind.Reloc_RIPRel_4]:
        return 4

    raise ValueError("kind")


class X64AsmBackend(MCAsmBackend):
    def __init__(self):
        super().__init__()

    def apply_fixup(self, fixup: MCFixup, fixed_value: int, contents):
        size = get_fixup_size_by_kind_x64(fixup.kind)
        offset = fixup.offset

        assert(offset < len(contents))
        assert(offset + size <= len(contents))

        order = 'little'

        bys = fixed_value.to_bytes(
            size, byteorder=order, signed=fixed_value < 0)

        contents[offset:(offset+size)] = bys

    def may_need_relaxation(self, inst: MCInst):
        if get_relaxed_opcode_br(inst) != inst.opcode:
            return True

        if get_relaxed_opcode_arith(inst) == inst.opcode:
            return False

        if inst.operands[-1].is_expr:
            return True

        return False

    def relax_instruction(self, inst: MCInst):
        relaxed_op = get_relaxed_opcode(inst)
        inst.opcode = relaxed_op

    def get_fixup_kind_info(self, kind):
        table = {
            X64FixupKind.Reloc_RIPRel_4: MCFixupKindInfo(
                "Reloc_RIPRel_4", 0, 32, MCFixupKindInfoFlag.IsPCRel)
        }

        if kind in table:
            return table[kind]

        return super().get_fixup_kind_info(kind)

    def is_fixup_kind_pcrel(self, fixup):
        kind = fixup.kind
        return self.get_fixup_kind_info(kind).flags & MCFixupKindInfoFlag.IsPCRel == MCFixupKindInfoFlag.IsPCRel


class X64InstForm(Enum):
    RawFrm = auto()
    AddCCFrm = auto()
    AddRegFrm = auto()

    MRMSrcReg = auto()
    MRMDestReg = auto()
    MRMSrcMem = auto()
    MRMDestMem = auto()

    MRMXrCC = auto()

    MRM0m = auto()
    MRM1m = auto()
    MRM2m = auto()
    MRM3m = auto()
    MRM4m = auto()
    MRM5m = auto()
    MRM6m = auto()
    MRM7m = auto()

    MRM0r = auto()
    MRM1r = auto()
    MRM2r = auto()
    MRM3r = auto()
    MRM4r = auto()
    MRM5r = auto()
    MRM6r = auto()
    MRM7r = auto()


class X64RegNo:
    EAX = 0
    ECX = 1
    EDX = 2
    EBX = 3
    ESP = 4
    EBP = 5
    ESI = 6
    EDI = 7


X64TSOpSizeShift = 7
X64TSOpSizeMask = (0x3 << X64TSOpSizeShift)

X64TSAdSizeShift = 9
X64TSAdSizeMask = (0x3 << X64TSAdSizeShift)

X64TSOpPrefixShift = 11
X64TSOpPrefixMask = (0x3 << X64TSOpPrefixShift)

X64TSOpMapShift = 13
X64TSOpMapMask = (0x7 << X64TSOpMapShift)

X64TSREXShift = 16
X64TSREXMask = (0x1 << X64TSREXShift)

X64TSOpEncShift = 28
X64TSOpEncMask = (0x3 << X64TSOpEncShift)

X64TSVEXShift = 38
X64TSVEXMask = (0x1 << X64TSVEXShift)

X64TSVEX_WShift = 38
X64TSVEX_WMask = (0x1 << X64TSVEX_WShift)

X64TSVEX_4VShift = 39
X64TSVEX_4VMask = (0x1 << X64TSVEX_4VShift)


class X64TSFlags(IntFlag):
    Pseudo = 0

    OpSizeFixed = 0 << X64TSOpSizeShift
    OpSize16 = 1 << X64TSOpSizeShift
    OpSize32 = 2 << X64TSOpSizeShift

    PD = 1 << X64TSOpPrefixShift
    XS = 2 << X64TSOpPrefixShift
    XD = 3 << X64TSOpPrefixShift
    PS = 0 << X64TSOpPrefixShift

    OB = 0 << X64TSOpMapShift  # 1 byte
    TB = 1 << X64TSOpMapShift  # 2 bytes
    T8 = 2 << X64TSOpMapShift
    TA = 3 << X64TSOpMapShift
    XOP8 = 4 << X64TSOpMapShift
    XOP9 = 5 << X64TSOpMapShift
    XOPA = 6 << X64TSOpMapShift
    ThreeDNow = 7 << X64TSOpMapShift

    REX_W = 1 << X64TSREXShift

    EncNormal = 0 << X64TSOpEncShift
    EncVEX = 1 << X64TSOpEncShift
    EncXOP = 2 << X64TSOpEncShift
    EncEVEX = 3 << X64TSOpEncShift

    VEX_W = 1 << X64TSVEX_WShift
    VEX_4V = 1 << X64TSVEX_4VShift


def is_rex_extended_reg(operand: MCOperandReg):
    encoding = operand.reg.spec.encoding

    return (encoding >> 3) & 1


def get_mem_operand_pos(tsflags, form):
    if form == X64InstForm.MRMSrcMem:
        return 1
    elif form == X64InstForm.MRMDestMem:
        return 0
    return -1


class X64FixupKind(Enum):
    Reloc_RIPRel_4 = auto()


class X64MCExprVarKind(Enum):
    Non = auto()
    SecRel = auto()


class X64MCExpr(MCTargetExpr):
    def __init__(self, kind, expr):
        super().__init__()

        self.kind = kind
        self.expr = expr

    def evaluate_expr_as_relocatable(self, layout, fixup):
        if self.kind == X64MCExprVarKind.SecRel:
            offset = layout.get_symbol_offset(fixup.value.expr.symbol)
            return MCValue(offset, None, None)

        raise ValueError("Invalid kind")


class X64CodeEmitter(MCCodeEmitter):
    def __init__(self):
        super().__init__()

    def emit_byte(self, value, output):
        output.write(value.to_bytes(1, byteorder="little"))

    def encode_mod_rm_byte(self, mod, reg, rm):
        assert(mod < 4 and reg < 8 and rm < 8)
        return rm | (reg << 3) | (mod << 6)

    def emit_sib_byte(self, ss, idx, base, output):
        self.emit_byte(self.encode_mod_rm_byte(ss, idx, base), output)

    def is_disp8(self, imm):
        return -128 <= imm and imm <= 127

    def is_16bit_mem_operand(self, inst, op_idx):
        disp = inst.operands[op_idx + 3]
        base = inst.operands[op_idx + 0]
        index = inst.operands[op_idx + 2]

    def emit_mem_mod_rm_byte(self, inst, op_idx, reg_op, fixups, output):
        opcode = inst.opcode
        disp = inst.operands[op_idx + 3]
        base = inst.operands[op_idx + 0]
        scale = inst.operands[op_idx + 1]
        index = inst.operands[op_idx + 2]

        basereg = base.reg.spec
        baseregno = self.get_register_number(base)

        is_64bit_mode = True

        if basereg == RIP or basereg == EIP:
            assert(is_64bit_mode)

            self.emit_byte(self.encode_mod_rm_byte(0, reg_op, 5), output)

            immsize = 4
            self.emit_immediate(
                disp, immsize, X64FixupKind.Reloc_RIPRel_4, fixups, output)
            return

        indexreg = index.reg.spec

        if indexreg == NOREG and baseregno != X64RegNo.ESP and ((not is_64bit_mode) or (basereg != NOREG)):
            if basereg == NOREG:
                self.emit_byte(self.encode_mod_rm_byte(0, reg_op, 5), output)
                self.emit_immediate(
                    disp, 4, MCFixupKind.Data_4, fixups, output)
                return

            if baseregno != X64RegNo.EBP:
                if disp.is_imm and disp.imm == 0:
                    self.emit_byte(self.encode_mod_rm_byte(
                        0, reg_op, baseregno), output)
                    return

                # if disp.is_expr:
                #     expr = disp.expr
                #     if isinstance(expr, MCSymbolRefExpr):
                #         fixups.append(MCFixup(0, expr, MCFixupKind.Noop))
                #         self.emit_byte(self.encode_mod_rm_byte(0, reg_op, baseregno), output)
                #         return

            if disp.is_imm:
                if self.is_disp8(disp.imm):
                    self.emit_byte(self.encode_mod_rm_byte(
                        1, reg_op, baseregno), output)
                    self.emit_immediate(
                        disp, 1, MCFixupKind.Data_1, fixups, output)
                    return

            self.emit_byte(self.encode_mod_rm_byte(
                2, reg_op, baseregno), output)
            self.emit_immediate(disp, 4, MCFixupKind.Data_4, fixups, output)
            return

        need_disp32 = False
        need_disp8 = False

        if basereg == NOREG:
            self.emit_byte(self.encode_mod_rm_byte(
                0, reg_op, 4), output)  # Specify in SIB
        elif not disp.is_imm:
            self.emit_byte(self.encode_mod_rm_byte(2, reg_op, 4), output)
        elif disp.imm == 0 and baseregno != X64RegNo.EBP:
            self.emit_byte(self.encode_mod_rm_byte(0, reg_op, 4), output)
        elif self.is_disp8(disp.imm):
            self.emit_byte(self.encode_mod_rm_byte(1, reg_op, 4), output)
            need_disp8 = True
        else:
            self.emit_byte(self.encode_mod_rm_byte(2, reg_op, 4), output)

        ss_table = [0xFFFFFFFF, 0, 1, 0xFFFFFFFF, 2,
                    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 3]
        ss = ss_table[scale.imm]

        if basereg == NOREG:
            if indexreg == NOREG:
                idx = 4
            else:
                idx = indexreg.encoding
            self.emit_sib_byte(ss, idx, 5, output)
        else:
            if indexreg == NOREG:
                idx = 4
            else:
                idx = indexreg.encoding
            self.emit_sib_byte(ss, idx, baseregno, output)

        if need_disp8:
            self.emit_immediate(disp, 1, MCFixupKind.Data_1, fixups, output)
        elif need_disp32 or not disp.is_imm or disp.imm != 0:
            self.emit_immediate(disp, 4, MCFixupKind.Data_4, fixups, output)

    def emit_reg_mod_rm_byte(self, operand, reg_op, output):
        self.emit_byte(self.encode_mod_rm_byte(
            3, reg_op, self.get_register_number(operand)), output)

    def emit_constant(self, value: int, size, output):
        signed = value < 0
        output.write(value.to_bytes(size, signed=signed, byteorder='little'))

    def emit_immediate(self, operand, size, fixup_kind, fixups, output, imm_offset=0):
        if operand.is_imm:
            # expr = MCConstantExpr(operand.imm)
            self.emit_constant(operand.imm, size, output)
            return
        else:
            assert(operand.is_expr)
            expr = operand.expr

        expr = operand.expr

        if fixup_kind in [
                MCFixupKind.PCRel_4,
                X64FixupKind.Reloc_RIPRel_4]:
            imm_offset -= 4

        if fixup_kind == MCFixupKind.PCRel_2:
            imm_offset -= 2

        if fixup_kind == MCFixupKind.PCRel_1:
            imm_offset -= 1

        if imm_offset != 0:
            expr = MCBinaryExpr(MCBinaryOpcode.Add, expr,
                                MCConstantExpr(imm_offset))

        if fixup_kind in [MCFixupKind.Data_4, MCFixupKind.Data_8]:
            if expr.ty == MCExprType.SymbolRef:
                if expr.kind == MCVariantKind.SECREL:
                    fixup_kind = MCFixupKind.SecRel_4
                    # expr = X64MCExpr(X64MCExprVarKind.SecRel, expr)

        offset = output.tell()
        bys = output.getvalue()
        fixups.append(MCFixup(offset, expr, fixup_kind))

        self.emit_constant(0, size, output)

    def get_register_number(self, operand):
        reg = operand.reg
        return reg.spec.encoding & 0x7

    def emit_segment_prefix(self, operand, output):
        from codegen.x64_def import CS, SS, DS, ES, FS, GS
        if operand.reg.spec == CS:
            self.emit_byte(0x2E, output)
        elif operand.reg.spec == SS:
            self.emit_byte(0x36, output)
        elif operand.reg.spec == DS:
            self.emit_byte(0x3E, output)
        elif operand.reg.spec == ES:
            self.emit_byte(0x26, output)
        elif operand.reg.spec == FS:
            self.emit_byte(0x64, output)
        elif operand.reg.spec == GS:
            self.emit_byte(0x65, output)

    def emit_opcode_prefix(self, tsflags, rex, inst, output):
        if tsflags & X64TSOpSizeMask == X64TSFlags.OpSize16:
            self.emit_byte(0x66, output)

        if tsflags & X64TSOpPrefixMask == X64TSFlags.PD:
            self.emit_byte(0x66, output)
        elif tsflags & X64TSOpPrefixMask == X64TSFlags.XS:
            self.emit_byte(0xF3, output)
        elif tsflags & X64TSOpPrefixMask == X64TSFlags.XD:
            self.emit_byte(0xF2, output)

        if rex != 0:
            self.emit_byte(0x40 | rex, output)

        if tsflags & X64TSOpMapMask in [X64TSFlags.TB, X64TSFlags.T8, X64TSFlags.TA, X64TSFlags.ThreeDNow]:
            self.emit_byte(0x0F, output)

    def compute_rex_prefix(self, inst, bias, form, tsflags, mem_op):
        rex = 0
        op_idx = bias

        from codegen.x64_def import SPL, BPL, SIL, DIL

        for operand in inst.operands:
            if not isinstance(operand, MCOperandReg):
                continue

            reg = operand.reg
            if reg.spec in [SPL, BPL, SIL, DIL]:
                rex |= 0x40

        if tsflags & X64TSFlags.REX_W == X64TSFlags.REX_W:
            rex |= (1 << 3)

        if form == X64InstForm.RawFrm:
            pass
        elif form == X64InstForm.AddRegFrm:
            rex |= is_rex_extended_reg(inst.operands[op_idx])
            op_idx += 1
        elif form == X64InstForm.AddCCFrm:
            pass
        elif form in [X64InstForm.MRM0m, X64InstForm.MRM1m, X64InstForm.MRM2m, X64InstForm.MRM3m,
                      X64InstForm.MRM4m, X64InstForm.MRM5m, X64InstForm.MRM6m, X64InstForm.MRM7m]:
            rex |= is_rex_extended_reg(inst.operands[op_idx + 0])  # base
            rex |= is_rex_extended_reg(inst.operands[op_idx + 2]) << 1  # index
            op_idx += 5
        elif form == X64InstForm.MRMDestReg:
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 0
            op_idx += 1
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 2
            op_idx += 1
        elif form == X64InstForm.MRMSrcReg:
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 2
            op_idx += 1
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 0
            op_idx += 1
        elif form in [X64InstForm.MRM0r, X64InstForm.MRM1r, X64InstForm.MRM2r, X64InstForm.MRM3r,
                      X64InstForm.MRM4r, X64InstForm.MRM5r, X64InstForm.MRM6r, X64InstForm.MRM7r, X64InstForm.MRMXrCC]:
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 0
            op_idx += 1
        elif form == X64InstForm.MRMSrcMem:
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 2
            op_idx += 1
            rex |= is_rex_extended_reg(inst.operands[op_idx + 0])  # base
            rex |= is_rex_extended_reg(inst.operands[op_idx + 2]) << 1  # index
            op_idx += 5
        elif form == X64InstForm.MRMDestMem:
            rex |= is_rex_extended_reg(inst.operands[op_idx + 0])  # base
            rex |= is_rex_extended_reg(inst.operands[op_idx + 2]) << 1  # index
            op_idx += 5
            rex |= is_rex_extended_reg(inst.operands[op_idx]) << 2
            op_idx += 1
        else:
            raise NotImplementedError()

        return rex

    def encode_instruction(self, inst: MCInst, fixups, output):
        opcode = inst.opcode
        num_operands = len(inst.operands)

        if opcode == X64MachineOps.MEMBARRIER:
            return

        op_idx = 0

        # flags
        is_pcrel = False

        tsflags = X64TSFlags.Pseudo

        if inst.opcode in [X64MachineOps.PUSH32r, X64MachineOps.PUSH64r]:
            base_opcode = 0x50
            form = X64InstForm.AddRegFrm
        elif inst.opcode in [X64MachineOps.POP32r, X64MachineOps.POP64r]:
            base_opcode = 0x58
            form = X64InstForm.AddRegFrm
        elif inst.opcode in [X64MachineOps.MOV8ri]:
            base_opcode = 0xB0
            form = X64InstForm.AddRegFrm

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.MOV8rr]:
            base_opcode = 0x88
            form = X64InstForm.MRMDestReg
        elif inst.opcode in [X64MachineOps.MOV8rm]:
            base_opcode = 0x8A
            form = X64InstForm.MRMSrcMem
        elif inst.opcode in [X64MachineOps.MOV8mr]:
            base_opcode = 0x88
            form = X64InstForm.MRMDestMem
        elif inst.opcode in [X64MachineOps.MOV16ri]:
            base_opcode = 0xB8
            form = X64InstForm.AddRegFrm

            # flags
            imm_size = 2
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOV16rr]:
            base_opcode = 0x89
            form = X64InstForm.MRMDestReg

            # flags
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOV16rm]:
            base_opcode = 0x8B
            form = X64InstForm.MRMSrcMem

            # flags
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOV16mr]:
            base_opcode = 0x89
            form = X64InstForm.MRMDestMem

            # flags
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOV16mi]:
            base_opcode = 0xC7
            form = X64InstForm.MRM0m

            # flags
            imm_size = 2
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOV32ri, X64MachineOps.MOV64ri]:
            base_opcode = 0xB8
            form = X64InstForm.AddRegFrm

            # flags
            imm_size = 4

            if inst.opcode == X64MachineOps.MOV64ri:
                tsflags |= X64TSFlags.REX_W
                imm_size = 8
        elif inst.opcode in [X64MachineOps.MOV32mi, X64MachineOps.MOV64mi]:
            base_opcode = 0xC7
            form = X64InstForm.MRM0m

            # flags
            imm_size = 4

            if inst.opcode == X64MachineOps.MOV64mi:
                tsflags |= X64TSFlags.REX_W
                imm_size = 8
        elif inst.opcode in [X64MachineOps.MOV32rr, X64MachineOps.MOV64rr]:
            base_opcode = 0x89
            form = X64InstForm.MRMDestReg

            if inst.opcode == X64MachineOps.MOV64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.MOV32rm, X64MachineOps.MOV64rm]:
            base_opcode = 0x8B
            form = X64InstForm.MRMSrcMem

            if inst.opcode == X64MachineOps.MOV64rm:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.MOV32mr, X64MachineOps.MOV64mr]:
            base_opcode = 0x89
            form = X64InstForm.MRMDestMem

            if inst.opcode == X64MachineOps.MOV64mr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.MOVSX32rr8]:
            base_opcode = 0xBE
            form = X64InstForm.MRMSrcReg

            tsflags |= X64TSFlags.TB
        elif inst.opcode in [X64MachineOps.MOVSX32rr16]:
            base_opcode = 0xBF
            form = X64InstForm.MRMSrcReg

            # flags
            tsflags |= X64TSFlags.TB
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOVSX64rr32]:
            base_opcode = 0x63
            form = X64InstForm.MRMSrcReg
        elif inst.opcode in [X64MachineOps.MOVZX16rr8]:
            base_opcode = 0xB6
            form = X64InstForm.MRMSrcReg

            # flags
            tsflags |= X64TSFlags.TB
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.MOVZX32rr8]:
            base_opcode = 0xB6
            form = X64InstForm.MRMSrcReg

            # flags
            tsflags |= X64TSFlags.TB
        elif inst.opcode in [X64MachineOps.MOVZX32rr16]:
            base_opcode = 0xB7
            form = X64InstForm.MRMSrcReg  # TODO: SizeOp

            tsflags |= X64TSFlags.TB
        elif inst.opcode in [X64MachineOps.MOVZX64rr8]:
            base_opcode = 0xB6
            form = X64InstForm.MRMSrcReg  # TODO: SizeOp

            tsflags |= X64TSFlags.TB
            tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.MOVZX64rr16]:
            base_opcode = 0xB7
            form = X64InstForm.MRMSrcReg  # TODO: SizeOp

            tsflags |= X64TSFlags.TB
            tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.MOVSSrm, X64MachineOps.MOVSDrm]:
            base_opcode = 0x10
            form = X64InstForm.MRMSrcMem

            if inst.opcode == X64MachineOps.MOVSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.MOVSDrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVSSmr, X64MachineOps.MOVSDmr]:
            base_opcode = 0x11
            form = X64InstForm.MRMDestMem

            if inst.opcode == X64MachineOps.MOVSSmr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.MOVSDmr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVSSrr, X64MachineOps.MOVSDrr]:
            base_opcode = 0x10
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.MOVSSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.MOVSDrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.VMOVSSrm]:
            base_opcode = 0x10
            form = X64InstForm.MRMSrcMem

            raise NotImplementedError()

            if inst.opcode == X64MachineOps.MOVSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS | X64TSFlags.EncVEX)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVAPSrr]:
            base_opcode = 0x28
            form = X64InstForm.MRMSrcReg

            if inst.opcode == X64MachineOps.MOVAPSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVAPSrm]:
            base_opcode = 0x28
            form = X64InstForm.MRMSrcMem

            if inst.opcode == X64MachineOps.MOVAPSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVAPSmr]:
            base_opcode = 0x29
            form = X64InstForm.MRMDestMem

            if inst.opcode == X64MachineOps.MOVAPSmr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVUPSrm]:
            base_opcode = 0x10
            form = X64InstForm.MRMSrcMem

            if inst.opcode == X64MachineOps.MOVUPSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVUPSmr]:
            base_opcode = 0x11
            form = X64InstForm.MRMDestMem

            if inst.opcode == X64MachineOps.MOVUPSmr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MOVPQIto64rr]:
            base_opcode = 0x7E
            form = X64InstForm.MRMDestReg

            if inst.opcode == X64MachineOps.MOVPQIto64rr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PD)
            else:
                raise ValueError()

            if inst.opcode == X64MachineOps.MOVPQIto64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.ADD32ri, X64MachineOps.ADD64ri]:
            base_opcode = 0x81
            form = X64InstForm.MRM0r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 4

            if inst.opcode == X64MachineOps.ADD64ri:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.OR32ri]:
            base_opcode = 0x81
            form = X64InstForm.MRM1r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 4
        elif inst.opcode in [X64MachineOps.AND32ri]:
            base_opcode = 0x81
            form = X64InstForm.MRM4r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 4
        elif inst.opcode in [X64MachineOps.SUB8ri]:
            base_opcode = 0x80
            form = X64InstForm.MRM5r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.SUB32ri, X64MachineOps.SUB64ri]:
            base_opcode = 0x81
            form = X64InstForm.MRM5r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 4

            if inst.opcode == X64MachineOps.SUB64ri:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.XOR8ri]:
            base_opcode = 0x80
            form = X64InstForm.MRM6r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.XOR32ri]:
            base_opcode = 0x81
            form = X64InstForm.MRM6r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 4
        elif inst.opcode in [X64MachineOps.CMP8ri]:
            base_opcode = 0x80
            form = X64InstForm.MRM7r

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.CMP32ri, X64MachineOps.CMP64ri]:
            base_opcode = 0x81
            form = X64InstForm.MRM7r
            imm_size = 4

            if inst.opcode == X64MachineOps.CMP64ri:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.OR8rr]:
            base_opcode = 0x08
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias
        elif inst.opcode in [X64MachineOps.OR32rr]:
            base_opcode = 0x09
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias
        elif inst.opcode in [X64MachineOps.XOR32rr, X64MachineOps.XOR64rr]:
            base_opcode = 0x31
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.XOR64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.ADDSSrr, X64MachineOps.ADDSDrr]:
            base_opcode = 0x58
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.ADDSSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.ADDSDrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.ADDSSrm, X64MachineOps.ADDSDrm]:
            base_opcode = 0x58
            form = X64InstForm.MRMSrcMem
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.ADDSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.ADDSDrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.DIVSSrr, X64MachineOps.DIVSDrr]:
            base_opcode = 0x5E
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.DIVSSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.DIVSDrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.DIVSSrm, X64MachineOps.DIVSDrm]:
            base_opcode = 0x5E
            form = X64InstForm.MRMSrcMem
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.DIVSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.DIVSDrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MULSSrr, X64MachineOps.MULSDrr]:
            base_opcode = 0x59
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.MULSSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.MULSDrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MULSSrm, X64MachineOps.MULSDrm]:
            base_opcode = 0x59
            form = X64InstForm.MRMSrcMem
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.MULSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.MULSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.SUBSSrr, X64MachineOps.SUBSDrr]:
            base_opcode = 0x5C
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SUBSSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.SUBSDrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.SUBSSrm, X64MachineOps.SUBSDrm]:
            base_opcode = 0x5C
            form = X64InstForm.MRMSrcMem
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SUBSSrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode == X64MachineOps.SUBSDrm:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.CVTSS2SDrr]:
            base_opcode = 0x5A
            form = X64InstForm.MRMSrcReg

            tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
        elif inst.opcode in [X64MachineOps.CVTSD2SSrr]:
            base_opcode = 0x5A
            form = X64InstForm.MRMSrcReg

            tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
        elif inst.opcode in [X64MachineOps.CVTTSS2SIrr, X64MachineOps.CVTTSS2SI64rr, X64MachineOps.CVTTSD2SIrr, X64MachineOps.CVTTSD2SI64rr]:
            base_opcode = 0x2C
            form = X64InstForm.MRMSrcReg

            if inst.opcode in [X64MachineOps.CVTTSS2SIrr, X64MachineOps.CVTTSS2SI64rr]:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode in [X64MachineOps.CVTTSD2SIrr, X64MachineOps.CVTTSD2SI64rr]:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()

            if inst.opcode in [X64MachineOps.CVTTSS2SI64rr, X64MachineOps.CVTTSD2SI64rr]:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.CVTSI2SSrr, X64MachineOps.CVTSI2SDrr, X64MachineOps.CVTSI642SSrr, X64MachineOps.CVTSI642SDrr]:
            base_opcode = 0x2A
            form = X64InstForm.MRMSrcReg

            if inst.opcode in [X64MachineOps.CVTSI2SSrr, X64MachineOps.CVTSI642SSrr]:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XS)
            elif inst.opcode in [X64MachineOps.CVTSI2SDrr, X64MachineOps.CVTSI642SDrr]:
                tsflags |= (X64TSFlags.TB | X64TSFlags.XD)
            else:
                raise ValueError()

            if inst.opcode in [X64MachineOps.CVTSI642SSrr, X64MachineOps.CVTSI642SDrr]:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.ADDPSrr]:
            base_opcode = 0x58
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.ADDPSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.SUBPSrr]:
            base_opcode = 0x5C
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SUBPSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.MULPSrr]:
            base_opcode = 0x59
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.MULPSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.XORPSrr]:
            base_opcode = 0x57
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.XORPSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.ADD16rr]:
            base_opcode = 0x01
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.ADD32rr, X64MachineOps.ADD64rr]:
            base_opcode = 0x01
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.ADD64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.SUB32rr, X64MachineOps.SUB64rr]:
            base_opcode = 0x29
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SUB64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.IMUL16rr]:
            base_opcode = 0xAF
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            tsflags |= X64TSFlags.TB
            tsflags |= X64TSFlags.OpSize16
        elif inst.opcode in [X64MachineOps.IMUL32rr, X64MachineOps.IMUL64rr]:
            base_opcode = 0xAF
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            tsflags |= X64TSFlags.TB

            if inst.opcode == X64MachineOps.IMUL64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.AND32rr]:
            base_opcode = 0x21
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias
        elif inst.opcode in [X64MachineOps.AND8rr]:
            base_opcode = 0x20
            form = X64InstForm.MRMDestReg
            op_idx += 1  # TODO: Should defined as operand bias
        elif inst.opcode in [X64MachineOps.CMP32rr, X64MachineOps.CMP64rr]:
            base_opcode = 0x39
            form = X64InstForm.MRMDestReg

            if inst.opcode == X64MachineOps.CMP64rr:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.CMP32rm]:
            base_opcode = 0x39
            form = X64InstForm.MRMSrcMem

            if inst.opcode == X64MachineOps.CMP64rm:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.DIV32r, X64MachineOps.DIV64r]:
            base_opcode = 0xF7
            form = X64InstForm.MRM6r

            if inst.opcode == X64MachineOps.DIV64r:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.IDIV32r, X64MachineOps.IDIV64r]:
            base_opcode = 0xF7
            form = X64InstForm.MRM7r

            if inst.opcode == X64MachineOps.DIV64r:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.UCOMISSrr, X64MachineOps.UCOMISDrr]:
            base_opcode = 0x2E
            form = X64InstForm.MRMSrcReg

            if inst.opcode == X64MachineOps.UCOMISSrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            elif inst.opcode == X64MachineOps.UCOMISDrr:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PD)
            else:
                raise ValueError()
        elif inst.opcode in [X64MachineOps.SHUFPSrri]:
            base_opcode = 0xC6
            form = X64InstForm.MRMSrcReg
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SHUFPSrri:
                tsflags |= (X64TSFlags.TB | X64TSFlags.PS)
            else:
                raise ValueError()

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.SHL32ri, X64MachineOps.SHL64ri]:
            base_opcode = 0xC1
            form = X64InstForm.MRM4r
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SHL64ri:
                tsflags |= X64TSFlags.REX_W

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.SHL32rCL, X64MachineOps.SHL64rCL]:
            base_opcode = 0xD3
            form = X64InstForm.MRM4r
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SHL64rCL:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.SHR32ri]:
            base_opcode = 0xC1
            form = X64InstForm.MRM5r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.SHR32rCL, X64MachineOps.SHR64rCL]:
            base_opcode = 0xD3
            form = X64InstForm.MRM5r
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SHR64rCL:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.SAR32ri]:
            base_opcode = 0xC1
            form = X64InstForm.MRM7r
            op_idx += 1  # TODO: Should defined as operand bias

            # flags
            imm_size = 1
        elif inst.opcode in [X64MachineOps.SAR32rCL, X64MachineOps.SAR64rCL]:
            base_opcode = 0xD3
            form = X64InstForm.MRM7r
            op_idx += 1  # TODO: Should defined as operand bias

            if inst.opcode == X64MachineOps.SAR64rCL:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.LEA32r, X64MachineOps.LEA64r]:
            base_opcode = 0x8D
            form = X64InstForm.MRMSrcMem

            if inst.opcode == X64MachineOps.LEA64r:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.CWD]:
            base_opcode = 0x99
            form = X64InstForm.RawFrm  # TODO: SizeOp
        elif inst.opcode in [X64MachineOps.CDQ, X64MachineOps.CQO]:
            base_opcode = 0x99
            form = X64InstForm.RawFrm  # TODO: SizeOp

            if inst.opcode == X64MachineOps.CQO:
                tsflags |= X64TSFlags.REX_W
        elif inst.opcode in [X64MachineOps.SETCCr]:
            base_opcode = 0x90
            form = X64InstForm.MRMXrCC

            tsflags |= X64TSFlags.TB
        elif inst.opcode in [X64MachineOps.JCC_1]:
            base_opcode = 0x70
            form = X64InstForm.AddCCFrm

            # flags
            imm_size = 1
            is_pcrel = True
        elif inst.opcode in [X64MachineOps.JCC_4]:
            base_opcode = 0x80
            form = X64InstForm.AddCCFrm

            # flags
            imm_size = 4
            is_pcrel = True
            tsflags |= X64TSFlags.TB
        elif inst.opcode in [X64MachineOps.JMP_1]:
            base_opcode = 0xEB
            form = X64InstForm.RawFrm

            # flags
            imm_size = 1
            is_pcrel = True
        elif inst.opcode in [X64MachineOps.JMP_4]:
            base_opcode = 0xE9
            form = X64InstForm.RawFrm

            # flags
            imm_size = 4
            is_pcrel = True
        elif inst.opcode in [X64MachineOps.RET]:
            base_opcode = 0xC3
            form = X64InstForm.RawFrm
        elif inst.opcode in [X64MachineOps.CALLpcrel32]:
            base_opcode = 0xE8
            form = X64InstForm.RawFrm

            # flags
            imm_size = 4
            is_pcrel = True
        elif inst.opcode in [X64MachineOps.DATA16_PREFIX]:
            base_opcode = 0x66
            form = X64InstForm.RawFrm
        elif inst.opcode in [X64MachineOps.REX64_PREFIX]:
            base_opcode = 0x48
            form = X64InstForm.RawFrm
        else:
            raise NotImplementedError()

        mem_operand_idx = get_mem_operand_pos(tsflags, form)
        if mem_operand_idx != -1:
            self.emit_segment_prefix(
                inst.operands[op_idx + mem_operand_idx + 4], output)
        rex = self.compute_rex_prefix(
            inst, op_idx, form, tsflags, mem_operand_idx)

        self.emit_opcode_prefix(tsflags, rex, inst, output)

        if form == X64InstForm.RawFrm:
            self.emit_byte(base_opcode, output)
        elif form == X64InstForm.AddRegFrm:
            self.emit_byte(
                base_opcode + self.get_register_number(inst.operands[op_idx]), output)
            op_idx += 1
        elif form == X64InstForm.AddCCFrm:
            self.emit_byte(
                base_opcode + inst.operands[num_operands - 1].imm, output)
            num_operands -= 1
        elif form in [X64InstForm.MRM0m, X64InstForm.MRM1m, X64InstForm.MRM2m, X64InstForm.MRM3m,
                      X64InstForm.MRM4m, X64InstForm.MRM5m, X64InstForm.MRM6m, X64InstForm.MRM7m]:
            self.emit_byte(base_opcode, output)
            self.emit_mem_mod_rm_byte(
                inst, op_idx, form.value - X64InstForm.MRM0m.value, fixups, output)
            op_idx += 5
        elif form == X64InstForm.MRMSrcReg:
            self.emit_byte(base_opcode, output)
            self.emit_reg_mod_rm_byte(
                inst.operands[op_idx + 1], self.get_register_number(inst.operands[op_idx]), output)
            op_idx += 2
        elif form == X64InstForm.MRMDestReg:
            self.emit_byte(base_opcode, output)
            self.emit_reg_mod_rm_byte(inst.operands[op_idx], self.get_register_number(
                inst.operands[op_idx + 1]), output)
            op_idx += 2
        elif form in [X64InstForm.MRM0r, X64InstForm.MRM1r, X64InstForm.MRM2r, X64InstForm.MRM3r,
                      X64InstForm.MRM4r, X64InstForm.MRM5r, X64InstForm.MRM6r, X64InstForm.MRM7r]:
            self.emit_byte(base_opcode, output)
            self.emit_reg_mod_rm_byte(
                inst.operands[op_idx], form.value - X64InstForm.MRM0r.value, output)
            op_idx += 1
        elif form == X64InstForm.MRMSrcMem:
            self.emit_byte(base_opcode, output)
            self.emit_mem_mod_rm_byte(
                inst, op_idx + 1, self.get_register_number(inst.operands[op_idx]), fixups, output)
            op_idx += 6
        elif form == X64InstForm.MRMDestMem:
            self.emit_byte(base_opcode, output)
            self.emit_mem_mod_rm_byte(inst, op_idx, self.get_register_number(
                inst.operands[op_idx + 5]), fixups, output)
            op_idx += 6
        elif form == X64InstForm.MRMXrCC:
            reg_op_idx = op_idx
            op_idx += 1
            cc = inst.operands[op_idx].imm
            self.emit_byte(base_opcode + cc, output)
            self.emit_reg_mod_rm_byte(inst.operands[reg_op_idx], 0, output)
            op_idx += 1
        else:
            raise NotImplementedError()

        while op_idx < num_operands:
            fixup_kind = get_fixup_kind_by_size(imm_size, is_pcrel)
            self.emit_immediate(
                inst.operands[op_idx], imm_size, fixup_kind, fixups, output)
            op_idx += 1
