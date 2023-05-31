from rachetwrench.codegen.mc import *

class MCAsmStream(MCStream):
    def __init__(self, context, output, printer, target_stream=None):
        super().__init__(context, target_stream)
        self.output = output
        self.printer = printer
        self.asm_info = context.asm_info
        self.current_section = None

    def emit_newline(self):
        self.output.write("\n")

    def emit_instruction(self, inst: MCInst):
        self.output.write("\t")

        if self.target_stream is not None:
            self.target_stream.print_inst(inst, self.output, self.printer)
        else:
            self.printer.print_inst(inst, self.output)

        self.emit_newline()

    def emit_label(self, symbol: MCSymbol):
        self.output.write(symbol.name)
        self.output.write(":")
        self.emit_newline()

    def emit_int_value(self, value: int, size: int):
        self.emit_value(MCConstantExpr(value), size)

    def emit_value(self, value: MCExpr, size: int):
        if size == 1:
            directive = self.asm_info.get_8bit_data_directive()
        elif size == 2:
            directive = self.asm_info.get_16bit_data_directive()
        elif size == 4:
            directive = self.asm_info.get_32bit_data_directive()
        elif size == 8:
            directive = self.asm_info.get_64bit_data_directive()
        else:
            raise NotImplementedError()

        self.output.write(directive)

        self.output.write("\t")

        if self.target_stream is not None:
            self.target_stream.emit_value(value, self.output)
        else:
            value.print(self.output, self.asm_info)

        self.emit_newline()

    def emit_symbol_attrib(self, symbol, attrib):
        if attrib == MCSymbolAttribute.Global:
            self.output.write(".globl")
            self.output.write("\t")
        else:
            return
            raise ValueError("The attribute is not supported.")

        self.output.write(symbol.name)
        self.emit_newline()

    def switch_section(self, section):
        if self.current_section != section:
            SECTION_NAME = {
                self.context.obj_file_info.bss_section: ".bss",
                self.context.obj_file_info.text_section: ".text",
                self.context.obj_file_info.rodata_section: ".rodata",
            }
            self.output.write(SECTION_NAME[section])
            self.output.write("\n")

            self.current_section = section

    def emit_elf_size(self, symbol, size):
        pass

    def emit_zeros(self, size):
        pass

    def emit_bytes(self, data):
        pass

    def emit_coff_symbol_storage_class(self, symbol, cls):
        pass

    def emit_coff_symbol_type(self, symbol, ty):
        pass

    def emit_syntax_directive(self):
        self.output.write(".intel_syntax noprefix")
        self.emit_newline()
