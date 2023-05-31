from rachetwrench.codegen.mc import MCContext, MCStream, MCCodeEmitter, MCSymbol, MCFragment
from rachetwrench.codegen.assembler import MCAssembler


class MCObjectStream(MCStream):
    def __init__(self, context: MCContext, backend, writer, emitter: MCCodeEmitter):
        super().__init__(context, None)
        self.assembler = MCAssembler(context, backend, emitter, writer)

    def finalize(self):
        self.assembler.finalize()


class MCObjectTargetWriter:
    def __init__(self):
        pass


class MCObjectWriter:
    def __init__(self):
        pass

    def compute_after_layout(self, asm, obj):
        raise NotImplementedError()

    def write_object(self, asm, obj):
        raise NotImplementedError()

    def can_fully_resolve_symbol_rel_diff(self, a, b):
        if isinstance(b, MCFragment):
            sec_b = b.section
        else:
            assert(isinstance(b, MCSymbol))
            sec_b = b.section
        return a.section == sec_b


class MCObjectFileInfo:
    def __init__(self):
        pass

    @property
    def text_section(self):
        raise NotImplementedError()

    @property
    def bss_section(self):
        raise NotImplementedError()

    @property
    def data_section(self):
        raise NotImplementedError()

    @property
    def rodata_section(self):
        raise NotImplementedError()

    def get_symbol_class(self):
        raise NotImplementedError()
