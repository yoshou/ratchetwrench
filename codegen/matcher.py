from codegen.dag import Dag


class PatternMatcher:
    def __init__(self, name):
        self.name = name

    def match(self, node, values, idx, dag):
        return None


class MatcherResult:
    def __init__(self, name, value, sub_values=None):
        if not sub_values:
            sub_values = []

        self.name = name
        self.value = value

        assert(not isinstance(value, MatcherResult))
        for sub_value in sub_values:
            assert(isinstance(sub_value, MatcherResult))

        assert(isinstance(sub_values, list))

        self.sub_values = sub_values

    @property
    def values(self):
        vals = [val.value for name, val in self.sub_values]
        if self.value:
            vals.append(self.value)
        return vals

    @property
    def values_as_dict(self):
        dic = {val.name: val for val in self.sub_values if val.name}

        for sub_value in self.sub_values:
            dic.update(sub_value.values_as_dict)
        if self.name:
            dic[self.name] = self
        return dic

    @property
    def values_as_list(self):
        vals = [(name, val) for name, val in self.sub_values]
        if self.value:
            vals.append((self.name, self))
        return vals


def construct(inst, node, dag: Dag, result: MatcherResult):
    from codegen.types import ValueType, MachineValueType
    from codegen.spec import MachineRegisterDef
    from codegen.dag import DagValue

    dic = result.values_as_dict

    ops = []
    for name, opnd in inst.ins.items():
        ops.extend(opnd.apply(dic[name].value, dag))

    operands = list(node.operands)

    # Capture chain
    chain = None

    operand_idx = 0
    if operand_idx < len(operands) and operands[operand_idx].ty.value_type == ValueType.OTHER:
        chain = operands[operand_idx]
        operand_idx += 1

    stack = []
    if chain is None:
        stack.append(node)

    while len(stack) > 0:
        parent_node = stack.pop()

        if len(parent_node.operands) == 0:
            break

        if parent_node.operands[0].ty.value_type == ValueType.OTHER:
            chain = parent_node.operands[0]
            break

        for operand in parent_node.operands:
            stack.append(operand.node)

    if not chain:
        chain = dag.entry

    glue = None

    if len(operands) > 0 and operands[-1].ty.value_type == ValueType.GLUE:
        glue = node.operands[-1]
        operands.pop()

    for reg in inst.uses:
        assert(isinstance(reg, MachineRegisterDef))
        if not reg in dic:
            continue

        operand = dic[reg].value

        if not operand:
            continue

        reg_node = DagValue(dag.add_target_register_node(
            operand.ty, reg), 0)

        copy_to_reg_ops = [chain, reg_node, operand]
        if glue:
            copy_to_reg_ops.append(glue)

        chain = DagValue(dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)],
                                      *copy_to_reg_ops), 0)

        glue = chain.get_value(1)

    operand_idx += len(ops)

    while operand_idx < len(operands):
        operand = operands[operand_idx]
        operand_idx += 1

        if operand == glue:
            continue

        ops.append(operand)

    if chain:
        ops.append(chain)

    if glue:
        ops.append(glue)

    return dag.add_machine_dag_node(inst, node.value_types, *ops)


class NodePatternMatcher(PatternMatcher):
    def __init__(self, opcode_matcher, operand_matchers, value_matchers, props, predicates=None, name=None):
        super().__init__(name)

        self.opcode = opcode_matcher
        self.operands = operand_matchers
        self.values = value_matchers
        self.props = dict(props)

        self.predicates = predicates
        if not self.predicates:
            self.predicates = []

    def match(self, node, values, idx, dag):
        from codegen.types import ValueType, MachineValueType
        from codegen.spec import MachineRegisterDef
        from codegen.mir import MachineRegister
        from codegen.dag import VirtualDagOps, DagValue, StoreDagNode, LoadDagNode

        if idx >= len(values):
            return idx, None

        value = values[idx]

        node = value.node

        # Check opcode
        _, res = self.opcode.match(None, [value], 0, dag)
        if not res:
            return idx, None

        mem_vt = self.props["mem_vt"] if "mem_vt" in self.props else None
        if mem_vt:
            assert(isinstance(node, (StoreDagNode, LoadDagNode)))

            if node.mem_operand.size != MachineValueType(mem_vt).get_size_in_byte():
                return idx, None

        for p in self.predicates:
            if not p(node):
                return idx, None

        # Check operands
        operand_idx = 0

        # Capture chain
        chain = None
        if operand_idx < len(node.operands) and node.operands[operand_idx].ty.value_type == ValueType.OTHER:
            chain = node.operands[operand_idx]
            operand_idx += 1

        match_results = []
        for matcher in self.operands:
            operand_idx, res = matcher.match(
                node, node.operands, operand_idx, dag)

            if not res:
                return idx, None

            match_results.append(res)

            for sub_value in res.sub_values:
                match_results.append(sub_value)

        # while operand_idx < len(node.operands) and node.operands[operand_idx].node.opcode == VirtualDagOps.UNDEF:
        #     operand_idx += 1

        # if operand_idx != len(node.operands):
        #     return None

        # Check values
        value_idx = 0
        node_values = [DagValue(node, i) for i in range(len(node.value_types))]

        for matcher in self.values:
            _, res = matcher.match(node, node_values, value_idx, dag)
            if not res:
                return idx, None

            match_results.append(res)

            for sub_value in res.sub_values:
                match_results.append(sub_value)

        return idx + 1, MatcherResult(None, None, match_results)


class NodeOpcodePatternMatcher(PatternMatcher):
    def __init__(self, opcode, name=None):
        super().__init__(name)

        self.opcode = opcode

    def match_opcode(self, node):
        if not self.opcode:
            return True

        return node.opcode == self.opcode

    def match(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]

        if not self.opcode:
            return idx + 1, MatcherResult(self.name, value)

        if not self.match_opcode(value.node):
            return idx, None

        return idx + 1, MatcherResult(self.name, value)


from copy import copy


def create_matcher(matcher_or_tuple):
    from codegen.spec import MachineRegisterClassDef, MachineRegisterDef

    if isinstance(matcher_or_tuple, tuple):
        name, operand = matcher_or_tuple
        if isinstance(operand, PatternMatcher):
            operand = copy(operand)
            operand.name = name
            return operand
        if isinstance(operand, MachineRegisterClassDef):
            return RegClassMatcher(operand, name)
        else:
            raise ValueError()
    elif isinstance(matcher_or_tuple, MachineRegisterDef):
        return RegValueMatcher(matcher_or_tuple, matcher_or_tuple)
    elif isinstance(matcher_or_tuple, int):
        return ConstantValueMatcher(matcher_or_tuple, matcher_or_tuple)
    elif isinstance(matcher_or_tuple, ValueTypeMatcherGen):
        return ValueTypeValueMatcher(matcher_or_tuple.value_type)
    else:
        assert(isinstance(matcher_or_tuple, PatternMatcher))
        return matcher_or_tuple


class NodePatternMatcherGen:
    def __init__(self, opcode: str, *props, mem_vt=None, predicates=None):
        self.opcode = opcode
        self.props = dict({"mem_vt": mem_vt})
        self.predicates = predicates

    def __call__(self, *operands):
        from codegen.spec import MachineRegisterDef

        opcode_matcher = NodeOpcodePatternMatcher(self.opcode)
        operand_matchers = []

        for matcher_or_tuple in operands:
            operand_matchers.append(create_matcher(matcher_or_tuple))

        return NodePatternMatcher(opcode_matcher, operand_matchers, [], self.props, predicates=self.predicates)


from enum import Enum, auto


class NodeProperty(Enum):
    Commutative = auto()
    Associative = auto()
    HasChain = auto()
    OutGlue = auto()
    InGlue = auto()
    MemOperand = auto()


from codegen.dag import VirtualDagOps
from codegen.types import ValueType


add_ = NodePatternMatcherGen(
    VirtualDagOps.ADD, NodeProperty.Commutative, NodeProperty.Associative)

sub_ = NodePatternMatcherGen(
    VirtualDagOps.SUB)

mul_ = NodePatternMatcherGen(
    VirtualDagOps.MUL, NodeProperty.Commutative, NodeProperty.Associative)

sdiv_ = NodePatternMatcherGen(VirtualDagOps.SDIV)
udiv_ = NodePatternMatcherGen(VirtualDagOps.UDIV)
srem_ = NodePatternMatcherGen(VirtualDagOps.SREM)
urem_ = NodePatternMatcherGen(VirtualDagOps.UREM)

and_ = NodePatternMatcherGen(
    VirtualDagOps.AND, NodeProperty.Commutative, NodeProperty.Associative)

or_ = NodePatternMatcherGen(
    VirtualDagOps.OR, NodeProperty.Commutative, NodeProperty.Associative)

xor_ = NodePatternMatcherGen(
    VirtualDagOps.XOR, NodeProperty.Commutative, NodeProperty.Associative)

sra_ = NodePatternMatcherGen(VirtualDagOps.SRA)
srl_ = NodePatternMatcherGen(VirtualDagOps.SRL)
shl_ = NodePatternMatcherGen(VirtualDagOps.SHL)

fadd_ = NodePatternMatcherGen(
    VirtualDagOps.FADD, NodeProperty.Commutative, NodeProperty.Associative)

fsub_ = NodePatternMatcherGen(VirtualDagOps.FSUB)

fmul_ = NodePatternMatcherGen(
    VirtualDagOps.FMUL, NodeProperty.Commutative, NodeProperty.Associative)

fdiv_ = NodePatternMatcherGen(VirtualDagOps.FDIV)

from codegen.dag import LoadExtType

load_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, predicates=[
        lambda node: node.ext_type == LoadExtType.NON
    ])

extloadi1_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I1, predicates=[
        lambda node: node.ext_type == LoadExtType.EXTLOAD
    ])

extloadi8_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I8, predicates=[
        lambda node: node.ext_type == LoadExtType.EXTLOAD
    ])

extloadi16_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I16, predicates=[
        lambda node: node.ext_type == LoadExtType.EXTLOAD
    ])

extloadi32_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I32, predicates=[
        lambda node: node.ext_type == LoadExtType.EXTLOAD
    ])

zextloadi1_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I1, predicates=[
        lambda node: node.ext_type == LoadExtType.ZEXTLOAD
    ])

zextloadi8_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I8, predicates=[
        lambda node: node.ext_type == LoadExtType.ZEXTLOAD
    ])

zextloadi16_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I16, predicates=[
        lambda node: node.ext_type == LoadExtType.ZEXTLOAD
    ])

zextloadi32_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I32, predicates=[
        lambda node: node.ext_type == LoadExtType.ZEXTLOAD
    ])

sextloadi1_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I1, predicates=[
        lambda node: node.ext_type == LoadExtType.SEXTLOAD
    ])

sextloadi8_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I8, predicates=[
        lambda node: node.ext_type == LoadExtType.SEXTLOAD
    ])

sextloadi16_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I16, predicates=[
        lambda node: node.ext_type == LoadExtType.SEXTLOAD
    ])

sextloadi32_ = NodePatternMatcherGen(
    VirtualDagOps.LOAD, NodeProperty.HasChain, mem_vt=ValueType.I32, predicates=[
        lambda node: node.ext_type == LoadExtType.SEXTLOAD
    ])

store_ = NodePatternMatcherGen(
    VirtualDagOps.STORE, NodeProperty.HasChain)

truncstorei1_ = NodePatternMatcherGen(
    VirtualDagOps.STORE, NodeProperty.HasChain, mem_vt=ValueType.I1)

truncstorei8_ = NodePatternMatcherGen(
    VirtualDagOps.STORE, NodeProperty.HasChain, mem_vt=ValueType.I8)

truncstorei16_ = NodePatternMatcherGen(
    VirtualDagOps.STORE, NodeProperty.HasChain, mem_vt=ValueType.I16)

truncstorei32_ = NodePatternMatcherGen(
    VirtualDagOps.STORE, NodeProperty.HasChain, mem_vt=ValueType.I32)

br_ = NodePatternMatcherGen(VirtualDagOps.BR)

brcond_ = NodePatternMatcherGen(VirtualDagOps.BRCOND)

setcc_ = NodePatternMatcherGen(VirtualDagOps.SETCC)
bitconvert_ = NodePatternMatcherGen(VirtualDagOps.BITCAST)
scalar_to_vector_ = NodePatternMatcherGen(VirtualDagOps.SCALAR_TO_VECTOR)
build_vector_ = NodePatternMatcherGen(VirtualDagOps.BUILD_VECTOR)
vector_insert_ = NodePatternMatcherGen(VirtualDagOps.INSERT_VECTOR_ELT)
vector_extract_ = NodePatternMatcherGen(VirtualDagOps.EXTRACT_VECTOR_ELT)

fp_to_sint_ = NodePatternMatcherGen(VirtualDagOps.FP_TO_SINT)
fp_to_uint_ = NodePatternMatcherGen(VirtualDagOps.FP_TO_UINT)
uint_to_fp_ = NodePatternMatcherGen(VirtualDagOps.UINT_TO_FP)
sint_to_fp_ = NodePatternMatcherGen(VirtualDagOps.SINT_TO_FP)

fp_extend_ = NodePatternMatcherGen(VirtualDagOps.FP_EXTEND)
fp_round_ = NodePatternMatcherGen(VirtualDagOps.FP_ROUND)

sext_ = NodePatternMatcherGen(VirtualDagOps.SIGN_EXTEND)
zext_ = NodePatternMatcherGen(VirtualDagOps.ZERO_EXTEND)
anyext_ = NodePatternMatcherGen(VirtualDagOps.ANY_EXTEND)
trunc_ = NodePatternMatcherGen(VirtualDagOps.TRUNCATE)
sext_inreg_ = NodePatternMatcherGen(VirtualDagOps.SIGN_EXTEND_INREG)


class ComplexPatternMatcher(PatternMatcher):
    def __init__(self, value_type, num_operands, fn, roots, name=None):
        super().__init__(name)

        self.value_type = value_type
        self.num_operands = num_operands
        self.fn = fn
        self.roots = roots

    def __call__(self, *operand_matchers):
        assert(len(operand_matchers) == self.num_operands)

        class ComplexPatternMatcherDeconstruction(PatternMatcher):
            def __init__(self, value_type, num_operands, fn, roots, operand_matchers, name=None):
                super().__init__(name)

                self.value_type = value_type
                self.num_operands = num_operands
                self.fn = fn
                self.roots = roots
                self.operand_matchers = [create_matcher(
                    matcher_or_tuple) for matcher_or_tuple in operand_matchers]

                self.matcher = ComplexOperandMatcher(self.fn, name)

            def match(self, node, values, idx, dag):
                for root_matcher in self.roots:
                    _, res = root_matcher.match(node)
                    if not res:
                        return idx, None

                res_idx, res = self.matcher.match(node, values, idx, dag)
                if not res:
                    return idx, None

                results = []
                for operand, res_value in zip(self.operand_matchers, res.values[0]):
                    results.append(MatcherResult(operand.name, res_value))

                return res_idx, MatcherResult(self.name, [], results)

        return ComplexPatternMatcherDeconstruction(self.value_type, self.num_operands, self.fn, self.roots, list(operand_matchers), self.name)


class NodeValuePatternMatcher(PatternMatcher):
    def __init__(self, name):
        super().__init__(name)

    def match(self, node, values, idx, dag):
        raise NotImplementedError()


bb = NodeOpcodePatternMatcher(VirtualDagOps.BASIC_BLOCK)
imm = NodeOpcodePatternMatcher(VirtualDagOps.CONSTANT)
timm = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_CONSTANT)
fpimm = NodeOpcodePatternMatcher(VirtualDagOps.CONSTANT_FP)
tfpimm = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_CONSTANT_FP)
globaladdr_ = NodeOpcodePatternMatcher(VirtualDagOps.GLOBAL_ADDRESS)
tglobaladdr_ = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_GLOBAL_ADDRESS)
constpool_ = NodeOpcodePatternMatcher(VirtualDagOps.CONSTANT_POOL)
tconstpool_ = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_CONSTANT_POOL)
frameindex_ = NodeOpcodePatternMatcher(VirtualDagOps.FRAME_INDEX)
tframeindex_ = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_FRAME_INDEX)
externalsym_ = NodeOpcodePatternMatcher(VirtualDagOps.EXTERNAL_SYMBOL)
texternalsym_ = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_EXTERNAL_SYMBOL)
tglobaltlsaddr_ = NodeOpcodePatternMatcher(
    VirtualDagOps.TARGET_GLOBAL_TLS_ADDRESS)


class ValueTypeMatcher(NodeValuePatternMatcher):
    def __init__(self, value_type, sub_matcher, operands, name=None):
        super().__init__(name)

        from codegen.spec import get_builder

        self.value_type = value_type
        self.sub_matcher = sub_matcher
        self.operands = [get_builder(operand) for operand in operands]

    def match(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]
        if value.ty != self.value_type:
            return idx, None

        sub_result_idx, sub_result = self.sub_matcher.match(
            node, values, 0, dag)

        if idx == sub_result_idx:
            return idx, None

        return idx + 1, MatcherResult(self.name, value, [sub_result] if sub_result else [])

    def construct(self, node, dag: Dag, result: MatcherResult):
        operands = []
        for operand in self.operands:
            value = operand.construct(node, dag, result)
            operands.extend(value)

        assert(len(operands) == 1)

        base_node = operands[0].node

        from codegen.dag import DagNode, DagValue

        if isinstance(base_node, DagNode):
            return [DagValue(base_node, 0)]
            # return [DagValue(dag.add_node(base_node.opcode, [self.value_type], *base_node.operands), 0)]
        else:
            raise NotImplementedError()


class ValueTypeMatcherGen:
    def __init__(self, value_type, **props):
        from codegen.types import MachineValueType

        self.value_type = MachineValueType(value_type)
        self.props = dict(props)

        self.matcher = ValueTypeMatcher(self.value_type, None, [])

    def match(self, node, values, idx, dag):
        from codegen.dag import DagValue

        return self.matcher.match(None, [DagValue(node, 0)], 0, dag)

    def __call__(self, *operands):
        opcode_matcher = NodeOpcodePatternMatcher(None)
        value_matchers = []
        operand_matchers = []

        for matcher_or_tuple in operands:
            value_matchers.append(create_matcher(matcher_or_tuple))

        # value_matchers.append(ValueTypeMatcher(self.value_type))

        sub_matcher = NodePatternMatcher(
            opcode_matcher, operand_matchers, value_matchers, self.props)

        return ValueTypeMatcher(self.value_type, sub_matcher, list(operands))


class ValueTypeValueMatcher(NodeValuePatternMatcher):
    def __init__(self, value_type, name=None):
        super().__init__(name)

        self.value_type = value_type

    def match(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]

        if value.node.opcode != VirtualDagOps.VALUETYPE:
            return idx, None

        if value.node.vt != self.value_type:
            return idx, None

        return idx + 1, MatcherResult(self.name, value)


from codegen.types import ValueType

i1_ = ValueTypeMatcherGen(ValueType.I1)
i8_ = ValueTypeMatcherGen(ValueType.I8)
i16_ = ValueTypeMatcherGen(ValueType.I16)
i32_ = ValueTypeMatcherGen(ValueType.I32)
i64_ = ValueTypeMatcherGen(ValueType.I64)

f32_ = ValueTypeMatcherGen(ValueType.F32)
f64_ = ValueTypeMatcherGen(ValueType.F64)
f128_ = ValueTypeMatcherGen(ValueType.F128)

v4f32_ = ValueTypeMatcherGen(ValueType.V4F32)


class CondCodePatternMatcher(PatternMatcher):
    def __init__(self, condcode, name=None):
        super().__init__(name)

        self.condcode = condcode

    def match(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]

        if value.node.opcode != VirtualDagOps.CONDCODE:
            return idx, None

        if value.node.cond != self.condcode:
            return idx, None

        return idx + 1, MatcherResult(self.name, value)


from codegen.dag import CondCode

SETOEQ = CondCodePatternMatcher(CondCode.SETOEQ)
SETOGT = CondCodePatternMatcher(CondCode.SETOGT)
SETOGE = CondCodePatternMatcher(CondCode.SETOGE)
SETOLT = CondCodePatternMatcher(CondCode.SETOLT)
SETOLE = CondCodePatternMatcher(CondCode.SETOLE)
SETONE = CondCodePatternMatcher(CondCode.SETONE)
SETO = CondCodePatternMatcher(CondCode.SETO)
SETUO = CondCodePatternMatcher(CondCode.SETUO)
SETUEQ = CondCodePatternMatcher(CondCode.SETUEQ)
SETUGT = CondCodePatternMatcher(CondCode.SETUGT)
SETUGE = CondCodePatternMatcher(CondCode.SETUGE)
SETULT = CondCodePatternMatcher(CondCode.SETULT)
SETULE = CondCodePatternMatcher(CondCode.SETULE)
SETUNE = CondCodePatternMatcher(CondCode.SETUNE)
SETEQ = CondCodePatternMatcher(CondCode.SETEQ)
SETGT = CondCodePatternMatcher(CondCode.SETGT)
SETGE = CondCodePatternMatcher(CondCode.SETGE)
SETLT = CondCodePatternMatcher(CondCode.SETLT)
SETLE = CondCodePatternMatcher(CondCode.SETLE)
SETNE = CondCodePatternMatcher(CondCode.SETNE)


class ComplexOperandMatcher(NodeValuePatternMatcher):
    def __init__(self, func, name=None):
        super().__init__(name)

        self.func = func

    def match(self, node, values, idx, dag):
        next_idx, res = self.func(node, values, idx, dag)

        if res:
            return next_idx, MatcherResult(self.name, res)

        return idx, None


class AllZeroOperandMatcher(NodeValuePatternMatcher):
    def __init__(self, name=None):
        super().__init__(name)

    def match(self, node, operands, idx, dag):
        from codegen.dag import VirtualDagOps

        for operand in operands[idx:]:
            if operand.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.CONSTANT_FP, VirtualDagOps.TARGET_CONSTANT, VirtualDagOps.TARGET_CONSTANT_FP]:
                return idx, None

            if operand.node.value.value != 0:
                return idx, None

        return len(operands), MatcherResult(self.name, operands[idx:])


is_all_zero = AllZeroOperandMatcher()


imm_zero_vec = build_vector_(("src", is_all_zero))


class RegClassMatcher(NodeValuePatternMatcher):
    def __init__(self, regclass, name=None):
        super().__init__(name)

        self.regclass = regclass

    def match(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        tys = self.regclass.get_types(dag.mfunc.target_info.hwmode)

        value = values[idx]

        if value.ty in tys:
            return idx + 1, MatcherResult(self.name, value)

        return idx, None


class RegValueMatcher(NodeValuePatternMatcher):
    def __init__(self, reg, name=None):
        super().__init__(name)

        self.reg = reg

    def match(self, node, values, idx, dag):
        value = None
        if idx < len(values):
            value = values[idx]
            idx += 1

        return idx, MatcherResult(self.name, value)

        # value = None
        # if idx < len(values) and values[idx].node.opcode == VirtualDagOps.REGISTER:
        #     value = values[idx]

        # if value and isinstance(value.node.reg, MachineRegister):
        #     if value.node.reg.spec != self.reg:
        #         return idx, None

        #     idx += 1


class ConstantValueMatcher(NodeValuePatternMatcher):
    def __init__(self, value, name=None):
        super().__init__(name)

        self.value = value

    def match(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]

        if value.node.opcode != VirtualDagOps.CONSTANT:
            return idx, None

        if value.node.value.value != self.value:
            return idx, None

        return idx + 1, MatcherResult(self.name, value)


class SetPatternMatcherGen(NodePatternMatcherGen):
    def __init__(self, **props):
        self.props = dict(props)

    def __call__(self, *operands):
        from codegen.spec import MachineRegisterClassDef, MachineRegisterDef

        opcode_matcher = NodeOpcodePatternMatcher(None)
        value_matchers = []

        for matcher_or_tuple in operands[:-1]:
            value_matchers.append(create_matcher(matcher_or_tuple))

        value_matchers.append(create_matcher(operands[-1]))

        return NodePatternMatcher(opcode_matcher, [], value_matchers, self.props)


set_ = SetPatternMatcherGen()


class SimplePattern(PatternMatcher):
    def __init__(self, pattern, result, enabled):
        self.pattern = pattern
        self.result = result
        self.enabled = enabled

    def match(self, node, dag):
        from codegen.types import ValueType
        from codegen.spec import MachineRegisterDef
        from codegen.mir import MachineRegister
        from codegen.dag import VirtualDagOps, DagValue

        value = DagValue(node, 0)

        return self.pattern.match(None, [value], 0, dag)

    def construct(self, node, dag: Dag, result: MatcherResult):
        return self.result.construct(node, dag, result)[0]


def def_pat(pattern, result, patterns, enabled=None):
    from codegen.spec import MachineRegisterDef

    opcode_matcher = NodeOpcodePatternMatcher(None)
    operand_matchers = []
    value_matchers = []

    value_matchers.append(create_matcher(pattern))

    matcher = NodePatternMatcher(
        opcode_matcher, operand_matchers, value_matchers, {})

    patterns.append(SimplePattern(matcher, result, enabled))


def setoeq_(lhs, rhs): return setcc_(lhs, rhs, SETOEQ)


def setoeq_(lhs, rhs): return setcc_(lhs, rhs, SETOEQ)


def setogt_(lhs, rhs): return setcc_(lhs, rhs, SETOGT)


def setoge_(lhs, rhs): return setcc_(lhs, rhs, SETOGE)


def setolt_(lhs, rhs): return setcc_(lhs, rhs, SETOLT)


def setole_(lhs, rhs): return setcc_(lhs, rhs, SETOLE)


def setone_(lhs, rhs): return setcc_(lhs, rhs, SETONE)


def seteq_(lhs, rhs): return setcc_(lhs, rhs, SETEQ)


def setgt_(lhs, rhs): return setcc_(lhs, rhs, SETGT)


def setge_(lhs, rhs): return setcc_(lhs, rhs, SETGE)


def setlt_(lhs, rhs): return setcc_(lhs, rhs, SETLT)


def setle_(lhs, rhs): return setcc_(lhs, rhs, SETLE)


def setne_(lhs, rhs): return setcc_(lhs, rhs, SETNE)


def setueq_(lhs, rhs): return setcc_(lhs, rhs, SETUEQ)


def setugt_(lhs, rhs): return setcc_(lhs, rhs, SETUGT)


def setuge_(lhs, rhs): return setcc_(lhs, rhs, SETUGE)


def setult_(lhs, rhs): return setcc_(lhs, rhs, SETULT)


def setule_(lhs, rhs): return setcc_(lhs, rhs, SETULE)


def setune_(lhs, rhs): return setcc_(lhs, rhs, SETUNE)
