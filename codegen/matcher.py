from codegen.dag import Dag


class Pattern:
    def __init__(self, pattern, generator):
        self.pattern = pattern
        self.generator = generator

    def match(self, node):
        return None


class MatcherResult:
    def __init__(self, operands):
        self.operands = operands


class NodePatternMatcher:
    def __init__(self, opcode_matcher, operand_matchers, value_matchers, props):
        self.opcode = opcode_matcher
        self.operands = operand_matchers
        self.values = value_matchers
        self.props = dict(props)

    def construct(self, inst, node, dag: Dag, match_results, chain):
        from codegen.types import ValueType, MachineValueType
        from codegen.spec import MachineRegisterDef
        from codegen.dag import DagValue

        ops = []
        for name, _ in inst.value.ins.items():
            for operand in match_results[name].operands:
                ops.append(operand)

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
        for reg in inst.value.uses:
            assert(isinstance(reg, MachineRegisterDef))
            operand = match_results[reg].operands[0]

            reg_node = DagValue(dag.add_target_register_node(
                operand.ty, reg), 0)

            chain = DagValue(dag.add_node(VirtualDagOps.COPY_TO_REG, [MachineValueType(ValueType.OTHER), MachineValueType(ValueType.GLUE)],
                                          chain, reg_node, operand), 0)

            glue = chain.get_value(1)

        if len(node.operands) > 0 and node.operands[-1].ty.value_type == ValueType.GLUE:
            glue = node.operands[-1]

        if chain:
            ops.append(chain)

        if glue:
            ops.append(glue)

        return dag.add_machine_dag_node(inst, node.value_types, *ops)

    def match(self, inst, node, dag):
        from codegen.types import ValueType
        from codegen.spec import MachineRegisterDef
        from codegen.mir import MachineRegister
        from codegen.dag import VirtualDagOps, DagValue

        # Check opcode
        if not self.opcode.match_opcode(node):
            return None

        # Check operands
        operand_idx = 0

        # Capture chain
        chain = None
        if operand_idx < len(node.operands) and node.operands[operand_idx].ty.value_type == ValueType.OTHER:
            chain = node.operands[operand_idx]
            operand_idx += 1

        match_results = {}
        for name, matcher in self.operands:
            operand_idx, res = matcher.match_value(
                node, node.operands, operand_idx, dag)
            if not res:
                return None
            match_results[name] = res

        # while operand_idx < len(node.operands) and node.operands[operand_idx].node.opcode == VirtualDagOps.UNDEF:
        #     operand_idx += 1

        # if operand_idx != len(node.operands):
        #     return None

        # Check values
        value_idx = 0
        node_values = [DagValue(node, i) for i in range(len(node.value_types))]

        for name, matcher in self.values:
            _, res = matcher.match_value(node, node_values, value_idx, dag)
            if not res:
                return None
            match_results[name] = res

        return self.construct(inst, node, dag, match_results, chain)


class NodeOpcodePatternMatcher:
    def __init__(self, opcode=None):
        self.opcode = opcode

    def match_opcode(self, node):
        if not self.opcode:
            return True

        return node.opcode == self.opcode

    def match_value(self, node, values, idx, dag):
        if not self.opcode:
            return True

        if idx >= len(values):
            return idx, None

        value = values[idx]

        if not self.match_opcode(value.node):
            return idx, None

        return idx + 1, MatcherResult([value])


def get_operand_matchers(name, operand):
    from codegen.spec import MachineRegisterClassDef

    if isinstance(operand, NodeValuePatternMatcher):
        return (name, operand)
    elif isinstance(operand, NodeOpcodePatternMatcher):
        return (name, operand)
    elif isinstance(operand, MachineRegisterClassDef):
        return (name, RegClassMatcher(operand))
    else:
        raise ValueError()


class NodePatternMatcherGen:
    def __init__(self, opcode: str, **props):
        self.opcode = opcode
        self.props = dict(props)

    def __call__(self, *operands):
        from codegen.spec import MachineRegisterDef

        opcode_matcher = NodeOpcodePatternMatcher(self.opcode)
        operand_matchers = []

        for matcher_or_tuple in operands:
            if isinstance(matcher_or_tuple, tuple):
                name, operand = matcher_or_tuple
                operand_matchers.append(get_operand_matchers(name, operand))
            elif isinstance(matcher_or_tuple, MachineRegisterDef):
                operand_matchers.append(
                    (matcher_or_tuple, RegValueMatcher(matcher_or_tuple)))
            else:
                raise ValueError()

        return NodePatternMatcher(opcode_matcher, operand_matchers, [], self.props)


from codegen.dag import VirtualDagOps


add_ = NodePatternMatcherGen(VirtualDagOps.ADD)
sub_ = NodePatternMatcherGen(VirtualDagOps.SUB)
mul_ = NodePatternMatcherGen(VirtualDagOps.MUL)
sdiv_ = NodePatternMatcherGen(VirtualDagOps.SDIV)

and_ = NodePatternMatcherGen(VirtualDagOps.AND)
or_ = NodePatternMatcherGen(VirtualDagOps.OR)
xor_ = NodePatternMatcherGen(VirtualDagOps.XOR)
sra_ = NodePatternMatcherGen(VirtualDagOps.SRA)
srl_ = NodePatternMatcherGen(VirtualDagOps.SRL)
shl_ = NodePatternMatcherGen(VirtualDagOps.SHL)

fadd_ = NodePatternMatcherGen(VirtualDagOps.FADD)
fsub_ = NodePatternMatcherGen(VirtualDagOps.FSUB)
fmul_ = NodePatternMatcherGen(VirtualDagOps.FMUL)
fdiv_ = NodePatternMatcherGen(VirtualDagOps.FDIV)

load_ = NodePatternMatcherGen(VirtualDagOps.LOAD)
store_ = NodePatternMatcherGen(VirtualDagOps.STORE)
frameindex_ = NodePatternMatcherGen(VirtualDagOps.FRAME_INDEX)
br_ = NodePatternMatcherGen(VirtualDagOps.BR)
brcond_ = NodePatternMatcherGen(VirtualDagOps.BRCOND)

setcc_ = NodePatternMatcherGen(VirtualDagOps.SETCC)
bitconvert_ = NodePatternMatcherGen(VirtualDagOps.BITCAST)
scalar_to_vector_ = NodePatternMatcherGen(VirtualDagOps.SCALAR_TO_VECTOR)


class NodeValuePatternMatcher:
    def match_value(self, node, values, idx, dag):
        raise NotImplementedError()


class ConstantIntOperandMatcher(NodeValuePatternMatcher):
    def __init__(self, value_type):
        self.value_type = value_type

    def match_value(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]
        if value.ty.value_type != self.value_type:
            return idx, None

        return idx + 1, MatcherResult([value])


bb = NodeOpcodePatternMatcher(VirtualDagOps.BASIC_BLOCK)
imm = NodeOpcodePatternMatcher(VirtualDagOps.CONSTANT)
timm = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_CONSTANT)
tglobaladdr_ = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_GLOBAL_ADDRESS)
tconstpool_ = NodeOpcodePatternMatcher(VirtualDagOps.TARGET_CONSTANT_POOL)


class ComplexOperandMatcher(NodeValuePatternMatcher):
    def __init__(self, func):
        self.func = func

    def match_value(self, node, values, idx, dag):
        return self.func(node, values, idx, dag)


def imm_zero_vec_ops(node, operands, idx, dag):
    from codegen.dag import VirtualDagOps

    for operand in operands[idx:]:
        if operand.node.opcode not in [VirtualDagOps.CONSTANT, VirtualDagOps.CONSTANT_FP, VirtualDagOps.TARGET_CONSTANT, VirtualDagOps.TARGET_CONSTANT_FP]:
            return idx, None

        if operand.node.value.value != 0:
            return idx, None

    return len(operands), MatcherResult(operands[idx:])


imm_zero_vec = NodePatternMatcherGen(
    VirtualDagOps.BUILD_VECTOR)(("src", ComplexOperandMatcher(imm_zero_vec_ops)))


class RegClassMatcher(NodeValuePatternMatcher):
    def __init__(self, regclass):
        self.regclass = regclass

    def match_value(self, node, values, idx, dag):
        if idx >= len(values):
            return idx, None

        value = values[idx]

        if value.ty in self.regclass.tys:
            return idx + 1, MatcherResult([value])

        return idx, None


class RegValueMatcher(NodeValuePatternMatcher):
    def __init__(self, reg):
        self.reg = reg

    def match_value(self, node, values, idx, dag):
        value = None
        if idx < len(values):
            value = values[idx]
            idx += 1

        return idx, MatcherResult([value])


class SetPatternMatcherGen(NodePatternMatcherGen):
    def __init__(self, **props):
        self.props = dict(props)

    def __call__(self, *operands):
        from codegen.spec import MachineRegisterClassDef, MachineRegisterDef

        opcode_matcher = NodeOpcodePatternMatcher(None)
        operand_matchers = []
        value_matchers = []

        for matcher_or_tuple in operands[:-1]:

            if isinstance(matcher_or_tuple, tuple):
                name, operand = matcher_or_tuple
                if isinstance(operand, MachineRegisterClassDef):
                    value_matchers.append(
                        (name, RegClassMatcher(operand)))
                else:
                    raise ValueError()
            elif isinstance(matcher_or_tuple, MachineRegisterDef):
                value_matchers.append(
                    (matcher_or_tuple, RegValueMatcher(matcher_or_tuple)))
            else:
                raise ValueError()

        if isinstance(operands[-1], tuple):
            name, operand = operands[-1]
            if isinstance(operand, NodeValuePatternMatcher):
                value_matchers.append((name, operand))
            elif isinstance(operand, MachineRegisterClassDef):
                operand_matchers.append((name, RegClassMatcher(operand)))
            else:
                raise ValueError()

            return NodePatternMatcher(opcode_matcher, operand_matchers, value_matchers, self.props)
        else:
            assert(isinstance(operands[-1], NodePatternMatcher))
            operands[-1].values = value_matchers
            return operands[-1]


set_ = SetPatternMatcherGen()
