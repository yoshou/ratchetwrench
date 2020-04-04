import collections
import struct


class FieldInstance:
    pass


class Field:
    pass


class OrderedClassMembers(type):
    @classmethod
    def __prepare__(self, name, bases):
        return collections.OrderedDict()

    def __new__(self, name, bases, classdict):
        classdict['__ordered__'] = {key: classdict[key] for key in classdict.keys()
                                    if key not in ('__module__', '__qualname__') and isinstance(classdict[key], Field)}
        return type.__new__(self, name, bases, classdict)


class FormatableFieldBase(type):
    def __new__(cls, order, ty):

        class FormatableFieldInstance(FieldInstance):
            def __init__(self, order, ty, value, field):
                self.order = order
                self.ty = ty
                self.value = value
                self.field = field

            def __str__(self):
                return f"int({self.value})"

            def serialize(self, stream):
                bys = struct.pack(f"{self.order}{self.ty}", self.value)
                stream.write(bys)

            def deserialize(self, stream):
                size = struct.calcsize(self.ty)
                bys = stream.read(size)
                self.value = struct.unpack(f"{self.order}{self.ty}", bys)[0]

            def __eq__(self, other):
                if isinstance(other, (str, int, float)):
                    return self.value == other
                return self.value == other.value

        class FormatableField(Field):
            def __init__(self, default=None):
                self.default = default

            def __call__(self, val):
                self.verify_value(val)
                return FormatableFieldInstance(order, ty, val, self)

            def verify_value(self, val):
                if ty in ["b", "B", "h", "H", "i", "I", "q", "Q"]:
                    if not isinstance(val, int):
                        raise ValueError("The type of value must be int.")
                else:
                    raise ValueError()

            @property
            def size(self):
                return struct.calcsize(ty)

        return FormatableField


class ArrayFieldInstance(FieldInstance):
    def __init__(self, field, values):
        self.field = field
        self._values = values

    def __str__(self):
        return f"array([{', '.join([str(val) for val in self._values])}])"

    def serialize(self, output):
        for val in self._values:
            val.serialize(output)

    def deserialize(self, input):
        for val in self._values:
            val.deserialize(input)

    def __getitem__(self, index):
        return self._values[index]

    @property
    def values(self):
        return [item.value for item in self._values]


class ArrayField(Field):
    def __init__(self, ty, count, default=None):
        self.ty = ty
        self.count = count
        if default is not None:
            self.default = default
        else:
            self.default = [self.ty.default] * self.count

    def __call__(self, val):
        self.verify_value(val)
        val = [self.ty(elem) for elem in val]
        return ArrayFieldInstance(self, val)

    def verify_value(self, val):
        if not isinstance(val, list):
            raise ValueError()
        if len(val) != self.count:
            raise ValueError()

        for elem in val:
            self.ty.verify_value(elem)

    @property
    def size(self):
        if self.count == 0:
            return 0

        return self.ty.size * self.count


class Struct(metaclass=OrderedClassMembers):
    def __init__(self, **kwargs):
        self.fields = {}

        for name in self.__ordered__:
            ty = self.__ordered__[name]
            if name not in kwargs:
                if ty.default == None:
                    raise ValueError(
                        f"Field \"{name}\" is not passed by a argument.")
                val = ty.default
            else:
                val = kwargs[name]

            self.fields[name] = ty(val)

    def __getattribute__(self, name):
        ordered = super().__getattribute__("__ordered__")
        if name in ordered:
            fields = super().__getattribute__("fields")
            return fields[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, val):
        if name in self.__ordered__:
            ty = self.__ordered__[name]
            if isinstance(val, FieldInstance):
                self.fields[name] = val
            else:
                self.fields[name] = ty(val)
        else:
            super().__setattr__(name, val)

    def serialize(self, output):
        fields = super().__getattribute__("fields")
        for name in self.__ordered__:
            val = fields[name]
            val.serialize(output)

    def deserialize(self, input):
        fields = super().__getattribute__("fields")
        for name in self.__ordered__:
            val = fields[name]
            val.deserialize(input)

    @property
    def size(self):
        fields = super().__getattribute__("fields")
        size = 0
        for name in self.__ordered__:
            val = fields[name]
            size += val.field.size

        return size


class StructFieldInstance(FieldInstance):
    def __init__(self, field, value):
        self.field = field
        self.value = value

    def __str__(self):
        fields = ', '.join(
            [f"{name} : {str(val)}" for name, val in self.value])
        return f"struct([{fields}])"

    def serialize(self, output):
        for name, val in self.value:
            val.serialize(output)

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self, other):
        return hash(self.value)


class StructField(Field):
    def __init__(self, ty):
        self.ty = ty

    def __call__(self, val):
        self.verify_value(val)
        dic = collections.OrderedDict()
        fields = [(name, ty(val[name])) for name, ty in self.ty.__ordered__]
        return StructFieldInstance(self, fields)

    def verify_value(self, val):
        if not isinstance(val, dict):
            raise ValueError()

        for name, ty in ty.__ordered__.items():
            if name not in val:
                raise ValueError(f"The field \"{name}\ is not passed.")

            ty.verify_value(val[name])


class UInt64B(FormatableFieldBase(">", "Q")):
    pass


class SInt64B(FormatableFieldBase(">", "q")):
    pass


class UInt32B(FormatableFieldBase(">", "I")):
    pass


class SInt32B(FormatableFieldBase(">", "i")):
    pass


class UInt16B(FormatableFieldBase(">", "H")):
    pass


class SInt16B(FormatableFieldBase(">", "h")):
    pass


class UInt8B(FormatableFieldBase(">", "B")):
    pass


class SInt8B(FormatableFieldBase(">", "b")):
    pass


class UInt64L(FormatableFieldBase("<", "Q")):
    pass


class SInt64L(FormatableFieldBase("<", "q")):
    pass


class UInt32L(FormatableFieldBase("<", "I")):
    pass


class SInt32L(FormatableFieldBase("<", "i")):
    pass


class UInt16L(FormatableFieldBase("<", "H")):
    pass


class SInt16L(FormatableFieldBase("<", "h")):
    pass


class UInt8L(FormatableFieldBase("<", "B")):
    pass


class SInt8L(FormatableFieldBase("<", "b")):
    pass
