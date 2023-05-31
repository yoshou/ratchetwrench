from enum import Enum, auto


class AllocationPurpose(Enum):
    Code = auto()
    ROData = auto()
    RWData = auto()


class MemoryManager:
    def __init__(self):
        pass

    def allocate_mapped_mem(self, size, alignment, purpose):
        raise NotImplementedError()

    def copy_to_mem(self, ptr, buffer):
        raise NotImplementedError()

    def copy_from_mem(self, ptr, buffer):
        raise NotImplementedError()

    def fill_to_mem(self, ptr, size, value):
        raise NotImplementedError()

    def enable_code_protect(self, ptr, size):
        raise NotImplementedError()

    def disable_code_protect(self, ptr, size):
        raise NotImplementedError()

    def allocate_data(self, size, alignment, is_readonly):
        if is_readonly:
            return self.allocate_mapped_mem(size, alignment, AllocationPurpose.ROData)

        return self.allocate_mapped_mem(size, alignment, AllocationPurpose.RWData)

    def allocate_code(self, size, alignment):
        return self.allocate_mapped_mem(size, alignment, AllocationPurpose.Code)
