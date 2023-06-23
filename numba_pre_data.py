from numba.core.types.containers import Container
from numba import types, typed
from numba.experimental import jitclass
from numba import cuda

@jitclass([('input_data', types.ListType(types.unicode_type)),
            ('target_data', types.ListType(types.unicode_type))])
class ContainerHolder(object):
    def __init__(self):
        self.input_data = typed.List.empty_list(types.unicode_type)
        self.target_data = typed.List.empty_list(types.unicode_type)