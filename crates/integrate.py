import ekuad
import numba as nb
import numpy as np
from scipy import LowLevelCallable, integrate


@nb.cfunc(
    nb.types.double(
        nb.types.double,
        nb.types.double,
        nb.types.CPointer(nb.types.double),
        nb.types.uint16,
        nb.types.CPointer(nb.types.double),
        nb.types.uint16,
    )
)
def true_py(x, y, ar_raw, len, areas_raw, areas_len):
    ar = nb.carray(ar_raw, len)
    areas = nb.carray(areas_raw, areas_len)
    return x + 1.23 * y + ar[0] + ar[1] + np.sum(areas)


areas = [2e6, 3e7]
extra = ekuad.lib.dummy()
extra.slope = 10.0
extra.shift = 1000.0
areas_ffi = ekuad.ffi.new("double[2]", areas)
extra.areas = areas_ffi
extra.areas_len = 2
extra.py = ekuad.ffi.cast("void *", true_py.address)
func = LowLevelCallable(ekuad.lib.quad_ker, ekuad.ffi.addressof(extra))

print(integrate.quad(func, 0.0, 1.0))
