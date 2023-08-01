import ekuad
import numba as nb
from scipy import LowLevelCallable, integrate


@nb.cfunc(nb.types.double(nb.types.double, nb.types.double))
def true_py(a, b):
    return a + 1.23 * b


py_ptr = ekuad.ffi.cast("void *", true_py.address)
extra_ptr = ekuad.ffi.new("Extra *", (10.0, 10000.0, py_ptr))
extra_void = ekuad.ffi.cast("void *", extra_ptr)
func = LowLevelCallable(ekuad.lib.quad_ker, extra_void)

print(integrate.quad(func, 0.0, 1.0))
