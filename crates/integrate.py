import ekuad
from scipy import LowLevelCallable, integrate

extra_ptr = ekuad.ffi.new("Extra *", (10.0, 1000.0))
extra_void = ekuad.ffi.cast("void *", extra_ptr)
func = LowLevelCallable(ekuad.lib.quad_ker, extra_void)

print(integrate.quad(func, 0.0, 1.0))
