"""CFFI interface."""
from cffi import FFI

ffi = FFI()
cernlibc = ffi.dlopen("./cernlibc.so")

ffi.cdef(
    """
int cern_polygamma(const double re_in, const double im_in, const unsigned int K, void* re_out, void* im_out);
double getdouble(void* ptr);
"""
)

# we need to "activate" the actual function first
c_cern_polygamma = cernlibc.cern_polygamma
c_getdouble = cernlibc.getdouble

# allocate the pointers and get their addresses
re_y = ffi.new("double*")
im_y = ffi.new("double*")
re_y_address = int(ffi.cast("uintptr_t", re_y))
im_y_address = int(ffi.cast("uintptr_t", im_y))
