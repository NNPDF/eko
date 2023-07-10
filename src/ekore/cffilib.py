"""CFFI interface."""
from cffi import FFI

ffi = FFI()
cernlibc = ffi.dlopen("./ekorepp.so")

ffi.cdef(
    """
int c_cern_polygamma(const double re_in, const double im_in, const unsigned int K, void* re_out, void* im_out);
double c_getdouble(void* ptr);
int c_ad_us_gamma_ns(const unsigned int order_qcd, const unsigned int mode, const double re_in, const double im_in, const unsigned int nf, void* re_out, void* im_out);
"""
)

# we need to "activate" the actual function first
c_cern_polygamma = cernlibc.c_cern_polygamma
c_getdouble = cernlibc.c_getdouble
c_ad_us_gamma_ns = cernlibc.c_ad_us_gamma_ns

# allocate the pointers and get their addresses
re_y = ffi.new("double*")
im_y = ffi.new("double*")
re_y_address = int(ffi.cast("uintptr_t", re_y))
im_y_address = int(ffi.cast("uintptr_t", im_y))

re_double_3 = ffi.new("double[3]")
im_double_3 = ffi.new("double[3]")
re_double_3_address = int(ffi.cast("uintptr_t", re_double_3))
im_double_3_address = int(ffi.cast("uintptr_t", im_double_3))
