"""CFFI interface."""
from cffi import FFI

ffi = FFI()
cernlibc = ffi.dlopen("./ekorepp.so")

ffi.cdef(
    """
double c_getdouble(void* ptr);
int c_ad_us_gamma_ns(const unsigned int order_qcd, const unsigned int mode, const double re_in, const double im_in, const unsigned int nf, void* re_out, void* im_out);
int c_ad_us_gamma_singlet(const unsigned int order_qcd, const double re_in, const double im_in, const unsigned int nf, void* re_out, void* im_out);
"""
)

# we need to "activate" the actual function first
c_getdouble = cernlibc.c_getdouble
c_ad_us_gamma_ns = cernlibc.c_ad_us_gamma_ns
c_ad_us_gamma_singlet = cernlibc.c_ad_us_gamma_singlet

# allocate the pointers and get their addresses
re_double_4 = ffi.new("double[4]")
im_double_4 = ffi.new("double[4]")
re_double_4_address = int(ffi.cast("uintptr_t", re_double_4))
im_double_4_address = int(ffi.cast("uintptr_t", im_double_4))

re_double_16 = ffi.new("double[16]")
im_double_16 = ffi.new("double[16]")
re_double_16_address = int(ffi.cast("uintptr_t", re_double_16))
im_double_16_address = int(ffi.cast("uintptr_t", im_double_16))
