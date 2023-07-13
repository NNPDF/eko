"""CFFI interface."""
import numba as nb
import numpy as np
import numpy.typing as npt
from cffi import FFI

ffi = FFI()
ekorepplibc = ffi.dlopen("./ekorepp.so")

ffi.cdef(
    """
double c_getdouble(void* ptr);
int c_ad_us_gamma_ns(const unsigned int order_qcd, const unsigned int mode, const double re_in, const double im_in, const unsigned int nf, void* re_out, void* im_out);
int c_ad_us_gamma_singlet(const unsigned int order_qcd, const double re_in, const double im_in, const unsigned int nf, void* re_out, void* im_out);
double c_quad_ker_qcd(const double u,
                    const unsigned int order_qcd,
                    const unsigned int mode0, const unsigned int mode1,
                    const bool is_polarized, const bool is_time_like,
                    const unsigned int nf,
                    void* py,
                    const bool is_log, const double logx, void* areas, const unsigned int polynomial_degree,
                    const double L,
                    const unsigned int method_num, const double as1, const double as0,
                    const unsigned int ev_op_iterations, const unsigned int ev_op_max_order,
                    const unsigned int sv_mode_num, const bool is_threshold);
"""
)

# we need to "activate" the actual function first
c_getdouble = ekorepplibc.c_getdouble
c_ad_us_gamma_ns = ekorepplibc.c_ad_us_gamma_ns
c_ad_us_gamma_singlet = ekorepplibc.c_ad_us_gamma_singlet
c_quad_ker_qcd = ekorepplibc.c_quad_ker_qcd

# allocate the pointers and get their addresses
MAX_DOUBLES = 16
_re_double = ffi.new(f"double[{MAX_DOUBLES}]")
_im_double = ffi.new(f"double[{MAX_DOUBLES}]")
re_double_address = int(ffi.cast("uintptr_t", _re_double))
im_double_address = int(ffi.cast("uintptr_t", _im_double))


@nb.njit()
def read_complex(size: int) -> npt.ArrayLike:
    """Read `size` complex numbers from C."""
    if size > MAX_DOUBLES:
        raise MemoryError("Not enough memory allocated")
    res = np.zeros(size, np.complex_)
    for j in range(size):
        res[j] = (
            c_getdouble(re_double_address + j * 8)
            + c_getdouble(im_double_address + j * 8) * 1j
        )
    return res
