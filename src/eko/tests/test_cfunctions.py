"""
    Test the C-functions implemented with the CFFI wrapper
"""

import scipy.special
import numpy as np
from eko import t_float, t_complex
import _gsl_digamma
c_digamma = _gsl_digamma.lib.digamma


def compare_functions(function_custom, function_ext, values):
    for value in values:
        custom = function_custom(value)
        extern = function_ext(value)
        np.testing.assert_approx_equal(custom, extern, significant=8)


def test_digamma():
    values = [t_float(1.3), t_float(0.56), t_complex(0.3 + 0.4j)]

    def customfun(x):
        out = (np.empty(2))
        if isinstance(x, t_float):
            c_digamma(x, 0.0, _gsl_digamma.ffi.from_buffer(out))
            return out[0]
        elif isinstance(x, t_complex):
            r = np.real(x)
            i = np.imag(x)
            c_digamma(r, i, _gsl_digamma.ffi.from_buffer(out))
            return t_complex(np.complex(out[0], out[1]))

    compare_functions(customfun, scipy.special.digamma, values)
