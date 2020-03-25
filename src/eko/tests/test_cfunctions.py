"""
    Test the C-functions implemented with the CFFI wrapper
"""

import scipy.special
import numpy as np
from eko import t_float, t_complex
import _gsl_digamma
c_digamma = _gsl_digamma.lib.digamma # pylint: disable=no-member


def compare_functions(function_custom, function_ext, values):
    customs = []
    externs = []
    for value in values:
        custom = function_custom(value)
        extern = function_ext(value)
        customs.append(custom)
        externs.append(extern)
    np.testing.assert_allclose(customs, externs)

def test_digamma():
    values = []
    for i,j in np.random.rand(20, 2):
        if np.random.rand() < 0.25:
            values.append(t_float(i+j))
        else:
            values.append(t_complex(complex(i,j)))

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
        raise ValueError(f"unknown x = {x}")

    compare_functions(customfun, scipy.special.digamma, values)
