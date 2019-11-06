"""
    Test the C-functions implemented with the CFFI wrapper
"""

import scipy.special
import numpy as np
from eko import t_float, t_complex
from _gsl_digamma import lib


def compare_functions(function_custom, function_ext, values):
    for value in values:
        custom = function_custom(value)
        extern = function_ext(value)
        np.testing.assert_approx_equal(custom, extern, significant=8)


def test_digamma():
    values = [t_float(1.3), t_float(0.56), t_complex(0.3 + 0.4j)]

    def customfun(x):
        if isinstance(x, t_float):
            result = lib.digamma(x, 0.0)
            return result.r
        elif isinstance(x, t_complex):
            r = np.real(x)
            i = np.imag(x)
            result = lib.digamma(r, i)
            return t_complex(np.complex(result.r, result.i))

    compare_functions(customfun, scipy.special.digamma, values)
