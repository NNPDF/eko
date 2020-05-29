# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.special import digamma as scipy_digamma

from eko import ekomath
from eko import t_complex

def test_gsl_digamma():
    """ test the cffi implementation of digamma """
    for r, i in np.random.rand(4, 2):
        test_val = np.complex(r, i)
        scipy_result = scipy_digamma(test_val)
        gsl_result = ekomath.gsl_digamma(test_val)
        assert_almost_equal(scipy_result, gsl_result)


def test_harmonic_S1():
    """test harmonic sum S1"""
    # test on real axis
    known_vals = [1.0, 1.0 + 1.0 / 2.0, 1.0 + 1.0 / 2.0 + 1.0 / 3.0]
    for i, val in enumerate(known_vals):
        cval = t_complex(val)
        result = ekomath.harmonic_S1(1 + i)
        assert_almost_equal(result, cval)
