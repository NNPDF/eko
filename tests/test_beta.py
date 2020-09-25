# -*- coding: utf-8 -*-
"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import numpy as np

from eko.beta import beta_0, beta_1, beta_2

def _flav_test(function):
    """Check that the given beta function `function` is valid
    for any number of flavours up to 5"""
    for nf in range(5):
        result = function(nf)
        assert result > 0.0

def test_beta_0():
    """Test first beta function coefficient"""
    _flav_test(beta_0)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta_0(5), 4 * 23 / 12)

def test_beta_1():
    """Test second beta function coefficient"""
    _flav_test(beta_1)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta_1(5), 4 ** 2 * 29 / 12)

def test_beta_2():
    """Test third beta function coefficient"""
    _flav_test(beta_2)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta_2(5), 4 ** 3 * 9769 / 3456)
