# -*- coding: utf-8 -*-
"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import numpy as np
import pytest

from eko import beta


def _flav_test(function):
    """Check that the given beta function `function` is valid
    for any number of flavors up to 5"""
    for nf in range(5):
        result = function(nf)
        assert result > 0.0


def test_beta_0():
    """Test first beta function coefficient"""
    _flav_test(beta.beta_0)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta.beta_0(5), 4 * 23 / 12)


def test_beta_1():
    """Test second beta function coefficient"""
    _flav_test(beta.beta_1)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta.beta_1(5), 4**2 * 29 / 12)


def test_beta_2():
    """Test third beta function coefficient"""
    _flav_test(beta.beta_2)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta.beta_2(5), 4**3 * 9769 / 3456)


def test_beta():
    """beta-wrapper"""
    nf = 3
    np.testing.assert_allclose(beta.beta(0, nf), beta.beta_0(nf))
    np.testing.assert_allclose(beta.beta(1, nf), beta.beta_1(nf))
    np.testing.assert_allclose(beta.beta(2, nf), beta.beta_2(nf))
    with pytest.raises(ValueError):
        beta.beta(3, 3)


def test_b():
    """b-wrapper"""
    np.testing.assert_allclose(beta.b(0, 3), 1.0)
