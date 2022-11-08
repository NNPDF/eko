"""
    This module tests the implemented gamma functions
"""
import numpy as np
import pytest

from eko import gamma


def test_gamma():
    """gamma-wrapper"""
    nf = 3
    np.testing.assert_allclose(gamma.gamma(1, nf), gamma.gamma_qcd_as1())
    np.testing.assert_allclose(gamma.gamma(2, nf), gamma.gamma_qcd_as2(nf))
    np.testing.assert_allclose(gamma.gamma(3, nf), gamma.gamma_qcd_as3(nf))
    np.testing.assert_allclose(gamma.gamma(4, nf), gamma.gamma_qcd_as4(nf))
    with pytest.raises(ValueError):
        gamma.gamma(5, 3)
