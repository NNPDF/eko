"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""

import numpy as np
import pytest

from eko import beta
from eko.constants import uplike_flavors, zeta3


def _flav_test(function):
    """Check that the given beta function `function` is valid
    for any number of flavors up to 5"""
    for nf in range(5):
        result = function(nf)
        assert result > 0.0


def test_beta_as2():
    """Test first beta function coefficient"""
    _flav_test(beta.beta_qcd_as2)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta.beta_qcd_as2(5), 4 * 23 / 12)


def test_beta_aem2():
    """Test first beta function coefficient"""
    # from hep-ph/9803211
    for nf in range(3, 6 + 1):
        for nl in range(2, 3 + 1):
            nu = uplike_flavors(nf)
            nd = nf - nu
            np.testing.assert_approx_equal(
                beta.beta_qed_aem2(nf, nl),
                -4.0 / 3 * (nl + 3 * (nu * 4 / 9 + nd * 1 / 9)),
            )


def test_beta_as3():
    """Test second beta function coefficient"""
    _flav_test(beta.beta_qcd_as3)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta.beta_qcd_as3(5), 4**2 * 29 / 12)


def test_beta_aem3():
    """Test second beta function coefficient"""
    # from hep-ph/9803211
    for nf in range(3, 6 + 1):
        for nl in range(2, 3 + 1):
            nu = uplike_flavors(nf)
            nd = nf - nu
            np.testing.assert_approx_equal(
                beta.beta_qed_aem3(nf, nl),
                -4.0 * (nl + 3 * (nu * 16 / 81 + nd * 1 / 81)),
            )


def test_beta_as4():
    """Test third beta function coefficient"""
    _flav_test(beta.beta_qcd_as4)
    # from hep-ph/9706430
    np.testing.assert_approx_equal(beta.beta_qcd_as4(5), 4**3 * 9769 / 3456)


def test_beta_as5():
    """Test fourth beta function coefficient"""
    _flav_test(beta.beta_qcd_as5)
    # from hep-ph/9706430
    np.testing.assert_allclose(
        beta.beta_qcd_as5(5), 4**4 * (11027.0 / 648.0 * zeta3 - 598391.0 / 373248.0)
    )


def test_beta():
    """beta-wrapper"""
    for nf in range(3, 6 + 1):
        for nl in range(2, 3 + 1):
            np.testing.assert_allclose(beta.beta_qcd((2, 0), nf), beta.beta_qcd_as2(nf))
            np.testing.assert_allclose(beta.beta_qcd((3, 0), nf), beta.beta_qcd_as3(nf))
            np.testing.assert_allclose(beta.beta_qcd((4, 0), nf), beta.beta_qcd_as4(nf))
            np.testing.assert_allclose(beta.beta_qcd((5, 0), nf), beta.beta_qcd_as5(nf))
            np.testing.assert_allclose(
                beta.beta_qcd((2, 1), nf), beta.beta_qcd_as2aem1(nf)
            )
            np.testing.assert_allclose(
                beta.beta_qed((0, 2), nf, nl), beta.beta_qed_aem2(nf, nl)
            )
            np.testing.assert_allclose(
                beta.beta_qed((0, 3), nf, nl), beta.beta_qed_aem3(nf, nl)
            )
    np.testing.assert_allclose(beta.beta_qed((1, 2), nf, nl), beta.beta_qed_aem2as1(nf))
    with pytest.raises(ValueError):
        beta.beta_qcd((6, 0), 3)
    with pytest.raises(ValueError):
        beta.beta_qed((0, 4), 3, 3)


def test_b():
    """b-wrapper"""
    np.testing.assert_allclose(beta.b_qcd((2, 0), 3), 1.0)
    np.testing.assert_allclose(beta.b_qed((0, 2), 3, 3), 1.0)


def test_zero():
    """test that beta_qed is zero for nf=nl=0"""
    for ord in range(2, 3 + 1):
        np.testing.assert_allclose(beta.beta_qed((0, ord), 0, 0), 0)


def test_linearity():
    """test linearity of QED beta function when nf=0"""
    for ord in range(2, 3 + 1):
        for nl in [0, 2]:
            np.testing.assert_allclose(
                beta.beta_qed((0, ord), 0, nl) / 2, beta.beta_qed((0, ord), 0, nl / 2)
            )
