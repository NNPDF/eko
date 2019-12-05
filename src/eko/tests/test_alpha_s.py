"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.

    The test consist on checking that the beta function is valid
    for a number of flavours and that the result is correct for
    nf = 5
"""

from numpy.testing import assert_approx_equal

from eko import t_float
from eko.alpha_s import beta_0, beta_1, beta_2, Alphas_Dispatcher 
from eko.constants import Constants

# the test will only pass for the default set of constants
constants = Constants()
CA = constants.CA
CF = constants.CF
Tf = constants.TF
NF = 5


def flav_test(function):
    """ Check that the given beta function `function` is valid
    for any number of flavours up to 5 """
    for nf in range(5):
        result = function(nf, CA, CF, Tf)
        assert result > 0.0


def check_result(function, value):
    """ Check that function evaluated in nf=5
    returns the value `value` """
    result = function(NF, CA, CF, Tf)
    assert_approx_equal(result, value, significant=5)


def test_beta_0():
    """Test first beta function coefficient"""
    flav_test(beta_0)
    check_result(beta_0, 23 / 3)


def test_beta_1():
    """Test second beta function coefficient"""
    flav_test(beta_1)
    check_result(beta_1, 116 / 3)


def test_beta_2():
    """Test third beta function coefficient"""
    flav_test(beta_2)
    check_result(beta_2, 9769 / 54)


def test_a_s():
    """ Tests the value of alpha_s (for now only at LO)
    for a given set of parameters
    """
    known_vals = {0: 0.0091807954}
    ref_as = 0.1181
    ref_mu = 90
    ask_q2 = 125
    alpha_s = Alphas_Dispatcher(constants, ref_as, ref_mu, NF, method = "None", order = 0)
    for order in range(1):
        result = alpha_s(ask_q2)
        assert_approx_equal(result, known_vals[order], significant=7)
