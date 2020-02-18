"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import platform
import numpy as np
from numpy.testing import assert_approx_equal

from eko.alpha_s import beta_0, beta_1, beta_2, StrongCoupling
from eko.constants import Constants

# TODO @JCM+@SC: you may want to add your match here
use_LHAPDF = platform.node() in ["FHe19b"]
if use_LHAPDF:
    import lhapdf

# these tests will only pass for the default set of constants
constants = Constants()
CA = constants.CA
CF = constants.CF
TF = constants.TF


def flav_test(function):
    """ Check that the given beta function `function` is valid
    for any number of flavours up to 5 """
    for nf in range(5):
        result = function(nf, CA, CF, TF)
        assert result > 0.0


def check_result(function, NF, value):
    """ Check that function evaluated in nf=5
    returns the value `value` """
    result = function(NF, CA, CF, TF)
    assert_approx_equal(result, value, significant=5)


def test_beta_0():
    """Test first beta function coefficient"""
    flav_test(beta_0)
    check_result(beta_0, 5, 23 / 3)


def test_beta_1():
    """Test second beta function coefficient"""
    flav_test(beta_1)
    check_result(beta_1, 5, 116 / 3)


def test_beta_2():
    """Test third beta function coefficient"""
    flav_test(beta_2)
    check_result(beta_2, 5, 9769 / 54)


def test_a_s():
    """ Tests the value of alpha_s (for now only at LO)
    for a given set of parameters
    """
    known_vals = {0: 0.0091807954}
    ref_as = 0.1181
    ref_mu = 90
    ask_q2 = 125
    as_FFNS_LO = StrongCoupling(constants, ref_as, ref_mu, 0, "FFNS", nf=5)
    for order in range(1):
        result = as_FFNS_LO.a_s(ask_q2)
        assert_approx_equal(result, known_vals[order], significant=7)


def test_LHA_benchmark_paper():
    """Check to :cite:`Giele:2002hx` and :cite:`Dittmar:2005ed`"""
    # LO - FFNS
    # note that the LO-FFNS value reported in :cite:`Giele:2002hx`
    # was corrected in :cite:`Dittmar:2005ed`
    as_FFNS_LO = StrongCoupling(constants, 0.35, 2, 0, "FFNS", nf=4)
    me = as_FFNS_LO.a_s(1e4) * 4 * np.pi
    ref = 0.117574
    assert_approx_equal(me, ref, significant=6)
    # LO - VFNS
    as_VFNS_LO = StrongCoupling(
        constants, 0.35, 2, 0, "VFNS", thresholds=[2, 4.5 ** 2, 175 ** 2]
    )
    me = as_VFNS_LO.a_s(1e4) * 4 * np.pi
    ref = 0.122306
    assert_approx_equal(me, ref, significant=6)


def _get_Lambda2_LO(as_ref, scale_ref, nf):
    """Transformation to Lambda_QCD"""
    beta0 = beta_0(nf, CA, CF, TF)
    return scale_ref * np.exp(-1.0 / (as_ref * beta0))


def test_lhapdf_ffns_lo():
    """test FFNS LO towards LHAPDF"""
    Q2s = [1, 1e1, 1e2, 1e3, 1e4]
    alphas_ref = 0.118
    scale_ref = 91.0 ** 2
    nf = 4
    # collect my values
    as_FFNS_LO = StrongCoupling(
        constants, alphas_ref, scale_ref, 0, "FFNS", nf=nf, method="analytic"
    )
    my_vals = []
    for Q2 in Q2s:
        my_vals.append(as_FFNS_LO.a_s(Q2))
    # LHAPDF cache
    lhapdf_vals = np.array(
        [
            0.031934929816669545,
            0.019801241565290697,
            0.01434924187307247,
            0.01125134004424113,
            0.009253560493881005,
        ]
    )
    if use_LHAPDF:
        # run lhapdf
        as_lhapdf = lhapdf.mkBareAlphaS("analytic")
        as_lhapdf.setOrderQCD(1)
        as_lhapdf.setFlavorScheme("FIXED", nf)
        Lambda2 = _get_Lambda2_LO(alphas_ref / (4.0 * np.pi), scale_ref, nf)
        as_lhapdf.setLambda(nf, np.sqrt(Lambda2))
        # collect a_s
        lhapdf_vals_cur = []
        for Q2 in Q2s:
            lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2) / (4.0 * np.pi))
        # print(lhapdf_vals_cur)
        np.testing.assert_allclose(lhapdf_vals, np.array(lhapdf_vals_cur))
    # check
    np.testing.assert_allclose(lhapdf_vals, np.array(my_vals))


def test_lhapdf_zmvfns_lo():
    """test ZM-VFNS LO towards LHAPDF"""
    Q2s = [1, 1e1, 1e2, 1e3, 1e4]
    alphas_ref = 0.118
    scale_ref = 900
    m2c = 2
    m2b = 25
    m2t = 1500
    thresholds = [m2c, m2b, m2t]
    # compute all Lambdas
    #Lambda2_5 = _get_Lambda2_LO(alphas_ref / (4.0 * np.pi), scale_ref, 5)
    #as_FFNS_LO_5 = StrongCoupling(
    #    constants, alphas_ref, scale_ref, 0, "FFNS", nf=5, method="analytic"
    #)
    #Lambda2_6 = _get_Lambda2_LO(as_FFNS_LO_5.a_s(m2t), m2t, 6)
    #as_b = as_FFNS_LO_5.a_s(m2b)
    #Lambda2_4 = _get_Lambda2_LO(as_b, m2b, 4)
    #as_FFNS_LO_4 = StrongCoupling(
    #    constants, as_b * 4.0 * np.pi, m2b, 0, "FFNS", nf=4, method="analytic"
    #)
    #Lambda2_3 = _get_Lambda2_LO(as_FFNS_LO_4.a_s(m2c), m2c, 3)

    # collect my values
    as_VFNS_LO = StrongCoupling(
        constants,
        alphas_ref,
        scale_ref,
        0,
        "VFNS",
        thresholds=thresholds,
        method="analytic",
    )
    my_vals = []
    for Q2 in Q2s:
        my_vals.append(as_VFNS_LO.a_s(Q2))
    # LHAPDF cache
    lhapdf_vals = np.array(
        [
            0.01932670387675251,
            0.014008394237618302,
            0.011154570468393434,
            0.009319430765984453,
            0.008084615274633044,
        ]
    )
    if use_LHAPDF:
        # run lhapdf - actually, let's use a different implementation here!
        # as_lhapdf = lhapdf.mkBareAlphaS("analytic")
        as_lhapdf = lhapdf.mkBareAlphaS("ODE")
        as_lhapdf.setOrderQCD(1)
        as_lhapdf.setFlavorScheme("VARIABLE", -1)
        as_lhapdf.setAlphaSMZ(alphas_ref)
        as_lhapdf.setMZ(np.sqrt(scale_ref))
        for k in range(3):
            as_lhapdf.setQuarkMass(1 + k, 0)
        for k, m2 in enumerate(thresholds):
            as_lhapdf.setQuarkMass(4 + k, np.sqrt(m2))
        # as_lhapdf.setLambda(3, np.sqrt(Lambda2_3))
        # as_lhapdf.setLambda(4, np.sqrt(Lambda2_4))
        # as_lhapdf.setLambda(5, np.sqrt(Lambda2_5))
        # as_lhapdf.setLambda(6, np.sqrt(Lambda2_6))
        # collect a_s
        lhapdf_vals_cur = []
        for Q2 in Q2s:
            lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2) / (4.0 * np.pi))
        # print(lhapdf_vals_cur)
        np.testing.assert_allclose(lhapdf_vals, np.array(lhapdf_vals_cur))
    # check - tolerance is determined from
    # Max absolute difference: 2.58611282e-06
    # Max relative difference: 0.00013379
    np.testing.assert_allclose(lhapdf_vals, np.array(my_vals), rtol=1.5e-4)


if __name__ == "__main__":
    test_lhapdf_zmvfns_lo()
