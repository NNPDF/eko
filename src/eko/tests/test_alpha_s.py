"""
    This module tests the implemented beta functions and the value
    of alpha_s for different orders.
"""
import platform
import numpy as np
from numpy.testing import assert_approx_equal

from eko.alpha_s import beta_0, beta_1, beta_2, StrongCoupling
from eko.constants import Constants

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
    beta0 = beta_0(nf,CA,CF,TF)
    return scale_ref * np.exp(-1.0/(as_ref * beta0))


def test_lhapdf():
    """test towards LHAPDF"""
    Q2s = [1,1e1,1e2,1e3,1e4]
    as_ref = 0.118/(4.0*np.pi)
    scale_ref = 91.0**2
    nf = 4
    as_FFNS_LO = StrongCoupling(constants, as_ref*4*np.pi, scale_ref, 0, "FFNS", nf=nf, method="analytic")
    my_vals = []
    for Q2 in Q2s:
        my_vals.append(as_FFNS_LO.a_s(Q2))
    print(my_vals)
    #lhapdf_vals = [0.31145233053669075,0.20076769332053362,0.15014705262149175,0.12048646995455557,0.10084143813676005]
    if use_LHAPDF:
        as_lhapdf = lhapdf.mkBareAlphaS("analytic")
        as_lhapdf.setOrderQCD(0)
        as_lhapdf.setFlavorScheme("FIXED",nf)
        Lambda2 = _get_Lambda2_LO(as_ref,scale_ref,nf)
        as_lhapdf.setLambda(nf, np.sqrt(Lambda2))
        lhapdf_vals_cur = []
        for Q2 in Q2s:
            lhapdf_vals_cur.append(as_lhapdf.alphasQ2(Q2)/(4.0*np.pi))
        print(lhapdf_vals_cur)
        #np.assert_approx_equal(lhapdf_vals,lhapdf_vals_cur)
    print(np.array(my_vals)/np.array(lhapdf_vals_cur))


if __name__ == "__main__":
    test_lhapdf()
