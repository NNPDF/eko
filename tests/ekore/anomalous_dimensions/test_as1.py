# Test LO splitting functions
import numpy as np

import ekore.anomalous_dimensions.aem1 as ad_aem1
import ekore.anomalous_dimensions.as1 as ad_as1
from ekore import harmonics

NF = 5


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(ad_as1.gamma_ns(N, s1), 0)


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad_as1.gamma_ns(N, s1) + ad_as1.gamma_gq(N),
        0,
    )


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad_as1.gamma_qg(N, NF) + ad_as1.gamma_gg(N, s1, NF), 0
    )


def test_gamma_qg_0():
    N = complex(1.0, 0.0)
    res = complex(-20.0 / 3.0, 0.0)
    np.testing.assert_almost_equal(ad_as1.gamma_qg(N, NF), res)


def test_gamma_gq_0():
    N = complex(0.0, 1.0)
    res = complex(4.0, -4.0) / 3.0
    np.testing.assert_almost_equal(ad_as1.gamma_gq(N), res)


def test_gamma_gg_0():
    N = complex(0.0, 1.0)
    s1 = harmonics.S1(N)
    res = complex(5.195725159621, 10.52008856962)
    np.testing.assert_almost_equal(ad_as1.gamma_gg(N, s1, NF), res)
