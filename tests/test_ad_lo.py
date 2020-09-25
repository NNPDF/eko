# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

import eko.anomalous_dimensions.lo as ad_lo
from eko import ekomath

NF = 5


def test_number_momentum_conservation():
    # number
    N = complex(1.0, 0.0)
    s1 = ekomath.harmonic_S1(N)
    np.testing.assert_almost_equal(ad_lo.gamma_ns_0(N, s1), 0)


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = ekomath.harmonic_S1(N)
    np.testing.assert_almost_equal(ad_lo.gamma_ns_0(N, s1) + ad_lo.gamma_gq_0(N), 0)


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    s1 = ekomath.harmonic_S1(N)
    np.testing.assert_almost_equal(
        ad_lo.gamma_qg_0(N, NF) + ad_lo.gamma_gg_0(N, s1, NF), 0
    )


def test_gamma_qg_0():
    N = complex(1.0, 0.0)
    res = complex(-20.0 / 3.0, 0.0)
    np.testing.assert_almost_equal(ad_lo.gamma_qg_0(N, NF), res)


def test_gamma_gq_0():
    N = complex(0.0, 1.0)
    res = complex(4.0, -4.0) / 3.0
    np.testing.assert_almost_equal(ad_lo.gamma_gq_0(N), res)


def test_gamma_gg_0():
    N = complex(0.0, 1.0)
    s1 = ekomath.harmonic_S1(N)
    res = complex(5.195725159621, 10.52008856962)
    np.testing.assert_almost_equal(ad_lo.gamma_gg_0(N, s1, NF), res)
