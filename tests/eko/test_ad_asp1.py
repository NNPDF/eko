# Test LO Polarised splitting functions
import numpy as np
import eko.anomalous_dimensions.asp1 as ad_asp1
from eko import harmonics

NF = 5


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad_asp1.gamma_pns(N, s1) + ad_asp1.gamma_pgq(N),
        0,
    )


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad_asp1.gamma_pqg(N, NF) + ad_asp1.gamma_pgg(N, s1, NF), 0
    )
