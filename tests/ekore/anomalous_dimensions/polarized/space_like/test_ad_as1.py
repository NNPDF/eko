# Test LO Polarised splitting functions
import numpy as np
import ekore.anomalous_dimensions.polarized.space_like.as1 as as1
from ekore import harmonics

NF = 5


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        as1.gamma_ns(N, s1) + as1.gamma_gq(N),
        1.7777777777777777,
    )


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        as1.gamma_qg(N, NF) + as1.gamma_gg(N, s1, NF), 4.666666666666668
    )

def test_qg_number_conservation():
    N = complex(1.0, 0.0)
    np.testing.assert_almost_equal(as1.gamma_qg(N, NF) , 0)
    
