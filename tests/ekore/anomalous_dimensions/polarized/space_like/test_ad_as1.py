# Test LO Polarised splitting functions
import numpy as np
import ekore.anomalous_dimensions.polarized.space_like.as1 as as1
from ekore import harmonics
from eko import constants

NF = 5


def test_quark_momentum():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        as1.gamma_ns(N, s1) + as1.gamma_gq(N),
        (4 * constants.CF)/3,
    )


def test_gluon_momentum():
    # gluon momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        as1.gamma_qg(N, NF) + as1.gamma_gg(N, s1, NF), 3 + NF/3
    )

def test_qg_helicity_conservation():
    N = complex(1.0, 0.0)
    np.testing.assert_almost_equal(as1.gamma_qg(N, NF) , 0)
    
