# Test LO Polarized splitting functions
import numpy as np

import ekore.anomalous_dimensions.polarized.space_like.as3 as as3
from ekore import harmonics

nf = 5


# def test_quark_momentum():
#     # quark momentum
#     N = complex(2.0, 0.0)
#     sx = harmonics.sx(N, max_weight=4)
#     np.testing.assert_almost_equal(
#         as3.gamma_ns(N, nf, sx) + as3.gamma_gq(N),
#         (4 * constants.CF)/3,
#     )


def test_gluon_momentum():
    # gluon momentum
    N = complex(2.0, 0.0)
    sx = harmonics.sx(N, max_weight=4)
    np.testing.assert_allclose(
        as3.gamma_qg(N, nf, sx) + as3.gamma_gg(N, nf, sx), 9.26335, rtol=7e-4
    )


def test_qg_helicity_conservation():
    N = complex(1.0, 0.0)
    sx = harmonics.sx(N, max_weight=4)
    np.testing.assert_almost_equal(as3.gamma_qg(N, nf, sx), 0.00294317)
