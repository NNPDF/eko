# Test NLO Polarized splitting functions
import numpy as np

import ekore.anomalous_dimensions.polarized.space_like.as2 as as2
from eko.constants import CA, CF, TR
from ekore import harmonics

nf = 5


def test_qg_helicity_conservation():
    N = complex(1.0, 0.0)
    sx = harmonics.sx(N, max_weight=4)
    np.testing.assert_almost_equal(as2.gamma_qg(N, nf, sx), 0)


def test_ps_momentum():
    N = complex(2.0, 0.0)
    np.testing.assert_allclose(-as2.gamma_ps(N, nf), -4.0 * TR * nf * CF * 13 / 27)


def test_qg_momentum():
    N = complex(2.0, 0.0)
    sx = harmonics.sx(N, 2)
    np.testing.assert_allclose(
        -as2.gamma_qg(N, nf, sx),
        4
        * nf
        * (0.574074 * CF - 2 * CA * (-7 / 18 + 1 / 6 * (5 - np.pi**2 / 3)))
        * TR,
    )


def test_gq_momentum():
    N = complex(2.0, 0.0)
    sx = harmonics.sx(N, 2)
    np.testing.assert_allclose(
        -as2.gamma_gq(N, nf, sx),
        4
        * (
            -2.074074074074074 * CF**2
            + CA * CF * (29 / 54 - 2 / 3 * (1 / 2 - np.pi**2 / 3))
            + (4 * CF * nf * TR) / 27
        ),
    )


def test_gg_momentum():
    N = complex(2.0, 0.0)
    sx = harmonics.sx(N, 2)
    np.testing.assert_almost_equal(
        -as2.gamma_gg(N, nf, sx),
        4
        * (-1.7537256813471833 * CA**2 + ((29 * CA) / 27 - (28 * CF) / 27) * nf * TR),
    )
