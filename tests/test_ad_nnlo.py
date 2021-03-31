# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np

import eko.anomalous_dimensions.nnlo as ad_nnlo
from eko.anomalous_dimensions import harmonics

NF = 5


def get_sx(N):
    """Collect the S-cache"""
    sx = np.array(
        [
            harmonics.harmonic_S1(N),
            harmonics.harmonic_S2(N),
            harmonics.harmonic_S3(N),
            harmonics.harmonic_S4(N),
        ]
    )
    return sx


# Reference numbers coming from Mathematica
def test_gamma_2():
    # number conservation - each is 0 on its own, see :cite:`Moch:2004pa`
    N = 1
    sx = get_sx(N)
    np.testing.assert_allclose(ad_nnlo.gamma_nsv_2(N, NF, sx), 0.000960586, rtol=3e-7)
    np.testing.assert_allclose(ad_nnlo.gamma_nsm_2(N, NF, sx), -0.000594225, rtol=6e-7)

    # get singlet sector
    N = 2
    sx = get_sx(N)
    gS2 = ad_nnlo.gamma_singlet_2(N, NF, sx)

    # gluon momentum conservation
    np.testing.assert_allclose(gS2[0, 1] + gS2[1, 1], 0.00388726, rtol=2e-6)
    # quark momentum conservation
    np.testing.assert_allclose(gS2[0, 0] + gS2[1, 0], -0.00169375, rtol=2e-6)

    assert gS2.shape == (2, 2)

    # test nsv_2 equal to referece value
    np.testing.assert_allclose(ad_nnlo.gamma_nsv_2(N, NF, sx), -188.325593, rtol=3e-7)
