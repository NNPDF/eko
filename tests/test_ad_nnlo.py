# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np

import eko.anomalous_dimensions.nnlo as ad_nnlo
from eko.anomalous_dimensions import harmonics

NF = 5

def get_sx(N):
    """Collect the S-cache"""
    sx = np.full(1, harmonics.harmonic_S1(N))
    sx = np.append(sx, harmonics.harmonic_S2(N))
    sx = np.append(sx, harmonics.harmonic_S3(N))
    sx = np.append(sx, harmonics.harmonic_S4(N))
    return sx


def test_gamma_2():
    # number conservation - each is 0 on its own, see :cite:`Moch:2004pa`
    N = 1
    sx = get_sx(N)
    np.testing.assert_allclose(ad_nnlo.gamma_nsv_2(N, NF, sx), 0, atol=1e-3)
    np.testing.assert_allclose(ad_nnlo.gamma_nsm_2(N, NF, sx), 0, atol=7e-4)

    # get singlet sector
    N = 2
    sx = get_sx(N)
    gS2 = ad_nnlo.gamma_singlet_2(N, NF, sx)

    # gluon momentum conservation
    np.testing.assert_allclose(gS2[0, 1] + gS2[1, 1], 0, atol=4e-3)
    # quark momentum conservation
    np.testing.assert_allclose(gS2[0, 0] + gS2[1, 0], 0, atol=2e-3)

    assert gS2.shape == (2, 2)
