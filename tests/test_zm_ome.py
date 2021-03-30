# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np

from eko.matching_conditions import nnlo as zm_ome
from eko.anomalous_dimensions import harmonics


def get_sx(N):
    """Collect the S-cache"""
    sx = np.array(
        [
            harmonics.harmonic_S1(N),
            harmonics.harmonic_S2(N),
            harmonics.harmonic_S3(N),
        ]
    )
    return sx


def test_A_2():
    N = 1
    sx = get_sx(N)
    np.testing.assert_allclose(zm_ome.A_ns_2(N, sx), 0.0, atol=3e-7)

    # get singlet sector
    N = 2
    sx = get_sx(N)
    aS2 = zm_ome.A_singlet_2(N, sx)

    # gluon momentum conservation
    # Reference numbers coming from Mathematica
    np.testing.assert_allclose(aS2[0, 1] + aS2[1, 1], 0.00035576, rtol=1e-6)
    # quark momentum conservation
    np.testing.assert_allclose(aS2[0, 0] + aS2[1, 0], 0.0, atol=3e-7)

    assert aS2.shape == (2, 2)
