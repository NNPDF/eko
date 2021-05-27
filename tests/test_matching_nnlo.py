# -*- coding: utf-8 -*-
# Test NNLO OME

import numpy as np

from eko.matching_conditions.nnlo import A_ns_2, A_singlet_2
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
    aNS2 = A_ns_2(N, sx)
    np.testing.assert_allclose(aNS2[0, 0], 0.0, atol=3e-7)

    # get singlet sector
    N = 2
    sx = get_sx(N)
    aS2 = A_singlet_2(N, sx)

    # gluon momentum conservation
    # Reference numbers coming from Mathematica
    np.testing.assert_allclose(aS2[0, 0] + aS2[1, 0], 0.00035576, rtol=1e-6)
    # quark momentum conservation
    np.testing.assert_allclose(aS2[1, 1] + aS2[0, 1], 0.0, atol=3e-7)

    assert aNS2.shape == (2, 2)
    assert aS2.shape == (3, 3)

    # check q line equal to the h line
    assert aNS2[0].all() == aNS2[1].all()
    assert aS2[1].all() == aS2[2].all()


def test_A_2_shape():

    N = 2
    sx = np.zeros(3, np.complex_)
    aNS2 = A_ns_2(N, sx)
    aS2 = A_singlet_2(N, sx)

    assert aNS2.shape == (2, 2)
    assert aS2.shape == (3, 3)

    # check q line equal to the h line for non intrinsic
    assert aS2[1].all() == aS2[2].all()
    assert aNS2[0].all() == aNS2[1].all()
