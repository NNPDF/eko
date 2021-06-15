# -*- coding: utf-8 -*-
# Test NNLO OME

import numpy as np

from eko.matching_conditions.nnlo import A_ns_2, A_singlet_2, A_qq_2_ns
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
    logs = [0, 100]

    for L in logs:
        N = 1
        sx = get_sx(N)
        aNSqq2 = A_qq_2_ns(N, sx, L)
        # quark number conservation
        np.testing.assert_allclose(aNSqq2, 0.0, atol=2e-11)

        N = 2
        sx = get_sx(N)
        aS2 = A_singlet_2(N, sx, L)

        # gluon momentum conservation
        # Reference numbers coming from Mathematica
        # note this difference is only due to the part non proportional to the logaritm
        np.testing.assert_allclose(aS2[0, 0] + aS2[1, 0] + aS2[2, 0], 0.00035576, rtol=1e-6)
        # quark momentum conservation
        np.testing.assert_allclose(aS2[0, 1] + aS2[1, 1] + aS2[2, 1], 0.0, atol=1e-11)

    aNS2 = A_ns_2(N, sx, L)
    assert aNS2.shape == (2, 2)
    assert aS2.shape == (3, 3)

    np.testing.assert_allclose( aS2[1,0], 0)
    np.testing.assert_allclose( aS2[:,2], np.zeros(3))
    np.testing.assert_allclose( aNS2[1,1],0)



def test_A_2_shape():

    N = 2
    L = 3
    sx = np.zeros(3, np.complex_)
    aNS2 = A_ns_2(N, sx, L)
    aS2 = A_singlet_2(N, sx, L)

    assert aNS2.shape == (2, 2)
    assert aS2.shape == (3, 3)

    # check q line equal to the h line for non intrinsic
    assert aS2[1].all() == aS2[2].all()
    assert aNS2[0].all() == aNS2[1].all()


def test_pegasus_sign():

    # reference value come from Mathematica
    ref_val = - 21133.9
    N = 2
    sx = get_sx(N)
    L = 100.
    aS2 = A_singlet_2(N, sx, L)

    np.testing.assert_allclose(aS2[0,0], ref_val, rtol=4e-5)
