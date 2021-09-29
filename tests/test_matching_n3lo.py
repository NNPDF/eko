# -*- coding: utf-8 -*-
# Test NNLO OME

import numpy as np

from eko.anomalous_dimensions import harmonics
from eko.matching_conditions.n3lo import A_ns_3, A_singlet_3, A_qqNS_3, s_functions
from eko.matching_conditions import n3lo


def get_sx(N):
    """Collect the S-cache"""
    sx = np.array(
        [
            harmonics.harmonic_S1(N),
            harmonics.harmonic_S2(N),
            harmonics.harmonic_S3(N),
            harmonics.harmonic_S4(N),
            harmonics.harmonic_S5(N),
        ]
    )
    return sx


def get_smx(N):
    """Collect the Sminus-cache"""
    smx = np.array(
        [
            s_functions.harmonic_Sm1(N),
            s_functions.harmonic_Sm2(N),
            s_functions.harmonic_Sm3(N),
            s_functions.harmonic_Sm4(N),
            s_functions.harmonic_Sm5(N),
        ]
    )
    return smx


def get_s3x(N, sx, smx):
    """Collect the Sminus-cache"""
    s3x = np.array(
        [
            s_functions.harmonic_S21(N, sx[0], sx[1]),
            s_functions.harmonic_S2m1(N, sx[1], smx[0], smx[1]),
            s_functions.harmonic_Sm21(N, smx[0]),
            s_functions.harmonic_Sm2m1(N, sx[0], sx[1], smx[1]),
        ]
    )
    return s3x


def get_s4x(N, sx, smx):
    """Collect the Sminus-cache"""
    Sm31 = s_functions.harmonic_Sm31(N, smx[0], smx[1])
    s4x = np.array(
        [
            s_functions.harmonic_S31(N, sx[1], sx[3]),
            s_functions.harmonic_S211(N, sx[0], sx[1], sx[2]),
            s_functions.harmonic_Sm22(N, Sm31),
            s_functions.harmonic_Sm211(N, smx[0]),
            Sm31,
        ]
    )
    return s4x


def test_A_2():
    nf = 3
    N = 1
    sx = get_sx(N)
    smx = get_smx(N)
    s3x = get_s3x(N, sx, smx)
    s4x = get_s4x(N, sx, smx)

    aNSqq3 = A_qqNS_3(N, sx, smx, s3x, s4x, nf)
    # quark number conservation
    # the accuracy of this test depends directly on the precision of the
    # F functions, thus is dominated by F19,F20,F21 accuracy are the worst ones
    # If needed, these Fs can be improved.
    np.testing.assert_allclose(aNSqq3, 0.0, atol=5e-3)

    N = complex(2.0)
    sx = get_sx(N)
    smx = get_smx(N)
    s3x = get_s3x(N, sx, smx)
    s4x = get_s4x(N, sx, smx)
    # reference value comes form Mathemtica, gg is not fullycomplete
    # thus the reference value is not 0.0
    # Here the acuracy of this test depends on the approximation of AggTF2
    np.testing.assert_allclose(
        n3lo.A_gg_3(N, sx, smx, s3x, s4x, nf)
        + n3lo.A_qg_3(N, sx, smx, s3x, s4x, nf)
        + n3lo.A_Hg_3(N, sx, smx, s3x, s4x, nf),
        145.148,
        rtol=4e-2,
    )

    # here you get division by 0 as in Mathematica
    # np.testing.assert_allclose(
    #     n3lo.A_gq_3(N, sx, smx, s3x, s4x, nf)
    #     + n3lo.A_qqNS_3(N, sx, smx, s3x, s4x, nf)
    #     + n3lo.A_qqPS_3(N, sx, nf)
    #     + n3lo.A_Hq_3(N, sx, smx, s3x, s4x, nf),
    #     0.0,
    #     atol=2e-6,
    # )

    # here you get division by 0 as in Mathematica
    # sx_all = get_sx(N)
    # sx_all = np.append(sx_all, get_smx(N))
    # sx_all = np.append(sx_all, get_s3x(N, get_sx(N),get_smx(N)))
    # sx_all = np.append(sx_all, get_s4x(N, get_sx(N),get_smx(N)))
    # aS3 = A_singlet_3(N, sx_all, nf)
    # gluon momentum conservation
    # np.testing.assert_allclose(aS3[0, 0] + aS3[1, 0] + aS3[2, 0], 0.0, atol=2e-6)
    # quark momentum conservation
    # np.testing.assert_allclose(aS3[0, 1] + aS3[1, 1] + aS3[2, 1], 0.0, atol=1e-11)

    N = 3 + 2j
    sx_all = np.random.rand(19) + 1j * np.random.rand(19)
    aS3 = A_singlet_3(N, sx_all, nf)
    aNS3 = A_ns_3(N, sx_all, nf)
    assert aNS3.shape == (2, 2)
    assert aS3.shape == (3, 3)

    np.testing.assert_allclose(aS3[:, 2], np.zeros(3))
    np.testing.assert_allclose(aNS3[1, 1], 0)
