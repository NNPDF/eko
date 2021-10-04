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


def test_A_3():
    nf = 3
    N = 1.0
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

    N = 2.0
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
        rtol=32e-3,
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


def test_Bluemlein_3():
    # Test against Blumlein OME implementation :cite:`Bierenbaum_2009`.
    # For singlet OME only even moments are available in that code.
    # Note there is a minus sign in the definition of L.

    # pylint: disable=too-many-locals
    ref_val_gg = {
        0: [-440.104, -1377.49, -1683.31, -2006.18, -3293.81],
    }
    # Mathematica not able to evaluate for N=100
    ref_val_ggTF2 = {
        0: [-33.4281, -187.903, -239.019, -294.571],
    }
    # diverging for N=2
    ref_val_gq = {
        0: [0, 22.7356, 16.4025, 10.5142, 0.98988],
    }
    ref_val_Hg = {
        0: [461.219, 682.728, 676.549, 626.857, 294.313],
    }
    ref_val_Hgstfac = {
        0: [109.766, 64.7224, 25.1745, -11.5071, -37.9846],
    }
    ref_val_Hq = {
        0: [15.6809, 1.82795, 1.01716, 0.595205, 0.0065763],
    }
    ref_val_qg = {
        0: [47.695, 44.8523, 32.6934, 19.8899, 0.397559],
    }
    ref_val_qqNS = {
        0: [-37.0244, -40.1562, -36.0358, -28.3506, 6.83759],
    }
    ref_val_qqPS = {
        0: [-8.65731, -0.766936, -0.0365199, 0.147675, 0.0155598],
    }
    nf = 3
    for i, N in enumerate([4.0, 6.0, 10.0, 100.0]):
        idx = i + 1
        for L in ref_val_Hg:
            sx_all = get_sx(N)
            sx_all = np.append(sx_all, get_smx(N))
            sx_all = np.append(sx_all, get_s3x(N, get_sx(N), get_smx(N)))
            sx_all = np.append(sx_all, get_s4x(N, get_sx(N), get_smx(N)))
            aS3 = A_singlet_3(N, sx_all, nf)

            # here we have a different approximation for AggTF2,
            # some terms are neglected
            if N != 100:
                np.testing.assert_allclose(
                    aS3[0, 0], ref_val_gg[L][idx] + ref_val_ggTF2[L][idx], rtol=6e-3
                )

            np.testing.assert_allclose(aS3[0, 1], ref_val_gq[L][idx], rtol=2e-6)
            np.testing.assert_allclose(aS3[1, 0], ref_val_qg[L][idx], rtol=2e-6)
            np.testing.assert_allclose(
                aS3[2, 0], ref_val_Hg[L][idx] + ref_val_Hgstfac[L][idx], rtol=2e-6
            )

            np.testing.assert_allclose(
                aS3[2, 1], ref_val_Hq[L][idx], rtol=5e-3, atol=3e-4
            )

            # here we have a different convention for (-1)^N,
            # for even values qqNS is analitically continued
            # as non singlet. The accuracy is worst for large N
            # due to the approximations of F functions.
            np.testing.assert_allclose(
                aS3[1, 1], ref_val_qqNS[L][idx] + ref_val_qqPS[L][idx], rtol=3e-2
            )

    # Here we test the critical parts
    nf = 3
    ref_ggTF_app = [-28.9075, -180.659, -229.537, -281.337, -467.164]
    for idx, N in enumerate([2.0, 4.0, 6.0, 10.0, 100.0]):
        sx = get_sx(N)
        smx = get_smx(N)
        s3x = get_s3x(N, sx, smx)
        s4x = get_s4x(N, sx, smx)
        Aggtf2 = n3lo.aggTF2.A_ggTF2_3(N, sx, s3x)
        if N != 100:
            # Limited in the small N region
            np.testing.assert_allclose(Aggtf2, ref_val_ggTF2[0][idx], rtol=15e-2)
        np.testing.assert_allclose(Aggtf2, ref_ggTF_app[idx], rtol=3e-6)

        np.testing.assert_allclose(
            n3lo.agg.A_gg_3(N, sx, smx, s3x, s4x, nf) - Aggtf2,
            ref_val_gg[0][idx],
            rtol=3e-6,
        )

    # odd numbers of qqNS
    # Limited accuracy due to F functions
    ref_qqNS_odd = [-40.95, -21.5988, 6.96633]
    for N, ref in zip([3.0, 15.0, 101.0], ref_qqNS_odd):
        sx = get_sx(N)
        smx = get_smx(N)
        s3x = get_s3x(N, sx, smx)
        s4x = get_s4x(N, sx, smx)
        np.testing.assert_allclose(
            n3lo.aqqNS.A_qqNS_3(N, sx, smx, s3x, s4x, nf), ref, rtol=3e-2
        )


def test_AHq_asymptotic():
    refs = [
        -1.06712,
        0.476901,
        -0.771605,
        0.388789,
        0.228768,
        0.114067,
        0.0654939,
        0.0409271,
        0.0270083,
        0.01848,
        0.0129479,
        0.00920106,
    ]
    Ns = [11.0, 12.0, 13.0, 14.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    # refs = [
    # -0.159229, 0.101182, -0.143408, 0.0901927,
    #  -0.130103, 0.0807537, -0.11879, 0.0725926, -0.109075
    # ]
    # Ns = [31.,32.,33.,34.,35.,36.,37.,38.,39.]
    nf = 3
    for N, r in zip(Ns, refs):
        sx = get_sx(N)
        smx = get_smx(N)
        s3x = get_s3x(N, sx, smx)
        s4x = get_s4x(N, sx, smx)
        np.testing.assert_allclose(
            n3lo.aHq.A_Hq_3(N, sx, smx, s3x, s4x, nf), r, rtol=5e-3, atol=5e-4
        )
