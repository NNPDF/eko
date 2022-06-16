# -*- coding: utf-8 -*-
# Test N3LO OME
import numpy as np

from eko.matching_conditions import as3
from eko.matching_conditions.as3 import A_ns, A_qqNS, A_singlet
from eko.matching_conditions.operator_matrix_element import compute_harmonics_cache


def test_A_3():
    logs = [0, 10]
    nf = 3

    for L in logs:
        N = 1.0
        sx_cache = compute_harmonics_cache(N, 3, False)
        aNSqq3 = A_qqNS(N, sx_cache, nf, L)
        # quark number conservation
        # the accuracy of this test depends directly on the precision of the
        # fitted part of aNSqq3
        np.testing.assert_allclose(aNSqq3, 0.0, atol=5e-5)

        N = 2.0
        sx_cache = compute_harmonics_cache(N, 3, True)
        # reference value comes form Mathematica, gg is not fullycomplete
        # thus the reference value is not 0.0
        # Here the accuracy of this test depends on the approximation of AggTF2
        np.testing.assert_allclose(
            as3.A_gg(N, sx_cache, nf, L)
            + as3.A_qg(N, sx_cache, nf, L)
            + as3.A_Hg(N, sx_cache, nf, L),
            145.148,
            rtol=32e-3,
        )

    # here you can't test the quark momentum conservation
    # since you get division by 0 as in Mathematica
    # due to a factor 1/(N-2) which should cancel when
    # doing a proper limit.

    N = 3 + 2j
    sx_cache = compute_harmonics_cache(np.random.rand(), 3, True)
    ns_sx_cache = compute_harmonics_cache(np.random.rand(), 3, False)
    aS3 = A_singlet(N, sx_cache, ns_sx_cache, nf, L)
    aNS3 = A_ns(N, ns_sx_cache, nf, L)
    assert aNS3.shape == (2, 2)
    assert aS3.shape == (3, 3)

    np.testing.assert_allclose(aS3[:, 2], np.zeros(3))
    np.testing.assert_allclose(aNS3[1, 1], 0)
    np.testing.assert_allclose(aNS3[0, 0], as3.A_qqNS(N, ns_sx_cache, nf, L))


def test_Blumlein_3():
    # Test against Blumlein OME implementation :cite:`Bierenbaum:2009mv`.
    # For singlet OME only even moments are available in that code.
    # Note there is a minus sign in the definition of L.

    # pylint: disable=too-many-locals
    # reference N are 2,4,6,10,100
    ref_val_gg = {
        0: [
            -440.1036857039252,
            -1377.491100682841,
            -1683.3145211964718,
            -2006.179382743264,
            -3293.80836498399,
        ],
        10: [-18344.3, -41742.6, -50808.5, -61319.1, -108626.0],
    }
    # Mathematica not able to evaluate for N=100
    ref_val_ggTF2 = {
        0: [-33.4281, -187.903, -239.019, -294.571],
        10: [-33.4281, -187.903, -239.019, -294.571],
    }
    # diverging for N=2
    ref_val_gq = {
        0: [0, 22.7356, 16.4025, 10.5142, 0.98988],
        10: [0, 4408.98, 2488.62, 1281.9650, 53.907],
    }
    ref_val_Hg = {
        0: [461.219, 682.728, 676.549, 626.857, 294.313],
        10: [
            18487.47133439009,
            23433.42707514685,
            20705.71711303489,
            16377.933296134988,
            3386.8649353946803,
        ],
    }
    ref_val_Hgstfac = {
        0: [109.766, 64.7224, 25.1745, -11.5071, -37.9846],
        10: [109.766, 64.7224, 25.1745, -11.5071, -37.9846],
    }
    ref_val_Hq = {
        0: [
            15.680876575375834,
            1.827951379087708,
            1.0171574688383518,
            0.5952046986637233,
            0.006576376037629228,
        ],
        10: [
            -9092.772439750246,
            -1952.147452952931,
            -856.1538615259986,
            -314.27582798540243,
            -3.467260112052196,
        ],
    }
    ref_val_qg = {
        0: [47.695, 44.8523, 32.6934, 19.8899, 0.397559],
        10: [-74.4038, -1347.17, -1278.72, -1080.31, -291.084],
    }
    ref_val_qqNS = {
        0: [-36.5531, -40.1257, -36.0358, -28.3555, 6.83735],
        10: [-7562.97, -14129.7, -17928.6, -22768.1, -45326.9],
    }
    ref_val_qqPS = {
        0: [-8.65731, -0.766936, -0.0365199, 0.147675, 0.0155598],
        10: [1672.99, 260.601, 112.651, 43.5204, 0.756621],
    }
    nf = 3
    for i, N in enumerate([4.0, 6.0, 10.0, 100.0]):
        idx = i + 1
        for L in [0, 10]:
            sx_cache = compute_harmonics_cache(N, 3, True)
            ns_sx_cache = compute_harmonics_cache(N, 3, False)
            aS3 = A_singlet(N, sx_cache, ns_sx_cache, nf, L)

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
                aS3[2, 1], ref_val_Hq[L][idx], rtol=2e-5, atol=2e-6
            )

            # here we have a different convention for (-1)^N,
            # for even values qqNS is analytically continued
            # as non-singlet. The accuracy is worst for large N
            # due to the approximations of F functions.
            np.testing.assert_allclose(
                aS3[1, 1], ref_val_qqNS[L][idx] + ref_val_qqPS[L][idx], rtol=8e-4
            )

    # Here we test the critical parts
    nf = 3
    ref_ggTF_app = [-28.9075, -180.659, -229.537, -281.337, -467.164]
    for idx, N in enumerate([2.0, 4.0, 6.0, 10.0, 100.0]):
        sx_cache = compute_harmonics_cache(N, 3, True)
        Aggtf2 = as3.aggTF2.A_ggTF2(N, sx_cache)
        if N != 100:
            # Limited in the small N region
            np.testing.assert_allclose(Aggtf2, ref_val_ggTF2[0][idx], rtol=15e-2)
        np.testing.assert_allclose(Aggtf2, ref_ggTF_app[idx], rtol=2e-4)

        np.testing.assert_allclose(
            as3.agg.A_gg(N, sx_cache, nf, L=0) - Aggtf2,
            ref_val_gg[0][idx],
            rtol=3e-6,
        )

    # odd numbers of qqNS
    ref_qqNS_odd = [-40.94998646588999, -21.598793547423504, 6.966325573931755]
    for N, ref in zip([3.0, 15.0, 101.0], ref_qqNS_odd):
        sx_cache = compute_harmonics_cache(N, 3, False)
        np.testing.assert_allclose(
            as3.aqqNS.A_qqNS(N, sx_cache, nf, L=0), ref, rtol=1e-4
        )


def test_AHq_asymptotic():
    # Odd moments can't be not tested, since in the
    # reference values coming from mathematica, some
    # harmonics still contains some (-1)**N factors which
    # should be continued with 1, but this is not doable.
    refs = [
        # -1.06712,
        0.476901,
        # -0.771605,
        0.388789,
        0.228768,
        0.114067,
        0.0654939,
        0.0409271,
        0.0270083,
        0.01848,
        0.0129479,
        0.00920106,
        0.00657638,
        -0.000672589,
        -0.00106298,
        -0.000560666,
    ]
    Ns = [
        # 11.0,
        12.0,
        # 13.0,
        14.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    ]
    # refs = [
    # -0.159229, 0.101182, -0.143408, 0.0901927,
    #  -0.130103, 0.0807537, -0.11879, 0.0725926, -0.109075
    # ]
    # Ns = [31.,32.,33.,34.,35.,36.,37.,38.,39.]
    nf = 3
    for N, r in zip(Ns, refs):
        sx_cache = compute_harmonics_cache(N, 3, True)
        np.testing.assert_allclose(
            as3.aHq.A_Hq(N, sx_cache, nf, L=0), r, rtol=7e-6, atol=1e-5
        )
