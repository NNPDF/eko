# Test N3LO OME
import numpy as np

from ekore.harmonics import compute_cache
from ekore.operator_matrix_elements.unpolarized.space_like import as3
from ekore.operator_matrix_elements.unpolarized.space_like.as3 import (
    A_ns,
    A_qqNS,
    A_singlet,
)


def test_A_3():
    logs = [0, 10]
    nf = 3

    for L in logs:
        N = 1.0
        sx_cache = compute_cache(N, 5, False)
        aNSqq3 = A_qqNS(N, sx_cache, nf, L, eta=-1)
        # quark number conservation
        # the accuracy of this test depends directly on the precision of the
        # fitted part of aNSqq3
        np.testing.assert_allclose(aNSqq3, 0.0, atol=6e-5)

        N = 2.0
        sx_cache = compute_cache(N, 5, True)
        # The accuracy of this test depends on the approximation of aHg3.
        # which is not fully available.
        atol = 2e-4 if L == 0 else 2e-3
        np.testing.assert_allclose(
            as3.A_gg(N, sx_cache, nf, L)
            + as3.A_qg(N, sx_cache, nf, L)
            + as3.A_Hg(N, sx_cache, nf, L),
            0,
            atol=atol,
        )

        # here you can't test the quark momentum conservation
        # since you get division by 0 as in Mathematica
        # due to a factor 1/(N-2) which should cancel when
        # doing a proper limit.
        # Note the part proportional to the log respect momentum conservation,
        # independently.
        if L == 0:
            eps = 1e-6
            atol = 3e-5
            N = 2.0 + eps
            sx_cache = compute_cache(N, 5, True)
            np.testing.assert_allclose(
                as3.A_gq(N, sx_cache, nf, L)
                + as3.A_qqNS(N, sx_cache, nf, L, 1)
                + as3.A_qqPS(N, sx_cache, nf, L)
                + as3.A_Hq(N, sx_cache, nf, L),
                0,
                atol=atol,
            )

    N = 3 + 2j
    sx_cache = compute_cache(np.random.rand(), 5, True)
    aS3 = A_singlet(N, sx_cache, nf, L)
    aNS3 = A_ns(N, sx_cache, nf, L)
    assert aNS3.shape == (2, 2)
    assert aS3.shape == (3, 3)

    np.testing.assert_allclose(aS3[:, 2], np.zeros(3))
    np.testing.assert_allclose(aNS3[1, 1], 0)
    np.testing.assert_allclose(aNS3[0, 0], as3.A_qqNS(N, sx_cache, nf, L, -1))


def test_Blumlein_3():
    # Test against Blumlein OME implementation :cite:`Bierenbaum:2009mv`.
    # For singlet OME only even moments are available in that code.
    # Note there is a minus sign in the definition of L.

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
    ref_val_a_gg = {
        0: [
            30.3552,
            1278.452724646308,
            1657.6142867299966,
            2043.4730165095527,
            3477.048152567501,
        ],
        10: [
            30.3552,
            1278.452724646308,
            1657.6142867299966,
            2043.4730165095527,
            3477.048152567501,
        ],
    }
    # diverging for N=2
    ref_val_gq = {
        0: [0, 22.7356, 16.4025, 10.5142, 0.98988],
        10: [0, 4408.98, 2488.62, 1281.9650, 53.907],
    }
    ref_val_Hg = {
        0: [
            461.2193170394263,
            682.7275943268593,
            676.5491752546094,
            626.8567254890951,
            294.3132833764657,
        ],
        10: [
            18487.47133439009,
            23433.42707514685,
            20705.71711303489,
            16377.933296134988,
            3386.8649353946803,
        ],
    }
    ref_val_a_Hg = {
        0: [
            -99.16581867356965,
            -676.0759818186247,
            -768.6183629349141,
            -789.7519719852811,
            -414.2873373741821,
        ],
        10: [
            -99.16581867356965,
            -676.0759818186247,
            -768.6183629349141,
            -789.7519719852811,
            -414.2873373741821,
        ],
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
        0: [
            -37.02436747945553,
            -40.156206321394976,
            -36.035766608587835,
            -28.35058671616347,
            6.837590678286162,
        ],
        10: [
            -7574.851254907043,
            -14130.328640707816,
            -17928.587311540465,
            -22767.991224937196,
            -45326.8491411019,
        ],
    }
    ref_val_qqPS = {
        0: [
            -8.657307152008933,
            -0.7669358960417618,
            -0.036519930744266293,
            0.14767512009641925,
            0.01555975620686876,
        ],
        10: [
            1672.9887833829705,
            260.6010476430529,
            112.65066658104867,
            43.5204392378118,
            0.7566205641145347,
        ],
    }
    nf = 3
    for i, N in enumerate([4.0, 6.0, 10.0, 100.0]):
        idx = i + 1
        for L in [0, 10]:
            sx_cache = compute_cache(N, 5, True)
            aS3 = A_singlet(N, sx_cache, nf, L)

            np.testing.assert_allclose(
                aS3[0, 0], ref_val_gg[L][idx] + ref_val_a_gg[L][idx], rtol=3e-6
            )

            np.testing.assert_allclose(aS3[0, 1], ref_val_gq[L][idx], rtol=2e-6)
            np.testing.assert_allclose(aS3[1, 0], ref_val_qg[L][idx], rtol=2e-6)
            np.testing.assert_allclose(
                aS3[2, 0], ref_val_Hg[L][idx] + ref_val_a_Hg[L][idx], rtol=6e-6
            )

            np.testing.assert_allclose(
                aS3[2, 1], ref_val_Hq[L][idx], rtol=2e-5, atol=2e-6
            )
            np.testing.assert_allclose(
                aS3[1, 1], ref_val_qqNS[L][idx] + ref_val_qqPS[L][idx], rtol=2e-6
            )

    # Here we test the critical parts
    for idx, N in enumerate([2.0, 4.0, 6.0, 10.0, 100.0]):
        sx_cache = compute_cache(N, 5, True)
        agg3 = as3.agg.a_gg3(N, sx_cache, nf)
        np.testing.assert_allclose(agg3, ref_val_a_gg[0][idx], rtol=3e-6)
        np.testing.assert_allclose(
            as3.agg.A_gg(N, sx_cache, nf, L=0) - agg3,
            ref_val_gg[0][idx],
            rtol=3e-6,
        )

    # odd numbers of qqNS
    ref_qqNS_odd = [-40.94998646588999, -21.598793547423504, 6.966325573931755]
    for N, ref in zip([3.0, 15.0, 101.0], ref_qqNS_odd):
        sx_cache = compute_cache(N, 5, False)
        np.testing.assert_allclose(
            as3.aqqNS.A_qqNS(N, sx_cache, nf, L=0, eta=-1), ref, rtol=2e-6
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
        sx_cache = compute_cache(N, 5, True)
        np.testing.assert_allclose(
            as3.aHq.A_Hq(N, sx_cache, nf, L=0), r, rtol=7e-6, atol=1e-5
        )
