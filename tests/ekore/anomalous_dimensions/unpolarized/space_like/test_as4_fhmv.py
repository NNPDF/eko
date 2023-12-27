# Test N3LO anomalous dimensions
import numpy as np
import pytest

from ekore import harmonics as h
from ekore.anomalous_dimensions.unpolarized.space_like.as4.fhmruvv_approximations import (
    gamma_singlet,
    ggg,
    ggq,
    gnsm,
    gnsp,
    gnsv,
    gps,
    gqg,
)

NF = 5


def test_quark_number_conservation():
    N = 1
    sx_cache = h.cache.reset()
    for imod in [0, 1, 2]:
        # (ns,s)
        refs = [-0.01100459, -0.007799, -0.01421]
        np.testing.assert_allclose(
            gnsv.gamma_nss(N, NF, sx_cache, imod), refs[imod], rtol=6e-05
        )

        # (ns,m)
        refs = [0.06776363, 0.064837, 0.07069]
        np.testing.assert_allclose(
            gnsm.gamma_nsm(N, NF, sx_cache, imod), refs[imod], rtol=6e-06
        )


def test_momentum_conservation():
    N = 2
    sx_cache = h.cache.reset()

    g_singlet = np.zeros((3, 2, 2), dtype=complex)
    for imod in [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]:
        g_singlet[imod, :, :] = gamma_singlet(N, NF, sx_cache, imod)

    # total
    np.testing.assert_allclose(
        g_singlet[:, 0, 0] + g_singlet[:, 1, 0],
        [0.08617, 0.220242, -0.047901],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        g_singlet[:, 0, 1] + g_singlet[:, 1, 1],
        [-0.134766, 0.465174, -0.734706],
        atol=2e-5,
    )


def test_vogt_parametriztions():
    def qg3_moment(N, nf):
        mom_list = [
            -654.4627782205557 * nf
            + 245.61061978871788 * nf**2
            - 0.9249909688301847 * nf**3,
            290.31106867034487 * nf
            - 76.51672403736478 * nf**2
            - 4.911625629947491 * nf**3,
            335.80080466045274 * nf
            - 124.57102255718002 * nf**2
            - 4.193871425027802 * nf**3,
            294.58768309440677 * nf
            - 135.3767647714609 * nf**2
            - 3.609775642729055 * nf**3,
            241.6153399044715 * nf
            - 135.18742470907011 * nf**2
            - 3.189394834180898 * nf**3,
            191.97124640777176 * nf
            - 131.16316638326697 * nf**2
            - 2.8771044305171913 * nf**3,
            148.5682948286098 * nf
            - 125.82310814280595 * nf**2
            - 2.635918561148907 * nf**3,
            111.34042526856348 * nf
            - 120.16819876888667 * nf**2
            - 2.4433790398202664 * nf**3,
            79.51561588665083 * nf
            - 114.61713540075442 * nf**2
            - 2.28548686108789 * nf**3,
            52.24329555231736 * nf
            - 109.34248910828198 * nf**2
            - 2.1531537251387527 * nf**3,
        ]
        return mom_list[int((N - 2) / 2)]

    def qq3ps_moment(N, nf):
        mom_list = [
            -691.5937093082381 * nf
            + 84.77398149891167 * nf**2
            + 4.4669568492355864 * nf**3,
            -109.33023358432462 * nf
            + 8.77688525974872 * nf**2
            + 0.3060771365698822 * nf**3,
            -46.030613749542226 * nf
            + 4.744075766957513 * nf**2
            + 0.042548957282380874 * nf**3,
            -24.01455020567638 * nf
            + 3.235193483272451 * nf**2
            - 0.007889256298951614 * nf**3,
            -13.730393879922417 * nf
            + 2.3750187592472374 * nf**2
            - 0.02102924056123573 * nf**3,
            -8.152592251923657 * nf
            + 1.8199581788320662 * nf**2
            - 0.024330231290833188 * nf**3,
            -4.8404471801109565 * nf
            + 1.4383273806219803 * nf**2
            - 0.024479943136069916 * nf**3,
            -2.7511363301137024 * nf
            + 1.164299642517469 * nf**2
            - 0.023546009234463816 * nf**3,
            -1.375969240387974 * nf
            + 0.9608733183576097 * nf**2
            - 0.022264393374041958 * nf**3,
            -0.4426815682220422 * nf
            + 0.8057453328332964 * nf**2
            - 0.02091826436475512 * nf**3,
        ]
        return mom_list[int((N - 2) / 2)]

    def gg3_moment(N, nf):
        mom_list = [
            654.4627782205557 * nf
            - 245.6106197887179 * nf**2
            + 0.9249909688301847 * nf**3,
            39876.123276008046
            - 10103.4511350227 * nf
            + 437.0988475397789 * nf**2
            + 12.955565459350593 * nf**3,
            53563.84353419538
            - 14339.131035160317 * nf
            + 652.7773306808972 * nf**2
            + 16.654103652963503 * nf**3,
            62279.7437813437
            - 17150.696783851945 * nf
            + 785.8806126875509 * nf**2
            + 18.933103109772713 * nf**3,
        ]
        return mom_list[int((N - 2) / 2)]

    def gq3_moment(N, nf):
        mom_list = [
            -16663.225488
            + 4439.143749608238 * nf
            - 202.55547919891168 * nf**2
            - 6.375390720235586 * nf**3,
            -6565.7531450230645
            + 1291.0067460871576 * nf
            - 16.146190170051486 * nf**2
            - 0.8397634037808341 * nf**3,
            -3937.479370556893
            + 679.7185057363981 * nf
            - 1.3720775271604673 * nf**2
            - 0.13979432728276966 * nf**3,
            -2803.644107251366
            + 436.39305738710254 * nf
            + 1.8149462465491055 * nf**2
            + 0.07358858022119033 * nf**3,
        ]
        return mom_list[int((N - 2) / 2)]

    for N in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        cache = h.cache.reset()
        for variation in [0, 1, 2]:
            for nf in [3, 4, 5]:
                np.testing.assert_allclose(
                    gps.gamma_ps(N, nf, cache, variation),
                    qq3ps_moment(N, nf),
                    rtol=4e-4,
                )
                np.testing.assert_allclose(
                    gqg.gamma_qg(N, nf, cache, variation),
                    qg3_moment(N, nf),
                    rtol=8e-4,
                )
                if N <= 8:
                    np.testing.assert_allclose(
                        ggg.gamma_gg(N, nf, cache, variation),
                        gg3_moment(N, nf),
                        rtol=4e-4,
                    )
                    np.testing.assert_allclose(
                        ggq.gamma_gq(N, nf, cache, variation),
                        gq3_moment(N, nf),
                        rtol=2e-4,
                    )

    with pytest.raises(NotImplementedError):
        gps.gamma_ps(N, 6, cache, variation)
        gqg.gamma_qg(N, 6, cache, variation)
        ggg.gamma_gg(N, 6, cache, variation)
        ggq.gamma_gq(N, 6, cache, variation)


def test_non_singlet_reference_moments():
    NF = 4
    nsm_nf4_refs = [
        4322.890485339998,
        5491.581109692005,
        6221.256799360004,
        6774.606221595994,
        7229.056043916002,
        7618.358743427995,
        7960.658678124,
    ]
    nss_nf4_refs = [
        50.10532524,
        39.001939964,
        21.141505811200002,
        12.4834195012,
        8.0006134908,
        5.4610639744,
        3.9114290952,
        2.90857799,
    ]
    for N in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]:
        sx_cache = h.cache.reset()
        idx = int((N - 3) / 2)
        if N != 17:
            np.testing.assert_allclose(
                gnsm.gamma_nsm(N, NF, sx_cache, variation=0),
                nsm_nf4_refs[idx],
                rtol=8e-5,
            )
            np.testing.assert_allclose(
                gnsv.gamma_nsv(N, NF, sx_cache, variation=0),
                nss_nf4_refs[idx] + nsm_nf4_refs[idx],
                rtol=8e-5,
            )
        np.testing.assert_allclose(
            gnsv.gamma_nss(N, NF, sx_cache, variation=0), nss_nf4_refs[idx], rtol=5e-5
        )


def test_singlet_reference_moments():
    NF = 4
    nsp_nf4_refs = [
        3679.6690577439995,
        5066.339235808004,
        5908.005605364002,
        6522.700744595994,
        7016.383458928004,
        7433.340927783997,
        7796.397038483998,
        8119.044600816003,
    ]
    for N in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]:
        sx_cache = h.cache.reset()
        np.testing.assert_allclose(
            gnsp.gamma_nsp(N, NF, sx_cache, variation=0),
            nsp_nf4_refs[int((N - 2) / 2)],
            rtol=4e-5,
        )


def test_gamma_ps_extrapolation():
    # Test the prediction of N=22 wrt to :cite:`Falcioni:2023luc`
    n22_ref = [6.2478570, 10.5202730, 15.6913948]
    N = 22
    sx_cache = h.cache.reset()
    my_res = []
    for nf in [3, 4, 5]:
        my_res.append(gps.gamma_ps(N, nf, sx_cache, 0))
    np.testing.assert_allclose(n22_ref, n22_ref)
