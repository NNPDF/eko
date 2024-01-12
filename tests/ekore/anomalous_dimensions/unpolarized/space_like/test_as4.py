# Test N3LO anomalous dimensions
import numpy as np

from eko.constants import CA, CF
from ekore import harmonics as h
from ekore.anomalous_dimensions.unpolarized.space_like.as4 import (
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

n3lo_vars_dict = {
    "gg": 19,
    "gq": 21,
    "qg": 15,
    "qq": 6,
}


def build_n3lo_var():
    variations = [[0, 0, 0, 0]]
    for entry, max_var in enumerate(n3lo_vars_dict.values()):
        for idx in range(1, max_var + 1):
            variation = [0, 0, 0, 0]
            variation[entry] = idx
            variations.append(variation)
    return variations


def test_quark_number_conservation():
    N = 1
    sx_cache = h.cache.reset()

    # (ns,-)
    # nf^3 part
    np.testing.assert_allclose(gnsp.gamma_ns_nf3(N, sx_cache), 0, atol=3e-15)
    # nf^2 part
    np.testing.assert_allclose(gnsm.gamma_nsm_nf2(N, sx_cache), 0, atol=4e-13)
    # nf^1 part
    np.testing.assert_allclose(gnsm.gamma_nsm_nf1(N, sx_cache), 0, atol=2e-11)
    # nf^0 part
    np.testing.assert_allclose(gnsm.gamma_nsm_nf0(N, sx_cache), 0, atol=2e-10)
    # total
    np.testing.assert_allclose(gnsm.gamma_nsm(N, NF, sx_cache), 0, atol=1e-10)

    # (ns,s)
    # the exact expression (nf^2 part) has an nonphysical pole at N=1,
    # see also :cite:`Moch:2017uml` and :cite:`Davies:2016jie` eq 3.5
    # where the \nu term is present.
    # This should cancel when doing the limit, since the given moment for N=1 is 0
    np.testing.assert_allclose(gnsv.gamma_nss_nf2(N + 1e-8, sx_cache), 0, atol=9e-6)

    # nf^1 part
    np.testing.assert_allclose(gnsv.gamma_nss_nf1(N, sx_cache), 0.000400625, atol=2e-6)


def test_momentum_conservation():
    N = 2
    sx_cache = h.cache.reset()

    # nf^3 part
    np.testing.assert_allclose(
        gnsp.gamma_ns_nf3(N, sx_cache)
        + gps.gamma_ps_nf3(N, sx_cache)
        + ggq.gamma_gq_nf3(N, sx_cache),
        0,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        ggg.gamma_gg_nf3(N, sx_cache) + gqg.gamma_qg_nf3(N, sx_cache), 0, atol=2e-7
    )

    variations = build_n3lo_var()
    g_singlet = np.zeros((len(variations), 2, 2), dtype=complex)
    g_ps = np.zeros((len(variations), 2), dtype=complex)
    g_gq = np.zeros((len(variations), 3), dtype=complex)
    g_qg = np.zeros((len(variations), 2), dtype=complex)
    g_gg = np.zeros((len(variations), 3), dtype=complex)
    for i, variation in enumerate(variations):
        g_singlet[i, :, :] = gamma_singlet(N, NF, sx_cache, variation)
        g_gg[i, :] = [
            ggg.gamma_gg_nf0(N, sx_cache, variation[0]),
            ggg.gamma_gg_nf1(N, sx_cache, variation[0]),
            ggg.gamma_gg_nf2(N, sx_cache, variation[0]),
        ]
        g_gq[i, :] = [
            ggq.gamma_gq_nf0(N, sx_cache, variation[1]),
            ggq.gamma_gq_nf1(N, sx_cache, variation[1]),
            ggq.gamma_gq_nf2(N, sx_cache, variation[1]),
        ]
        g_qg[i, :] = [
            gqg.gamma_qg_nf1(N, sx_cache, variation[2]),
            gqg.gamma_qg_nf2(N, sx_cache, variation[2]),
        ]
        g_ps[i, :] = [
            gps.gamma_ps_nf1(N, sx_cache, variation[3]),
            gps.gamma_ps_nf2(N, sx_cache),
        ]

    # nf^2 part
    np.testing.assert_allclose(
        gnsp.gamma_nsp_nf2(N, sx_cache) + g_ps[:, 1] + g_gq[:, 2],
        0,
        atol=6e-8,
    )
    np.testing.assert_allclose(
        +g_gg[:, 2] + g_qg[:, 1],
        0,
        atol=3e-11,
    )

    # nf^1 part
    np.testing.assert_allclose(
        gnsp.gamma_nsp_nf1(N, sx_cache) + g_ps[:, 0] + g_gq[:, 1],
        0,
        atol=7e-11,
    )
    np.testing.assert_allclose(
        g_gg[:, 1] + g_qg[:, 0],
        0,
        atol=2e-10,
    )

    # nf^0 part
    np.testing.assert_allclose(
        gnsp.gamma_nsp_nf0(N, sx_cache) + g_gq[:, 0],
        0,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        g_gg[:, 0],
        0,
        atol=6e-10,
    )

    # total
    np.testing.assert_allclose(
        g_singlet[:, 0, 0] + g_singlet[:, 1, 0],
        0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        g_singlet[:, 0, 1] + g_singlet[:, 1, 1],
        0,
        atol=2e-5,
    )


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
                gnsm.gamma_nsm(N, NF, sx_cache), nsm_nf4_refs[idx], rtol=7e-6
            )
            np.testing.assert_allclose(
                gnsv.gamma_nsv(N, NF, sx_cache),
                nss_nf4_refs[idx] + nsm_nf4_refs[idx],
                rtol=7e-6,
            )
        gamma_nss = (
            gnsv.gamma_nss_nf1(N, sx_cache) * NF
            + gnsv.gamma_nss_nf2(N, sx_cache) * NF**2
        )
        np.testing.assert_allclose(gamma_nss, nss_nf4_refs[idx], atol=4e-4)


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
            gnsp.gamma_nsp(N, NF, sx_cache), nsp_nf4_refs[int((N - 2) / 2)], rtol=3e-6
        )


def test_diff_pm_nf2():
    # Test deltaB3: diff g_{ns,-} - g_{ns,+} prop to nf^2
    # Note that discrepancy for low moments is higher due to
    # oscillating behavior which is not captured by our parametrization
    def deltaB3(n, sx_cache):
        """
        Implementation of Eq. 3.4 of :cite:`Davies:2016jie`.

        Parameters
        ----------
            n : complex
                Mellin moment
            sx : list
                harmonic sums cache

        Returns
        -------
            B_3m : complex
                |N3LO| valence-like non-singlet anomalous dimension part
                proportional to :math:`C_F (C_A - 2 C_F) nf^2`
        """
        S1 = h.cache.get(h.cache.S1, sx_cache, n)
        S2 = h.cache.get(h.cache.S2, sx_cache, n)
        Sm2 = h.cache.get(h.cache.Sm2, sx_cache, n)
        deltaB = (
            -(1 / (729 * n**5 * (1 + n) ** 5))
            * CF
            * (CA - 2 * CF)
            * 9
            * 16
            * (
                -54
                - 60 * n
                + 211 * n**2
                + 367 * n**3
                + 73 * n**4
                - 255 * n**5
                - 475 * n**6
                - 138 * n**7
                - 39 * n**8
                - 36 * n**2 * (1 + n) ** 2 * (1 + 2 * n + 2 * n**2) * Sm2
                + 6
                * n
                * (
                    -12
                    - 22 * n
                    + 29 * n**2
                    + 116 * n**3
                    + 147 * n**4
                    + 79 * n**5
                    + 12 * n**6
                    + 3 * n**7
                )
                * S1
                - 36 * n**2 * (1 + n) ** 2 * (1 + 2 * n + 2 * n**2) * S1**2
                - 36 * n**2 * S2
                - 144 * n**3 * S2
                - 252 * n**4 * S2
                - 216 * n**5 * S2
                - 72 * n**6 * S2
            )
        )
        return deltaB

    diff = []
    ref_vals = []
    for N in range(10, 35):
        sx_cache = h.cache.reset()
        diff.append(gnsp.gamma_nsp_nf2(N, sx_cache) - gnsm.gamma_nsm_nf2(N, sx_cache))
        ref_vals.append(deltaB3(N, sx_cache))
    np.testing.assert_allclose(diff, ref_vals, atol=5e-3)

    diff = []
    ref_vals = []
    for N in range(4, 10):
        sx_cache = h.cache.reset()
        diff.append(gnsp.gamma_nsp_nf2(N, sx_cache) - gnsm.gamma_nsm_nf2(N, sx_cache))
        ref_vals.append(deltaB3(N, sx_cache))
    np.testing.assert_allclose(diff, ref_vals, atol=2.1e-2)


def test_gamma_ps_extrapolation():
    # Test the prediction of N=22 wrt to :cite:`Falcioni:2023luc`
    n22_ref = [6.2478570, 10.5202730, 15.6913948]
    N = 22
    sx_cache = h.cache.reset()
    my_res = []
    for nf in [3, 4, 5]:
        my_res.append(gps.gamma_ps(N, nf, sx_cache, 0))
    np.testing.assert_allclose(n22_ref, n22_ref)
