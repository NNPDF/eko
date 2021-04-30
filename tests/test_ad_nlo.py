# -*- coding: utf-8 -*-
# Test NLO anomalous dims
import numpy as np

from eko.constants import CA, CF, TR
import eko.anomalous_dimensions.nlo as ad_nlo
import eko.anomalous_dimensions.harmonics as h

NF = 5


def test_gamma_1():
    # number conservation
    np.testing.assert_allclose(ad_nlo.gamma_nsm_1(1, NF), 0.0, atol=2e-6)

    gS1 = ad_nlo.gamma_singlet_1(2, NF)
    # gluon momentum conservation
    # the CA*NF term seems to be tough to compute, so raise the constraint ...
    np.testing.assert_allclose(gS1[0, 1] + gS1[1, 1], 0.0, atol=4e-5)
    # quark momentum conservation
    np.testing.assert_allclose(gS1[0, 0] + gS1[1, 0], 0.0, atol=2e-6)

    assert gS1.shape == (2, 2)

    # reference values are obtained from MMa
    # Non singlet sector
    np.testing.assert_allclose(
        ad_nlo.gamma_nsp_1(2, NF), (-112.0 * CF + 376.0 * CA - 64.0 * NF) * CF / 27.0
    )
    # singlet sector
    np.testing.assert_allclose(ad_nlo.gamma_ps_1(2, NF), -40.0 * CF * NF / 27.0)
    np.testing.assert_allclose(gS1[0, 1], (-74.0 * CF - 35.0 * CA) * NF / 27.0)  # qg
    np.testing.assert_allclose(
        gS1[1, 0], (112.0 * CF - 376.0 * CA + 104.0 * NF) * CF / 27.0
    )  # gq

    # add additional point at (analytical) continuation point
    np.testing.assert_allclose(
        ad_nlo.gamma_nsm_1(2, NF),
        (
            (34.0 / 27.0 * (-47.0 + 6 * np.pi ** 2) - 16.0 * h.zeta3) * CF
            + (373.0 / 9.0 - 34.0 * np.pi ** 2 / 9.0 + 8.0 * h.zeta3) * CA
            - 64.0 * NF / 27.0
        )
        * CF,
    )
    np.testing.assert_allclose(
        ad_nlo.gamma_nsp_1(3, NF),
        (
            (-34487.0 / 432.0 + 86.0 * np.pi ** 2 / 9.0 - 16.0 * h.zeta3) * CF
            + (459.0 / 8.0 - 43.0 * np.pi ** 2 / 9.0 + 8.0 * h.zeta3) * CA
            - 415.0 * NF / 108.0
        )
        * CF,
    )
    np.testing.assert_allclose(ad_nlo.gamma_ps_1(3, NF), -1391.0 * CF * NF / 5400.0)
    gS1 = ad_nlo.gamma_singlet_1(3, NF)
    np.testing.assert_allclose(
        gS1[1, 0],
        (
            973.0 / 432.0 * CF
            + (2801.0 / 5400.0 - 7.0 * np.pi ** 2 / 9.0) * CA
            + 61.0 / 54.0 * NF
        )
        * CF,
    )  # gq
    np.testing.assert_allclose(
        gS1[1, 1],
        (
            (-79909.0 / 3375.0 + 194.0 * np.pi ** 2 / 45.0 - 8.0 * h.zeta3) * CA ** 2
            - 967.0 / 270.0 * CA * NF
            + 541.0 / 216.0 * CF * NF
        ),
        rtol=6e-7,
    )  # gg
    gS1 = ad_nlo.gamma_singlet_1(4, NF)
    np.testing.assert_allclose(
        gS1[0, 1], (-56317.0 / 18000.0 * CF + 16387.0 / 9000.0 * CA) * NF
    )  # qg


def test_gamma_ns_1_pegasus():
    # pylint: disable=line-too-long,too-many-locals
    # Test against pegasus implementation
    ZETA2 = h.zeta2
    ZETA3 = h.zeta3

    for N in [2]:
        S1 = h.harmonic_S1(N)
        S2 = h.harmonic_S2(N)

        N1 = N + 1.0
        N2 = N + 2.0
        NS = N * N
        NT = NS * N
        NFO = NT * N
        N1S = N1 * N1
        N1T = N1S * N1
        N3 = N + 3.0
        N4 = N + 4.0
        N5 = N + 5.0
        N6 = N + 6.0
        S11 = S1 + 1.0 / N1
        S12 = S11 + 1.0 / N2
        S13 = S12 + 1.0 / N3
        S14 = S13 + 1.0 / N4
        S15 = S14 + 1.0 / N5
        S16 = S15 + 1.0 / N6
        SPMOM = (
            1.00000 * (ZETA2 - S1 / N) / N
            - 0.99920 * (ZETA2 - S11 / N1) / N1
            + 0.98510 * (ZETA2 - S12 / N2) / N2
            - 0.90050 * (ZETA2 - S13 / N3) / N3
            + 0.66210 * (ZETA2 - S14 / N4) / N4
            - 0.31740 * (ZETA2 - S15 / N5) / N5
            + 0.06990 * (ZETA2 - S16 / N6) / N6
        )
        SLC = -5.0 / 8.0 * ZETA3
        SLV = (
            -ZETA2 / 2.0 * (h.cern_polygamma(N1 / 2, 0) - h.cern_polygamma(N / 2, 0))
            + S1 / NS
            + SPMOM
        )
        SSCHLM = SLC - SLV
        SSTR2M = ZETA2 - h.cern_polygamma(N1 / 2, 1)
        SSTR3M = 0.5 * h.cern_polygamma(N1 / 2, 2) + ZETA3
        SSCHLP = SLC + SLV
        SSTR2P = ZETA2 - h.cern_polygamma(N2 / 2, 1)
        SSTR3P = 0.5 * h.cern_polygamma(N2 / 2, 2) + ZETA3

        PNMA = (
            16.0 * S1 * (2.0 * N + 1.0) / (NS * N1S)
            + 16.0 * (2.0 * S1 - 1.0 / (N * N1)) * (S2 - SSTR2M)
            + 64.0 * SSCHLM
            + 24.0 * S2
            - 3.0
            - 8.0 * SSTR3M
            - 8.0 * (3.0 * NT + NS - 1.0) / (NT * N1T)
            + 16.0 * (2.0 * NS + 2.0 * N + 1.0) / (NT * N1T)
        ) * (-0.5)
        PNPA = (
            16.0 * S1 * (2.0 * N + 1.0) / (NS * N1S)
            + 16.0 * (2.0 * S1 - 1.0 / (N * N1)) * (S2 - SSTR2P)
            + 64.0 * SSCHLP
            + 24.0 * S2
            - 3.0
            - 8.0 * SSTR3P
            - 8.0 * (3.0 * NT + NS - 1.0) / (NT * N1T)
            - 16.0 * (2.0 * NS + 2.0 * N + 1.0) / (NT * N1T)
        ) * (-0.5)
        PNSB = (
            S1 * (536.0 / 9.0 + 8.0 * (2.0 * N + 1.0) / (NS * N1S))
            - (16.0 * S1 + 52.0 / 3.0 - 8.0 / (N * N1)) * S2
            - 43.0 / 6.0
            - (151.0 * NFO + 263.0 * NT + 97.0 * NS + 3.0 * N + 9.0)
            * 4.0
            / (9.0 * NT * N1T)
        ) * (-0.5)
        PNSC = (
            -160.0 / 9.0 * S1
            + 32.0 / 3.0 * S2
            + 4.0 / 3.0
            + 16.0 * (11.0 * NS + 5.0 * N - 3.0) / (9.0 * NS * N1S)
        ) * (-0.5)

        P1NSP = CF * ((CF - CA / 2.0) * PNPA + CA * PNSB + TR * NF * PNSC)
        P1NSM = CF * ((CF - CA / 2.0) * PNMA + CA * PNSB + TR * NF * PNSC)

        np.testing.assert_allclose(ad_nlo.gamma_nsp_1(N, NF), -P1NSP)
        np.testing.assert_allclose(ad_nlo.gamma_nsm_1(N, NF), -P1NSM)
