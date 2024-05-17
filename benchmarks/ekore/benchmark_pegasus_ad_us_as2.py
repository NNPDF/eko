"""Benchmark the unpolarized NLO anomalous dimensions against PEGASUS.

Recall that we obtained our representation not from PEGASUS, but derived it
ourselves (see comment there)."""

import numpy as np
import pytest

import ekore.anomalous_dimensions.unpolarized.space_like.as2 as ad_as2
import ekore.harmonics as h
from eko.constants import CA, CF, TR, zeta2, zeta3


@pytest.mark.isolated
def benchmark_gamma_ns_1_pegasus():
    # remember that singlet has pole at N=1
    for N in [2, 3, 4, 1 + 1j, 1 - 1j, 2 + 1j, 3 + 1j]:
        for NF in [3, 4, 5]:
            check_gamma_1_pegasus(N, NF)


def check_gamma_1_pegasus(N, NF):
    # Test against pegasus implementation
    ZETA2 = zeta2
    ZETA3 = zeta3

    S1 = h.S1(N)
    S2 = h.S2(N)

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
        -ZETA2
        / 2.0
        * (h.polygamma.cern_polygamma(N1 / 2, 0) - h.polygamma.cern_polygamma(N / 2, 0))
        + S1 / NS
        + SPMOM
    )
    SSCHLM = SLC - SLV
    SSTR2M = ZETA2 - h.polygamma.cern_polygamma(N1 / 2, 1)
    SSTR3M = 0.5 * h.polygamma.cern_polygamma(N1 / 2, 2) + ZETA3
    SSCHLP = SLC + SLV
    SSTR2P = ZETA2 - h.polygamma.cern_polygamma(N2 / 2, 1)
    SSTR3P = 0.5 * h.polygamma.cern_polygamma(N2 / 2, 2) + ZETA3

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

    cache = h.cache.reset()
    np.testing.assert_allclose(ad_as2.gamma_nsp(N, NF, cache), -P1NSP)
    np.testing.assert_allclose(ad_as2.gamma_nsm(N, NF, cache), -P1NSM)

    NS = N * N
    NT = NS * N
    NFO = NT * N
    NFI = NFO * N
    NSI = NFI * N
    NSE = NSI * N
    NE = NSE * N
    NN = NE * N

    NM = N - 1.0
    N1 = N + 1.0
    N2 = N + 2.0
    NMS = NM * NM
    N1S = N1 * N1
    N1T = N1S * N1
    N2S = N2 * N2
    N2T = N2S * N2

    PPSA = (
        (5.0 * NFI + 32.0 * NFO + 49.0 * NT + 38.0 * NS + 28.0 * N + 8.0)
        / (NM * NT * N1T * N2S)
        * 2.0
    )

    PQGA = (
        (-2.0 * S1 * S1 + 2.0 * S2 - 2.0 * SSTR2P) * (NS + N + 2.0) / (N * N1 * N2)
        + (8.0 * S1 * (2.0 * N + 3.0)) / (N1S * N2S)
        + 2.0
        * (
            NN
            + 6.0 * NE
            + 15.0 * NSE
            + 25.0 * NSI
            + 36.0 * NFI
            + 85.0 * NFO
            + 128.0 * NT
            + 104.0 * NS
            + 64.0 * N
            + 16.0
        )
        / (NM * NT * N1T * N2T)
    )
    PQGB = (
        (2.0 * S1 * S1 - 2.0 * S2 + 5.0) * (NS + N + 2.0) / (N * N1 * N2)
        - 4.0 * S1 / NS
        + (11.0 * NFO + 26.0 * NT + 15.0 * NS + 8.0 * N + 4.0) / (NT * N1T * N2)
    )

    PGQA = (
        (-S1 * S1 + 5.0 * S1 - S2) * (NS + N + 2.0) / (NM * N * N1)
        - 2.0 * S1 / N1S
        - (12.0 * NSI + 30.0 * NFI + 43.0 * NFO + 28.0 * NT - NS - 12.0 * N - 4.0)
        / (2.0 * NM * NT * N1T)
    )
    PGQB = (
        (S1 * S1 + S2 - SSTR2P) * (NS + N + 2.0) / (NM * N * N1)
        - S1 * (17.0 * NFO + 41.0 * NS - 22.0 * N - 12.0) / (3.0 * NMS * NS * N1)
        + (
            109.0 * NN
            + 621.0 * NE
            + 1400.0 * NSE
            + 1678.0 * NSI
            + 695.0 * NFI
            - 1031.0 * NFO
            - 1304.0 * NT
            - 152.0 * NS
            + 432.0 * N
            + 144.0
        )
        / (9.0 * NMS * NT * N1T * N2S)
    )
    PGQC = (S1 - 8.0 / 3.0) * (NS + N + 2.0) / (NM * N * N1) + 1.0 / N1S
    PGQC = 4.0 / 3.0 * PGQC

    PGGA = (
        -(2.0 * NFI + 5.0 * NFO + 8.0 * NT + 7.0 * NS - 2.0 * N - 2.0)
        * 8.0
        * S1
        / (NMS * NS * N1S * N2S)
        - 67.0 / 9.0 * S1
        + 8.0 / 3.0
        - 4.0 * SSTR2P * (NS + N + 1.0) / (NM * N * N1 * N2)
        + 2.0 * S1 * SSTR2P
        - 4.0 * SSCHLP
        + 0.5 * SSTR3P
        + (
            457.0 * NN
            + 2742.0 * NE
            + 6040.0 * NSE
            + 6098.0 * NSI
            + 1567.0 * NFI
            - 2344.0 * NFO
            - 1632.0 * NT
            + 560.0 * NS
            + 1488.0 * N
            + 576.0
        )
        / (18.0 * NMS * NT * N1T * N2T)
    )
    PGGB = (
        (38.0 * NFO + 76.0 * NT + 94.0 * NS + 56.0 * N + 12.0)
        * (-2.0)
        / (9.0 * NM * NS * N1S * N2)
        + 20.0 / 9.0 * S1
        - 4.0 / 3.0
    )
    PGGC = (2.0 * NSI + 4.0 * NFI + NFO - 10.0 * NT - 5.0 * NS - 4.0 * N - 4.0) * (
        -2.0
    ) / (NM * NT * N1T * N2) - 1.0

    P1Sqq = P1NSP + TR * NF * CF * PPSA * 4.0
    P1Sqg = TR * NF * (CA * PQGA + CF * PQGB) * 4.0
    P1Sgq = (CF * CF * PGQA + CF * CA * PGQB + TR * NF * CF * PGQC) * 4.0
    P1Sgg = (CA * CA * PGGA + TR * NF * (CA * PGGB + CF * PGGC)) * 4.0

    gS1 = ad_as2.gamma_singlet(N, NF, cache)
    np.testing.assert_allclose(gS1[0, 0], -P1Sqq)
    np.testing.assert_allclose(gS1[0, 1], -P1Sqg)
    np.testing.assert_allclose(gS1[1, 0], -P1Sgq)
    np.testing.assert_allclose(gS1[1, 1], -P1Sgg)
