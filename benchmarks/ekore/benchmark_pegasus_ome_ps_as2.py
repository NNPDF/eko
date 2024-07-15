"""Benchmark the polarized NNLO OME against PEGASUS"""

import numpy as np
import pytest

import ekore.harmonics as h
import ekore.operator_matrix_elements.polarized.space_like.as2 as ome_as2
from eko.constants import CA, CF, TR, zeta2, zeta3


@pytest.mark.isolated
def benchmark_pegasus_ome_ps_as2():
    # remember that singlet has pole at N=1
    for N in [2, 3, 4, +1j, -1j, +1j, +1j]:
        for NF in [3, 4, 5]:
            check_pegasus_ome_ps_as2_s(N, NF)


def check_pegasus_ome_ps_as2_s(N, NF):
    # Test against pegasus implementation by Ignacio Borsa
    ZETA2 = zeta2
    ZETA3 = zeta3

    S1 = h.S1(N)
    S2 = h.S2(N)
    S3 = h.S3(N)
    #
    NM = N - 1.0
    N1 = N + 1.0
    N2 = N + 2.0
    NI = 1.0 / N
    NMI = 1.0 / NM
    N1I = 1.0 / N1
    N2I = 1.0 / N2
    #
    S1M = S1 - NI
    S2M = S2 - NI * NI
    S3M = S3 - NI**3
    S11 = S1 + N1I
    S21 = S2 + N1I * N1I
    S31 = S3 + N1I**3
    S22 = S21 + N2I * N2I
    ACG3 = h.g_functions.mellin_g3(N1, S11)
    #
    #   CALL BET(N1,V1)
    #   CALL BET1(N1,V2)
    #   CALL BET2(N1,V3)
    #   CALL BET3(N1,V4)
    V1 = (
        h.polygamma.cern_polygamma((N1 + 1.0) / 2.0, 0)
        - h.polygamma.cern_polygamma(N1 / 2.0, 0)
    ) / 2.0
    V2 = (
        h.polygamma.cern_polygamma((N1 + 1.0) / 2.0, 1)
        - h.polygamma.cern_polygamma(N1 / 2.0, 1)
    ) / 4.0
    V3 = (
        h.polygamma.cern_polygamma((N1 + 1.0) / 2.0, 2)
        - h.polygamma.cern_polygamma(N1 / 2.0, 2)
    ) / 8.0
    #
    #
    # ..The moments of the OME's DA_Hq^{PS,(2)} and DA_Hg^{S,(2)} given in
    #    Eqs. (138) and (111) of BBDKS. Note that for the former, an
    #    additional finite renormalization is needed to go from the Larin
    #    to the the M scheme
    #
    #
    # ... Anomalous dimension terms
    #
    G0QG_HAT = -8 * TR * NM / N / N1
    #
    G0GQ = -4 * CF * N2 / N / N1
    #
    G0QQ = -CF * (2 * (2.0 + 3.0 * N + 3.0 * N**2) / N / N1 - 8.0 * S1)
    #
    # ... Polinomials in N
    POL1 = (
        12 * N**8 + 52 * N**7 + 60 * N**6 - 25 * N**4 - 2 * N**3 + 3 * N**2 + 8 * N + 4
    )
    #
    POL2 = (
        2.0 * N**8
        + 10.0 * N**7
        + 22.0 * N**6
        + 36.0 * N**5
        + 29.0 * N**4
        + 4.0 * N**3
        + 33.0 * N**2
        + 12.0 * N
        + 4
    )
    #
    POLR3 = 15 * N**6 + 45 * N**5 + 374 * N**4 + 601 * N**3 + 161 * N**2 - 24 * N + 36
    #
    POLR8 = (
        -15 * N**8
        - 60 * N**7
        - 82 * N**6
        - 44 * N**5
        - 15 * N**4
        - 4 * N**2
        - 12 * N
        - 8
    )
    #
    # ... Finite renormalization term from Larin to M scheme
    #
    ZQQPS = -CF * TR * 8 * N2 * (N**2 - N - 1.0) / N**3 / N1**3
    #
    A2HQ = (
        -4
        * CF
        * TR
        * N2
        / N**2
        / N1**2
        * (NM * (2 * S2 + ZETA2) - (4 * N**3 - 4 * N**2 - 3 * N - 1) / N**2 / N1**2)
        + ZETA2 / 8 * G0QG_HAT * G0GQ
    )
    #
    #
    A2HG = (
        CF
        * TR
        * (
            4.0
            / 3
            * NM
            / N
            / N1
            * (-4.0 * S3 + S1**3 + 3.0 * S1 * S2 + 6.0 * S1 * ZETA2)
            - 4
            * (N**4 + 17.0 * N**3 + 43.0 * N**2 + 33.0 * N + 2)
            * S2
            / N**2
            / N1**2
            / N2
            - 4 * (3.0 * N**2 + 3.0 * N - 2) * S1**2 / N**2 / N1 / N2
            - 2 * NM * (3.0 * N**2 + 3.0 * N + 2) * ZETA2 / N**2 / N1**2
            - 4 * (N**3 - 2.0 * N**2 - 22.0 * N - 36) * S1 / N**2 / N1 / N2
            - 2 * POL1 / N**4 / N1**4 / N2
        )
        + CA
        * TR
        * (
            4 * (N**2 + 4.0 * N + 5) * S1**2 / N / N1**2 / N2
            + 4 * (7.0 * N**3 + 24.0 * N**2 + 15.0 * N - 16) * S2 / N**2 / N1**2 / N2
            + 8 * NM * N2 * ZETA2 / N**2 / N1**2
            + 4 * (N**4 + 4.0 * N**3 - N**2 - 10.0 * N + 2) * S1 / N / N1**3 / N2
            - 4 * POL2 / N**4 / N1**4 / N2
            - 16 * NM / N / N1**2 * V2
            + 4
            * NM
            / 3.0
            / N
            / N1
            * (
                12.0 * ACG3
                + 3.0 * V3
                - 8.0 * S3
                - S1**3
                - 9.0 * S1 * S2
                - 12.0 * S1 * V2
                - 12.0 * V1 * ZETA2
                - 3.0 * ZETA3
            )
        )
        # ...  added simplified Gamma0_gg+2*beta0
        + 1 / 8 * G0QG_HAT * (8 * CA * (-2.0 / N / N1 + S1) - G0QQ)
    )
    #
    # ..The moments of the OME's DA_{gq,H}^{S,(2)} and DA_{gg,H}^{S,(2)}
    #    given in Eqs. (175) and (188) of Bierenblaum et al.
    #
    A2GQ = (
        CF
        * TR
        * N2
        * (
            8 * (22.0 + 41.0 * N + 28.0 * N**2) / 27.0 / N / N1**3
            - 8 * (2.0 + 5.0 * N) * S1 / 9.0 / N / N1**2
            + 4 * (S1**2 + S2) / 3.0 / N / N1
        )
    )
    #
    A2GG = (
        CA
        * TR
        * (2 * POLR3 / 27.0 / N**3 / N1**3 - 4 * (47.0 + 56.0 * N) * S1 / 27.0 / N1)
        + CF * TR * POLR8 / N**4 / N1**4
    )

    cache = h.cache.reset()
    omeS2 = ome_as2.A_singlet(N, cache, 0.0, NF)
    np.testing.assert_allclose(omeS2[0, 0], A2GG, err_msg=f"gg,{N=}")
    np.testing.assert_allclose(omeS2[0, 1], A2GQ, err_msg=f"gq,{N=}")
    np.testing.assert_allclose(omeS2[2, 1], A2HQ + NF * ZQQPS, err_msg=f"hq,{N=}")
    np.testing.assert_allclose(omeS2[2, 0], A2HG, err_msg=f"hg,{N=}")
