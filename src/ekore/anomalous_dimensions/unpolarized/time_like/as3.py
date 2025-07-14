"""The unpolarized, time-like |NNLO| Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np
from numpy import power as npp

from eko.constants import zeta2, zeta3

from ....harmonics import cache as c


@nb.njit(cache=True)
def gamma_nsp(N, nf, cache):
    r"""Compute the |NNLO| non-singlet positive anomalous dimension.

    Implements :eqref:`15` from  :cite:`Mitov:2006ic` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    ache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO non-singlet positive anomalous dimension
        :math:`\gamma_{ns,+}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    N1 = N + 1
    N1I = 1 / N1
    N1I2 = N1I * N1I
    N1I3 = N1I * N1I2
    N2 = N + 2
    N2I = 1 / N2
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S1M = S1 - NI
    S11 = S1 + N1I
    S21 = S2 + N1I2
    S31 = S3 + N1I3
    A0 = -S1M
    B1 = -S1 * NI
    C0 = NI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    D1 = -NI2
    D11 = -N1I2
    D2 = 2 * NI3
    D3 = -6 * NI2 * NI2
    D4 = 24 * NI2 * NI3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    E11 = S11 * N1I2 + (S21 - zeta2) * N1I
    E2 = 2 * (-S1 * NI3 + (zeta2 - S2) * NI2 - (S3 - zeta3) * NI)
    E21 = 2 * (-S11 * N1I3 + (zeta2 - S21) * N1I2 - (S31 - zeta3) * N1I)

    PP2 = (
        1174.898 * A0
        + 1295.625
        - 707.67 * B1
        + 593.9 * C3
        - 1075.3 * C2
        - 4249.4 * C1
        + 1658.7 * C0
        + 1327.5 * D1
        - 189.37 * D2
        - 352 / 9 * D3
        + 128 / 81 * D4
        - 56.907 * E1
        - 559.1 * E11
        - 519.37 * E2
        + nf
        * (
            -183.187 * A0
            - 173.935
            + 5120 / 81 * B1
            - 31.84 * C3
            + 181.18 * C2
            + 466.29 * C1
            - 198.10 * C0
            - 168.89 * D1
            - 176 / 81 * D2
            + 64 / 27 * D3
            - 50.758 * E1
            + 85.72 * E11
            + 28.551 * E2
            - 23.102 * E21
            - 39.113 * D11
        )
    )
    PF2 = (
        -(
            17 / 72
            - 2 / 27 * S1
            - 10 / 27 * S2
            + 2 / 9 * S3
            - (12 * npp(N, 4) + 2 * npp(N, 3) - 12 * npp(N, 2) - 2 * N + 3)
            / (27 * npp(N, 3) * npp(N1, 3))
        )
        * 32
        / 3
    )

    result = PP2 + npp(nf, 2) * PF2
    return -result


@nb.njit(cache=True)
def gamma_nsm(N, nf, cache):
    r"""Compute the |NNLO| non-singlet negative anomalous dimension.

    Implements :eqref:`16` from  :cite:`Mitov:2006ic` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO non-singlet negative anomalous dimension
        :math:`\gamma_{ns,-}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    N1 = N + 1
    N1I = 1 / N1
    N1I2 = N1I * N1I
    N1I3 = N1I * N1I2
    N2 = N + 2
    N2I = 1 / N2
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S1M = S1 - NI
    S11 = S1 + N1I
    S21 = S2 + N1I2
    S31 = S3 + N1I3
    A0 = -S1M
    B1 = -S1 * NI
    C0 = NI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    D1 = -NI2
    D11 = -N1I2
    D2 = 2 * NI3
    D3 = -6 * NI2 * NI2
    D31 = -6 * N1I2 * N1I2
    D4 = 24 * NI2 * NI3
    D41 = 24 * N1I2 * N1I3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    E11 = S11 * N1I2 + (S21 - zeta2) * N1I
    E2 = 2 * (-S1 * NI3 + (zeta2 - S2) * NI2 - (S3 - zeta3) * NI)
    E21 = 2 * (-S11 * N1I3 + (zeta2 - S21) * N1I2 - (S31 - zeta3) * N1I)

    PM2 = (
        1174.898 * A0
        + 1295.622
        - 707.94 * B1
        + 407.89 * C3
        - 577.42 * C2
        - 4885.7 * C1
        + 1981.3 * C0
        + 1625.5 * D1
        - 38.298 * D2
        - 3072 / 81 * D3
        - 140 / 81 * D4
        + 4563.2 * E1
        - 5140.6 * E11
        + 1905.4 * E2
        + 1969.5 * E21
        - 437.03 * D31
        - 34.683 * D41
        + nf
        * (
            -183.187 * A0
            - 173.9376
            + 5120 / 81 * B1
            - 85.786 * C3
            + 209.19 * C2
            + 511.92 * C1
            - 217.84 * C0
            - 188.99 * D1
            - 784 / 81 * D2
            + 128 / 81 * D3
            + 71.428 * E1
            - 23.722 * E11
            + 30.554 * E2
            - 18.975 * E21
            + 92.453 * D11
        )
    )
    PF2 = (
        -(
            17 / 72
            - 2 / 27 * S1
            - 10 / 27 * S2
            + 2 / 9 * S3
            - (12 * npp(N, 4) + 2 * npp(N, 3) - 12 * npp(N, 2) - 2 * N + 3)
            / (27 * npp(N, 3) * npp(N1, 3))
        )
        * 32
        / 3
    )

    result = PM2 + npp(nf, 2) * PF2
    return -result


@nb.njit(cache=True)
def gamma_nsv(N, nf, cache):
    r"""Compute the |NNLO| non-singlet valence anomalous dimension.

    Implements :eqref:`16` from  :cite:`Mitov:2006ic` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO non-singlet valence anomalous dimension
        :math:`\gamma_{ns,v}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    NM = N - 1
    NMI = 1 / NM
    N1 = N + 1
    N1I = 1 / N1
    N2 = N + 2
    N2I = 1 / N2
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S1M = S1 - NI
    S11 = S1 + N1I
    S12 = S11 + N2I
    B1 = -S1 * NI
    if abs(N.imag) < 0.00001 and abs(N.real - 1) < 0.00001:
        B1M = -zeta2
    else:
        B1M = -S1M * NMI
    B11 = -S11 * N1I
    B12 = -S12 * N2I
    C0 = NI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    C4 = 1 / (N + 4)
    D1 = -NI2
    D2 = 2 * NI3
    D3 = -6 * NI2 * NI2
    D4 = 24 * NI2 * NI3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    E2 = 2 * (-S1 * NI3 + (zeta2 - S2) * NI2 - (S3 - zeta3) * NI)

    PS2 = (
        -163.9 * (B1M - B1)
        - 7.208 * (B11 - B12)
        + 4.82 * (C3 - C4)
        - 43.12 * (C2 - C3)
        + 44.51 * (C1 - C2)
        + 151.49 * (C0 - C1)
        + 178.04 * D1
        + 6.892 * D2
        - 40 / 27 * (2 * D3 - D4)
        - 173.1 * E1
        + 46.18 * E2
    )

    result = gamma_nsm(N, nf, cache) + nf * PS2
    return -result


@nb.njit(cache=True)
def gamma_qq(N, nf, cache):
    r"""Compute the |NNLO| quark-quark anomalous dimension.

    Implements :eqref:`11` from  :cite:`Moch:2007tx` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO quark-quark anomalous dimension
        :math:`\gamma_{qq}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    NM = N - 1
    NMI = 1 / NM
    NMI2 = NMI * NMI
    NMI3 = NMI * NMI2
    N1 = N + 1
    N1I = 1 / N1
    N1I2 = N1I * N1I
    N1I3 = N1I * N1I2
    N2 = N + 2
    N2I = 1 / N2
    N2I2 = N2I * N2I
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S11 = S1 + N1I
    S21 = S2 + N1I2
    S31 = S3 + N1I3
    B1 = -S1 * NI
    B11 = -S11 * N1I
    B2 = (npp(S1, 2) + S2) * NI
    B21 = (npp(S11, 2) + S21) * N1I
    B3 = -(npp(S1, 3) + 3 * S1 * S2 + 2 * S3) * NI
    B31 = -(npp(S11, 3) + 3 * S11 * S21 + 2 * S31) * N1I
    C0 = NI
    CM = NMI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    C4 = 1 / (N + 4)
    C5 = 1 / (N + 5)
    D1 = -NI2
    D1M = -NMI2
    D11 = -N1I2
    D12 = -N2I2
    D2 = 2 * NI3
    D2M = 2 * NMI3
    D21 = 2 * N1I3
    D3 = -6 * NI2 * NI2
    D3M = -6 * NMI2 * NMI2
    D31 = -6 * N1I2 * N1I2
    D32 = -6 * N2I2 * N2I2
    D4 = 24 * NI2 * NI3
    D41 = 24 * N1I2 * N1I3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    E11 = S11 * N1I2 + (S21 - zeta2) * N1I

    PS1 = (
        -256 / 9 * (D3M - D3)
        - 128 / 9 * (D2M - D2)
        + 324.07 * (D1M - D1)
        + 479.87 * (CM - C0)
        + 9.072 * (D4 - D41)
        + 47.322 * (D3 - D31)
        + 425.14 * (D2 - D21)
        + 656.49 * (D1 - D11)
        - 5.926 * (B3 - B31)
        - 9.751 * (B2 - B21)
        - 8.650 * (B1 - B11)
        - 106.65 * (C0 - C1)
        - 848.97 * (C1 - C2)
        + 368.79 * (C2 - C3)
        - 61.284 * (C3 - C4)
        + 96.171 * (E1 - E11)
    )
    PS2 = (
        -128 / 81 * (CM - C0)
        + 0.019122 * (D4 - D41)
        - 1.900 * (D3 - D31)
        + 9.1682 * (D2 - D21)
        + 57.713 * (D1 - D11)
        + 1.778 * (B2 - B21)
        + 16.611 * (B1 - B11)
        + 87.795 * (C0 - C1)
        - 57.688 * (C1 - C2)
        - 41.827 * (C2 - C3)
        + 25.628 * (C3 - C4)
        - 7.9934 * (C4 - C5)
        - 2.1031 * (E1 - E11)
        + 26.294 * (D11 - D12)
        - 7.8645 * (D31 - D32)
    )

    result = nf * (PS1 + nf * PS2)
    return -result


@nb.njit(cache=True)
def gamma_qg(N, nf, cache):
    r"""Compute the |NNLO| quark-gluon anomalous dimension.

    Implements :eqref:`18` from  :cite:`Almasy:2011eq` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO quark-gluon anomalous dimension
        :math:`\gamma_{qg}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    NM = N - 1
    NMI = 1 / NM
    NMI2 = NMI * NMI
    NMI3 = NMI * NMI2
    N1 = N + 1
    N1I = 1 / N1
    N1I2 = N1I * N1I
    N1I3 = N1I * N1I2
    N2 = N + 2
    N2I = 1 / N2
    N2I2 = N2I * N2I
    N2I3 = N2I * N2I2
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S4 = c.get(c.S4, cache, N)
    S11 = S1 + N1I
    S12 = S11 + N2I
    S21 = S2 + N1I2
    S22 = S21 + N2I2
    B1 = -S1 * NI
    B11 = -S11 * N1I
    B12 = -S12 * N2I
    B2 = (npp(S1, 2) + S2) * NI
    B21 = (npp(S11, 2) + S21) * N1I
    B22 = (npp(S12, 2) + S22) * N2I
    B3 = -(npp(S1, 3) + 3 * S1 * S2 + 2 * S3) * NI
    B4 = (npp(S1, 4) + 6 * npp(S1, 2) * S2 + 8 * S1 * S3 + 3 * npp(S2, 2) + 6 * S4) * NI
    C0 = NI
    CM = NMI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    C4 = 1 / (N + 4)
    D1 = -NI2
    D1M = -NMI2
    D11 = -N1I2
    D12 = -N2I2
    D2 = 2 * NI3
    D2M = 2 * NMI3
    D21 = 2 * N1I3
    D22 = 2 * N2I3
    D3 = -6 * NI2 * NI2
    D3M = -6 * NMI2 * NMI2
    D31 = -6 * N1I2 * N1I2
    D4 = 24 * NI2 * NI3
    D41 = 24 * N1I2 * N1I3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    E11 = S11 * N1I2 + (S21 - zeta2) * N1I
    E12 = S12 * N2I2 + (S22 - zeta2) * N2I
    F1 = 2 * NI * (zeta3 + zeta2 * S1 - 0.5 * NI * (S1 * S1 + S2) - S1 * S2 - S3)

    QG1 = (
        -64 * (D3M + D2M)
        + 675.83 * D1M
        + 1141.7 * CM
        + 42.328 * D4
        + 361.28 * D3
        + 1512 * D2
        + 1864 * D1
        + 100 / 27 * B4
        + 350 / 9 * B3
        + 263.07 * B2
        + 693.84 * B1
        + 603.71 * C0
        - 882.48 * C1
        + 4723.2 * C2
        - 4745.8 * C3
        - 175.28 * C4
        - 1809.4 * E1
        - 107.59 * E11
        - 885.5 * D41
    )
    QG2 = (
        -32 / 27 * D2M
        - 3.1752 * D1M
        - 2.8986 * CM
        + 21.569 * D3
        + 255.62 * D2
        + 619.75 * D1
        - 100 / 27 * B3
        - 35.446 * B2
        - 103.609 * B1
        - 113.81 * C0
        + 341.26 * C1
        - 853.35 * C2
        + 492.1 * C3
        + 14.803 * C4
        + 966.96 * E1
        - 709.1 * E11
        - 1.593 * F1
        - 333.8 * D31
    )
    QG3 = (
        (
            4 * C0
            + 6 * (D1 + B1)
            + 3.8696 * (C0 - 2 * C1 + 2 * C2)
            + 4 * (D1 - 2 * D11 + 2 * D12 + B1 - 2 * B11 + 2 * B12)
            + 3 * (D2 - 2 * D21 + 2 * D22 + B2 - 2 * B21 + 2 * B22)
            + 6 * (E1 - 2 * E11 + 2 * E12)
        )
        * 4
        / 9
    )

    result = (QG1 + nf * (QG2 + nf * QG3)) / 2
    return -result


@nb.njit(cache=True)
def gamma_gq(N, nf, cache):
    r"""Compute the |NNLO| gluon-quark anomalous dimension.

    Implements :eqref:`19` from  :cite:`Almasy:2011eq` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO gluon-quark anomalous dimension
        :math:`\gamma_{gq}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    NM = N - 1
    NMI = 1 / NM
    NMI2 = NMI * NMI
    NMI3 = NMI * NMI2
    N1 = N + 1
    N1I = 1 / N1
    N1I2 = N1I * N1I
    N2 = N + 2
    N2I = 1 / N2
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S4 = c.get(c.S4, cache, N)
    B1 = -S1 * NI
    B2 = (npp(S1, 2) + S2) * NI
    B3 = -(npp(S1, 3) + 3 * S1 * S2 + 2 * S3) * NI
    B4 = (npp(S1, 4) + 6 * npp(S1, 2) * S2 + 8 * S1 * S3 + 3 * npp(S2, 2) + 6 * S4) * NI
    C0 = NI
    CM = NMI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    C4 = 1 / (N + 4)
    D1 = -NI2
    D1M = -NMI2
    D2 = 2 * NI3
    D2M = 2 * NMI3
    D3 = -6 * NI2 * NI2
    D3M = -6 * NMI2 * NMI2
    D31 = -6 * N1I2 * N1I2
    D4 = 24 * NI2 * NI3
    D4M = 24 * NMI2 * NMI3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    F1 = 2 * NI * (zeta3 + zeta2 * S1 - 0.5 * NI * (S1 * S1 + S2) - S1 * S2 - S3)

    GQ0 = (
        256 * D4M
        + 3712 / 3 * D3M
        + 1001.89 * D2M
        + 4776.5 * D1M
        + 5803.7 * CM
        - 30.062 * D4
        - 126.38 * D3
        - 0.71252 * D2
        + 4.4136 * D1
        + 400 / 81 * B4
        + 520 / 27 * B3
        - 220.13 * B2
        - 152.6 * B1
        + 272.85 * C0
        - 7188.7 * C1
        + 5693.2 * C2
        + 146.98 * C3
        + 128.19 * C4
        - 1300.6 * E1
        - 71.23 * F1
        + 543.8 * D31
    )
    GQ1 = (
        1280 / 81 * D3M
        + 2912 / 27 * D2M
        + 141.93 * D1M
        + 6.0041 * CM
        - 48.60 * D3
        - 343.1 * D2
        - 492.0 * D1
        + 80 / 81 * B3
        + 1040 / 81 * B2
        - 16.914 * B1
        - 871.3 * C0
        + 790.13 * C1
        - 241.23 * C2
        + 43.252 * C3
        - 4.3465 * D31
        + 55.048 * E1
    )

    result = 2 * nf * (GQ0 + nf * GQ1)
    return -result


@nb.njit(cache=True)
def gamma_gg(N, nf, cache):
    r"""Compute the |NNLO| gluon-gluon anomalous dimension.

    Implements :eqref:`12` from  :cite:`Moch:2007tx` via the
    N-space translation from A. Vogt.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NNLO gluon-gluon anomalous dimension
        :math:`\gamma_{gg}^{(2)}(N)`
    """
    NI = 1 / N
    NI2 = NI * NI
    NI3 = NI * NI2
    NM = N - 1
    NMI = 1 / NM
    NMI2 = NMI * NMI
    NMI3 = NMI * NMI2
    N1 = N + 1
    N1I = 1 / N1
    N1I2 = N1I * N1I
    N2 = N + 2
    N2I = 1 / N2
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S1M = S1 - NI
    S11 = S1 + N1I
    S21 = S2 + N1I2
    A0 = -S1M
    B1 = -S1 * NI
    C0 = NI
    CM = NMI
    C1 = N1I
    C2 = N2I
    C3 = 1 / (N + 3)
    C4 = 1 / (N + 4)
    D1 = -NI2
    D1M = -NMI2
    D2 = 2 * NI3
    D2M = 2 * NMI3
    D3 = -6 * NI2 * NI2
    D3M = -6 * NMI2 * NMI2
    D31 = -6 * N1I2 * N1I2
    D4 = 24 * NI2 * NI3
    D4M = 24 * NMI2 * NMI3
    E1 = S1 * NI2 + (S2 - zeta2) * NI
    E11 = S11 * N1I2 + (S21 - zeta2) * N1I
    E2 = 2 * (-S1 * NI3 + (zeta2 - S2) * NI2 - (S3 - zeta3) * NI)
    F1 = 2 * NI * (zeta3 + zeta2 * S1 - 0.5 * NI * (S1 * S1 + S2) - S1 * S2 - S3)

    GG0 = (
        576 * D4M
        + 3168 * D3M
        + 3651.1 * D2M
        + 10233 * D1M
        + 14214.4 * CM
        + 191.99 * D4
        + 3281.7 * D3
        + 13528 * D2
        + 12258 * D1
        - 28489 * C0
        + 7469 * C1
        + 30421 * C2
        - 53017 * C3
        + 19556 * C4
        - 186.4 * E1
        - 21328 * E2
        + 5685.8 * D31
        - 3590.1 * B1
        + 4425.451
        + 2643.521 * A0
    )
    GG1 = (
        448 / 9 * D3M
        + 2368 / 9 * D2M
        - 5.470 * D1M
        - 804.13 * CM
        + 18.085 * D4
        + 155.10 * D3
        + 482.94 * D2
        + 4.9934 * D1
        + 248.95 * C0
        + 260.6 * C1
        + 272.79 * C2
        + 2133.2 * C3
        - 926.87 * C4
        + 1266.5 * E1
        - 29.709 * E2
        + 87.771 * F1
        + 485.18 * D31
        + 319.97 * B1
        - 528.719
        - 412.172 * A0
    )
    GG2 = (
        32 / 27 * D2M
        + 368 / 81 * D1M
        + 472 / 243 * CM
        - 5.0372 * D3
        - 44.80 * D2
        - 69.712 * D1
        - 77.190 * C0
        + 153.27 * C1
        - 106.03 * C2
        + 11.995 * C3
        - 115.01 * E1
        + 96.522 * E11
        - 62.908 * E2
        + 6.4628
        - 16 / 9 * A0
    )

    result = GG0 + nf * (GG1 + nf * GG2)
    return -result


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the |NNLO| singlet anomalous dimension matrix.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        NNLO singlet anomalous dimension matrix
        :math:`\gamma_{s}^{(2)}`
    """
    result = np.array(
        [
            [gamma_qq(N, nf, cache), gamma_gq(N, nf, cache)],
            [gamma_qg(N, nf, cache), gamma_gg(N, nf, cache)],
        ],
        np.complex128,
    )
    return result
