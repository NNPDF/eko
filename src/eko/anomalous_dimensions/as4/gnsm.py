# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,-}^{(3)}`
"""
import numba as nb

from ...constants import CF
from ...harmonics.constants import zeta3


@nb.njit(cache=True)
def gamma_ns_nf3(n, sx):
    """Implements the common part proportional to :math:`nf^3`,
    of :math:`\\gamma_{ns,+}^{(3)},\\gamma_{ns,-}^{(3)},\\gamma_{ns,v}^{(3)}`.
    The expression is copied exact from Eq. 3.6. of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf3 : complex
        |N3LO| non-singlet anomalous dimension :math:`\\gamma_{ns}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    eta = 1 / n * 1 / (n + 1)
    g_ns_nf3 = CF * (
        -32 / 27 * zeta3 * eta
        - 16 / 9 * zeta3
        - 16 / 27 * eta**4
        - 16 / 81 * eta**3
        + 80 / 27 * eta**2
        - 320 / 81 * eta
        + 32 / 27 * 1 / (n + 1) ** 4
        + 128 / 27 * 1 / (n + 1) ** 2
        + 64 / 27 * S1 * zeta3
        - 32 / 81 * S1
        - 32 / 81 * S2
        - 160 / 81 * S3
        + 32 / 27 * S4
        + 131 / 81
    )
    return g_ns_nf3


@nb.njit(cache=True)
def gamma_nsm_nf2(n, sx):
    """Implements the parametrized valence-like non-singlet part proportional to :math:`nf^2`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf2 : complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    return (
        -193.7261824250166
        - 24.0 / n**5
        + 120.0 / n**4
        - 267.9804151038487 / n**3
        + 468.9321427047462 / n**2
        + 278.2205375565073 / n
        - 189.83542754839095 / (1.0 + n) ** 2
        + 285.7074536014645 / (1.0 + n)
        - 743.4405001163758 / (2.0 + n)
        + 943.1309142049942 / (3.0 + n)
        - 497.2568774192954 / (4.0 + n)
        + 195.5772257829161 * S1
        - (579.4502541796713 * S1) / n**2
        + (26.68861454046639 * S1) / n
        - (16.34878363666273 * S1) / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_nsm_nf1(n, sx):
    """Implements the parametrized valence-like non-singlet part proportional to :math:`nf^1`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf1 : complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    return (
        5549.01315461886
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        - 14963.650180654404 / (1.0 + n) ** 3
        + 12500.017478367376 / (1.0 + n) ** 2
        + 1888.352613656361 / (2.0 + n)
        - 5171.916129085788 * S1
        + (13630.581061908193 * S1) / n**2
        - (2741.830025124657 * S1) / n
        - (4176.27234870538 * S1) / (1.0 + n) ** 3
        - (6841.089476550462 * S1) / (1.0 + n) ** 2
        + (132.24791252411356 * S1) / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_nsm_nf0(n, sx):
    """Implements the parametrized valence-like non-singlet part proportional to :math:`nf^0`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
        g_ns_nf0 : complex
            |N3LO| valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-}^{(3)}|_{nf^0}`
    """
    S1 = sx[0][0]
    return (
        -23372.81433209934
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        + 127537.68077952677 / (1.0 + n) ** 3
        + 112779.53701686356 / (1.0 + n) ** 2
        + 1290.9133307724996 / (2.0 + n)
        + 20702.353028966703 * S1
        - (107354.75965428875 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - (67085.44144692074 * S1) / (1.0 + n) ** 3
        + (15726.094928140881 * S1) / (1.0 + n) ** 2
        - (2301.017215132865 * S1) / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    """Computes the |N3LO| valence-like non-singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : list
        harmonic sums cache

    Returns
    -------
    gamma_nsp : complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(3)}(N)`

    See Also
    --------
    gamma_nsm_nf0: :math:`\\gamma_{ns,-}^{(3)}|_{nf^0}`
    gamma_nsm_nf1: :math:`\\gamma_{ns,-}^{(3)}|_{nf^1}`
    gamma_nsm_nf2: :math:`\\gamma_{ns,-}^{(3)}|_{nf^2}`
    gamma_ns_nf3: :math:`\\gamma_{ns}^{(3)}|_{nf^3}`

    """
    return (
        gamma_nsm_nf0(n, sx)
        + nf * gamma_nsm_nf1(n, sx)
        + nf**2 * gamma_nsm_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
