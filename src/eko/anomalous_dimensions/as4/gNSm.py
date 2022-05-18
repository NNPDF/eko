# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,-}^{(3)}`
"""
import numba as nb

from ...constants import CF
from ...harmonics.constants import zeta3


@nb.njit(cache=True)
def gamma_ns_nf3(n, sx):
    """
    Implements the common part proportional to :math:`nf^3`,
    of :math:`\\gamma_{ns,+}^{(3)},`\\gamma_{ns,-}^{(3)},`\\gamma_{ns,v}^{(3)}`

    The expression is copied exact from eq. 3.6. of :cite:`Davies:2016jie`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : np.ndarray
            List of harmonic sums: :math:`S_{1},S_{2},S_{3},S_{4}`

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
    """
    Implements the valence-like non-singlet part proportional to :math:`nf^2`

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
        -193.82677593897
        - 18.962962962962962963 / n**5
        + 99.1604938271604938271 / n**4
        - 226.4407530689903577046 / n**3
        + 395.6049773287730058193 / n**2
        - 552.1227765742141 / n
        + 84.67346826269055 / (1.0 + n)
        + 1118.2890034305392 / (2.0 + n)
        - 920.0583289838216 / (3.0 + n)
        + 446.1840201766196 / (4.0 + n)
        + 195.5772257829161 * S1
        + (26.68861454046639 * S1) / n
    )


@nb.njit(cache=True)
def gamma_nsm_nf1(n, sx):
    """
    Implements the valence-like non-singlet part proportional to :math:`nf^1`

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
        5548.200657713827
        - 126.4197530864197530864 / n**6
        + 752.1975308641975308642 / n**5
        - 2253.1105700880141651845 / n**4
        + 5247.1769880520207943186 / n**3
        - 8769.1532172950735377909 / n**2
        + 16400.857773299274 / n
        - 18274.65686868211 / (1.0 + n)
        + 7065.972162233177 / (2.0 + n)
        - 8922.588123903408 / (3.0 + n)
        + 633.2407788734786 / (4.0 + n)
        - 5171.916129085788 * S1
        - (2741.830025124657 * S1) / n
    )


@nb.njit(cache=True)
def gamma_nsm_nf0(n, sx):
    """
    Implements the valence-like non-singlet part proportional to :math:`nf^0`

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
        -23356.05285237236
        - 252.8395061728395061729 / n**7
        + 1580.2469135802469135803 / n**6
        - 5806.8001047043730187531 / n**5
        + 14899.9171192990199710453 / n**4
        - 28546.3876850661858598445 / n**3
        + 50759.6554123258829900607 / n**2
        - 118212.2312749378 / n
        + 290436.1449634707 / (1.0 + n)
        - 504425.7962380428 / (2.0 + n)
        + 521633.62063111813 / (3.0 + n)
        - 181016.71971921017 / (4.0 + n)
        + 20702.353028966703 * S1
        + (16950.937339235086 * S1) / n
    )


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    """
    Computes the |N3LO| valence-like non-singlet anomalous dimension.

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
