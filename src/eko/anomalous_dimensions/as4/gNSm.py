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
    of :math:`\\gamma_{ns,+}^{(3)},\\gamma_{ns,-}^{(3)},\\gamma_{ns,v}^{(3)}`

    The expression is copied exact from Eq. 3.6. of :cite:`Davies:2016jie`

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
        -193.85047343098682
        - 24.0 / n**5
        + 120.0 / n**4
        - 267.9804151038487 / n**3
        + 468.9321427047462 / n**2
        - 506.0520992835391 / n
        + 417.66660411336596 / (1.0 + n)
        + 243.84785694441783 / (2.0 + n)
        + 29.6046684644854 / (3.0 + n)
        + 195.5772257829161 * S1
        - (106.76271145909243 * S1) / n**2
        + (7.370132630198569 * S1) / n
        + (18.498220010659097 * S1) / (1.0 + n)
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
        5549.455619402364
        - 180.0 / n**6
        + 972.0 / n**5
        - 2906.082724682349 / n**4
        + 6563.952547153958 / n**3
        - 10394.296019235715 / n**2
        + 5747.561166966485 / n
        - 2513.3427774527718 / (1.0 + n)
        - 7019.134751026074 / (2.0 + n)
        + 141.04914866657916 / (3.0 + n)
        - 5171.916129085788 * S1
        + (7346.606718977439 * S1) / n**2
        - (5265.7498783155925 * S1) / n
        + (2599.1787681085734 * S1) / (1.0 + n)
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
        -23401.29036139005
        - 405.0 / n**7
        + 2160.0 / n**6
        - 8041.289536388223 / n**5
        + 19326.77625956448 / n**4
        - 38522.12974851202 / n**3
        + 58130.241100269224 / n**2
        - 74342.48799518021 / n
        + 74592.2912373345 / (1.0 + n)
        - 19659.361374964632 / (2.0 + n)
        + 25877.48324834917 / (3.0 + n)
        + 20702.353028966703 * S1
        - (13905.818349399015 * S1) / n**2
        + (24339.258697369463 * S1) / n
        - (6506.018135466623 * S1) / (1.0 + n)
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
