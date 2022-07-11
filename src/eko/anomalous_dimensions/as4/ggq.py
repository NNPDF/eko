# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{gq}^{(3)}`
"""
import numba as nb
import numpy as np


@nb.njit(cache=True)
def gamma_gq_nf3(n, sx):
    """Implements the part proportional to :math:`nf^3` of :math:`\\gamma_{gq}^{(3)}`
    The expression is copied exact from Eq. 3.13 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\\gamma_{gq}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return 1.3333333333333333 * (
        -11.39728026699467 / (-1.0 + n)
        + 11.39728026699467 / n
        - 2.3703703703703702 / np.power(1.0 + n, 4)
        + 6.320987654320987 / np.power(1.0 + n, 3)
        - 3.1604938271604937 / np.power(1.0 + n, 2)
        - 5.698640133497335 / (1.0 + n)
        - (6.320987654320987 * S1) / (-1.0 + n)
        + (6.320987654320987 * S1) / n
        - (2.3703703703703702 * S1) / np.power(1.0 + n, 3)
        + (6.320987654320987 * S1) / np.power(1.0 + n, 2)
        - (3.1604938271604937 * S1) / (1.0 + n)
        + (6.320987654320987 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        - (6.320987654320987 * (np.power(S1, 2) + S2)) / n
        - (1.1851851851851851 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        + (3.1604938271604937 * (np.power(S1, 2) + S2)) / (1.0 + n)
        - (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        - (0.3950617283950617 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_gq_nf2(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_gq_nf1(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_gq(n, nf, sx):
    """Computes the |N3LO| gluon-quark singlet anomalous dimension.

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
    complex
        |N3LO| gluon-quark singlet anomalous dimension
        :math:`\\gamma_{gq}^{(3)}(N)`

    See Also
    --------
    gamma_gq_nf1: :math:`\\gamma_{gq}^{(3)}|_{nf^1}`
    gamma_gq_nf2: :math:`\\gamma_{gq}^{(3)}|_{nf^2}`
    gamma_gq_nf3: :math:`\\gamma_{gq}^{(3)}|_{nf^3}`

    """
    return (
        +nf * gamma_gq_nf1(n, sx)
        + nf**2 * gamma_gq_nf2(n, sx)
        + nf**3 * gamma_gq_nf3(n, sx)
    )
