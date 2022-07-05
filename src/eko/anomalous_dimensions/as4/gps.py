# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ps}^{(3)}`
"""
import numba as nb
import numpy as np


@nb.njit(cache=True)
def gamma_ps_nf3(n, sx):
    """Implements the part proportional to :math:`nf^3` of :math:`\\gamma_{ps}^{(3)}`
    The expression is copied exact from Eq. 3.10 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\\gamma_{ps}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return 1.3333333333333333 * (
        16.305796943701882 / (-1.0 + n)
        + 3.5555555555555554 / np.power(n, 5)
        - 17.185185185185187 / np.power(n, 4)
        + 28.839506172839506 / np.power(n, 3)
        - 48.95252257604665 / np.power(n, 2)
        + 23.09354523864061 / n
        + 39.111111111111114 / np.power(1.0 + n, 5)
        - 61.03703703703704 / np.power(1.0 + n, 4)
        - 10.666666666666666 / np.power(1.0 + n, 3)
        + 59.29439100420026 / np.power(1.0 + n, 2)
        - 94.20465634975173 / (1.0 + n)
        + 18.962962962962962 / np.power(2.0 + n, 4)
        - 34.76543209876543 / np.power(2.0 + n, 3)
        + 14.222222222222221 / np.power(2.0 + n, 2)
        + 54.805314167409236 / (2.0 + n)
        - (1.5802469135802468 * S1) / (-1.0 + n)
        + (7.111111111111111 * S1) / np.power(n, 4)
        - (34.370370370370374 * S1) / np.power(n, 3)
        + (38.71604938271605 * S1) / np.power(n, 2)
        - (37.135802469135804 * S1) / n
        + (35.55555555555556 * S1) / np.power(1.0 + n, 4)
        - (43.851851851851855 * S1) / np.power(1.0 + n, 3)
        - (39.50617283950617 * S1) / np.power(1.0 + n, 2)
        + (89.28395061728395 * S1) / (1.0 + n)
        + (18.962962962962962 * S1) / np.power(2.0 + n, 3)
        - (34.76543209876543 * S1) / np.power(2.0 + n, 2)
        - (50.5679012345679 * S1) / (2.0 + n)
        + (4.7407407407407405 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        + (7.111111111111111 * (np.power(S1, 2) + S2)) / np.power(n, 3)
        - (15.407407407407407 * (np.power(S1, 2) + S2)) / np.power(n, 2)
        + (13.037037037037036 * (np.power(S1, 2) + S2)) / n
        + (14.222222222222221 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 3)
        - (4.7407407407407405 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        - (20.14814814814815 * (np.power(S1, 2) + S2)) / (1.0 + n)
        + (9.481481481481481 * (np.power(S1, 2) + S2)) / np.power(2.0 + n, 2)
        + (2.3703703703703702 * (np.power(S1, 2) + S2)) / (2.0 + n)
        - (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(n, 2)
        - (1.1851851851851851 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        + (2.3703703703703702 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / np.power(1.0 + n, 2)
        + (1.1851851851851851 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
        + (1.5802469135802468 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (2.0 + n)
    )


@nb.njit(cache=True)
def gamma_ps_nf2(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_ps_nf1(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_ps(n, nf, sx):
    """Computes the |N3LO| pure singlet quark-quark anomalous dimension.

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
        |N3LO| pure singlet quark-quark anomalous dimension
        :math:`\\gamma_{ps}^{(3)}(N)`

    See Also
    --------
    gamma_ps_nf1: :math:`\\gamma_{ps}^{(3)}|_{nf^1}`
    gamma_ps_nf2: :math:`\\gamma_{ps}^{(3)}|_{nf^2}`
    gamma_ps_nf3: :math:`\\gamma_{ps}^{(3)}|_{nf^3}`

    """
    return (
        +nf * gamma_ps_nf1(n, sx)
        + nf**2 * gamma_ps_nf2(n, sx)
        + nf**3 * gamma_ps_nf3(n, sx)
    )
