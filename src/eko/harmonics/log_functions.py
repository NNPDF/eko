# -*- coding: utf-8 -*-
"""Implementation of some Mellin transformation of functions as:

    :math:`(1-x)\\ln^k(1-x), \\quad k = 1,2,3`

"""
import numba as nb

from .constants import zeta3


@nb.njit(cache=True)
def lm11m1(n, S1):
    """Mellin transform of :math:`(1-x)\\ln(1-x)`

    Parameters
    ----------
    n : complex
        Mellin moment
    S1:  complex
        Harmonic sum :math:`S_{1}(N)`

    Returns
    -------
    complex

    """
    return 1 / (1 + n) ** 2 - S1 / (1 + n) ** 2 - S1 / (n * (1 + n) ** 2)


@nb.njit(cache=True)
def lm12m1(n, S1, S2):
    """Mellin transform of :math:`(1-x)\\ln^2(1-x)`

    Parameters
    ----------
    n : complex
        Mellin moment
    S1:  complex
        Harmonic sum :math:`S_{1}(N)`
    S2:  complex
        Harmonic sum :math:`S_{2}(N)`

    Returns
    -------
    complex

    """
    return (
        -2 / (1 + n) ** 3
        - (2 * S1) / (1 + n) ** 2
        + S1**2 / n
        - S1**2 / (1 + n)
        + S2 / n
        - S2 / (1 + n)
    )


@nb.njit(cache=True)
def lm13m1(n, S1, S2, S3):
    """Mellin transform of :math:`(1-x)\\ln^3(1-x)`

    Parameters
    ----------
    n : complex
        Mellin moment
    S1:  complex
        Harmonic sum :math:`S_{1}(N)`
    S2:  complex
        Harmonic sum :math:`S_{2}(N)`
    S3:  complex
        Harmonic sum :math:`S_{3}(N)`

    Returns
    -------
    complex

    """
    return (
        6 / (1 + n) ** 4
        + (6 * S1) / (1 + n) ** 3
        + (3 * S1**2) / (1 + n) ** 2
        - S1**3 / n
        + S1**3 / (1 + n)
        + (3 * S2) / (1 + n) ** 2
        - (3 * S1 * S2) / n
        + (3 * S1 * S2) / (1 + n)
        - (2 * (6 * (2 * S3 - 2 * zeta3) + zeta3)) / n
        + (2 * (6 * (2 * S3 - 2 * zeta3) + zeta3)) / (1 + n)
    )
