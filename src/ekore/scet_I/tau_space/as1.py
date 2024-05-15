r"""SCET 1 kernel entries.

"""
import numba as nb
import numpy as np

from eko.constants import CF, zeta2
from ...harmonics import cache as c


@nb.njit(cache=True)
def A_gg(n, order, cache):
    r"""
    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| :math:`A_{gg}`

    """

    if order == (1, -1):
        res = Agg1m1(n, cache)

    if order == (1, 0):
        res = Agg10(n, cache)

    if order == (1, 1):
        res = Agg11(n, cache)

    return res


@nb.njit(cache=True)
def A_gq(n, order, cache):
    r"""
    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| :math:`A_{gq}`

    """

    if order == (1, -1):
        res = Agq1m1(n, cache)

    if order == (1, 0):
        res = Agq10(n, cache)

    if order == (1, 1):
        res = Agq11(n, cache)

    return res


@nb.njit(cache=True)
def A_qg(n, order, cache):
    r"""
    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| :math:`A_{qg}`

    """

    if order == (1, -1):
        res = Aqg1m1(n, cache)

    if order == (1, 0):
        res = Aqg10(n, cache)

    if order == (1, 1):
        res = Aqg11(n, cache)

    return res


@nb.njit(cache=True)
def A_qq(n, order, cache):
    r"""
    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NLO| :math:`A_{qq}`

    """

    if order == (1, -1):
        res = Aqq1m1(n, cache)

    if order == (1, 0):
        res = Aqq10(n, cache)

    if order == (1, 1):
        res = Aqq11(n, cache)

    return res


@nb.njit(cache=True)
def A_qQ2(n, order):
    if order == (1, -1):
        res = 0.0

    if order == (1, 0):
        res = 0.0

    if order == (1, 1):
        res = 0.0

    return res


@nb.njit(cache=True)
def A_qQ2bar(n, order):
    if order == (1, -1):
        res = 0.0

    if order == (1, 0):
        res = 0.0

    if order == (1, 1):
        res = 0.0

    return res


@nb.njit(cache=True)
def A_qqbar(n, order):
    if order == (1, -1):
        res = 0.0

    if order == (1, 0):
        res = 0.0

    if order == (1, 1):
        res = 0.0

    return res


@nb.njit(cache=True)
def Agg1m1(n, cache):
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    res = (
        (
            3
            * (
                -4
                - 2 * n
                + 10 * np.power(n, 2)
                + 4 * np.power(n, 3)
                + 4 * np.power(n, 4)
                + 2 * np.power(n, 2) * zeta2
                - np.power(n, 3) * zeta2
                - 3 * np.power(n, 4) * zeta2
                + np.power(n, 5) * zeta2
                + np.power(n, 6) * zeta2
            )
        )
        / (2.0 * np.power(-1 + n, 2) * np.power(n, 2) * (1 + n) * (2 + n))
        - (6 * (1 + n + np.power(n, 2)) * S1) / ((-1 + n) * n * (1 + n) * (2 + n))
        + (3 * np.power(S1, 2)) / 2.0
        - (3 * S2) / 2.0
    )
    return res


@nb.njit(cache=True)
def Agg10(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = (6 * (1 + n + np.power(n, 2))) / ((-1 + n) * n * (1 + n) * (2 + n)) - 3 * S1
    return res


@nb.njit(cache=True)
def Agg11(n, cache):
    res = 3.0
    return res


@nb.njit(cache=True)
def Agq1m1(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = (2 * (-2 + 5 * np.power(n, 2) + np.power(n, 4))) / (
        3.0 * np.power(-1 + n, 2) * np.power(n, 2) * (1 + n)
    ) - (2 * (2 + n + np.power(n, 2)) * S1) / (3.0 * (-1 + n) * n * (1 + n))
    return res


@nb.njit(cache=True)
def Agq10(n, cache):
    res = (2 * (2 + n + np.power(n, 2))) / (3.0 * (-1 + n) * n * (1 + n))
    return res


@nb.njit(cache=True)
def Agq11(n, cache):
    res = 0
    return res


@nb.njit(cache=True)
def Aqg1m1(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = 1 / (4.0 * np.power(n, 2)) + ((-2 - n - np.power(n, 2)) * S1) / (
        4.0 * n * (1 + n) * (2 + n)
    )
    return res


@nb.njit(cache=True)
def Aqg10(n, cache):
    res = (2 + n + np.power(n, 2)) / (4.0 * n * (1 + n) * (2 + n))
    return res


@nb.njit(cache=True)
def Aqg11(n, cache):
    res = 0
    return res


@nb.njit(cache=True)
def Aqq1m1(n, cache):
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    res = (
        (2 * (1 + 2 * n + np.power(n, 2) * zeta2 + np.power(n, 3) * zeta2))
        / (3.0 * np.power(n, 2) * (1 + n))
        - (2 * S1) / (3.0 * n * (1 + n))
        + (2 * np.power(S1, 2)) / 3.0
        - (2 * S2) / 3.0
    )
    return res


@nb.njit(cache=True)
def Aqq10(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = 2 / (3.0 * n * (1 + n)) - (4 * S1) / 3.0
    return res


@nb.njit(cache=True)
def Aqq11(n, cache):
    res = 1.3333333333333333
    return res
