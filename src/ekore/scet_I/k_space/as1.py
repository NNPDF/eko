r"""SCET 1 kernel entries.

"""
import numba as nb
import numpy as np

from eko.constants import CF, zeta2
from ...harmonics import cache as c


@nb.njit(cache=True)
def Agg10(n, cache):
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    res = (
        (
            3
            * (
                -2
                - n
                + 5 * np.power(n, 2)
                + 2 * np.power(n, 3)
                + 2 * np.power(n, 4)
                + 2 * np.power(n, 2) * zeta2
                - np.power(n, 3) * zeta2
                - 3 * np.power(n, 4) * zeta2
                + np.power(n, 5) * zeta2
                + np.power(n, 6) * zeta2
            )
        )
        / (np.power(-1 + n, 2) * np.power(n, 2) * (1 + n) * (2 + n))
        - (6 * (1 + n + np.power(n, 2)) * S1) / ((-1 + n) * n * (1 + n) * (2 + n))
        + (3 * np.power(S1, 2)) / 2.0
        - (3 * S2) / 2.0
    )
    return res


@nb.njit(cache=True)
def Agg11(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = (-6 * (1 + n + np.power(n, 2))) / ((-1 + n) * n * (1 + n) * (2 + n)) + 3 * S1
    return res


@nb.njit(cache=True)
def Agg12(n, cache):
    res = 1.5
    return res


@nb.njit(cache=True)
def Agq10(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = (2 * (-2 + 5 * np.power(n, 2) + np.power(n, 4))) / (
        3.0 * np.power(-1 + n, 2) * np.power(n, 2) * (1 + n)
    ) - (2 * (2 + n + np.power(n, 2)) * S1) / (3.0 * (-1 + n) * n * (1 + n))
    return res


@nb.njit(cache=True)
def Agq11(n, cache):
    res = (-2 * (2 + n + np.power(n, 2))) / (3.0 * (-1 + n) * n * (1 + n))
    return res


@nb.njit(cache=True)
def Agq12(n, cache):
    res = 0
    return res


@nb.njit(cache=True)
def Aqg10(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = 1 / (4.0 * np.power(n, 2)) + ((-2 - n - np.power(n, 2)) * S1) / (
        4.0 * n * (1 + n) * (2 + n)
    )
    return res


@nb.njit(cache=True)
def Aqg11(n, cache):
    res = (-2 - n - np.power(n, 2)) / (4.0 * n * (1 + n) * (2 + n))
    return res


@nb.njit(cache=True)
def Aqg12(n, cache):
    res = 0
    return res


@nb.njit(cache=True)
def Aqq10(n, cache):
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    res = (
        (2 * (1 + 2 * n + 2 * np.power(n, 2) * zeta2 + 2 * np.power(n, 3) * zeta2))
        / (3.0 * np.power(n, 2) * (1 + n))
        - (2 * S1) / (3.0 * n * (1 + n))
        + (2 * np.power(S1, 2)) / 3.0
        - (2 * S2) / 3.0
    )
    return res


@nb.njit(cache=True)
def Aqq11(n, cache):
    S1 = c.get(c.S1, cache, n)
    res = -2 / (3.0 * n * (1 + n)) + (4 * S1) / 3.0
    return res


@nb.njit(cache=True)
def Aqq12(n, cache):
    res = 0.6666666666666666
    return res

