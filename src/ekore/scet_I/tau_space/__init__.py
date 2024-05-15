import numba as nb
import numpy as np

from eko.constants import CF, zeta2
from . import as1, as2
from ...harmonics import cache as c


@nb.njit(cache=True)
def A_gg(n, nf, order, cache):
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
        :math:`A_{gg}`

    """

    if order == (1, -1):
        res = as1.Agg1m1(n, cache)

    if order == (1, 0):
        res = as1.Agg10(n, cache)

    if order == (1, 1):
        res = as1.Agg11(n, cache)

    if order == (2, -1):
        res = as2.Agg2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.Agg20(n, nf, cache)

    if order == (2, 1):
        res = as2.Agg21(n, nf, cache)

    if order == (2, 2):
        res = as2.Agg22(n, nf, cache)

    if order == (2, 3):
        res = as2.Agg23(n, nf, cache)

    return res

@nb.njit(cache=True)
def A_gq(n, nf, order, cache):
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
        :math:`A_{gq}`

    """

    if order == (1, -1):
        res = as1.Agq1m1(n, cache)

    if order == (1, 0):
        res = as1.Agq10(n, cache)

    if order == (1, 1):
        res = as1.Agq11(n, cache)
    
    if order == (2, -1):
        res = as2.Agq2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.Agq20(n, nf, cache)

    if order == (2, 1):
        res = as2.Agq21(n, nf, cache)

    if order == (2, 2):
        res = as2.Agq22(n, nf, cache)

    if order == (2, 3):
        res = as2.Agq23(n, nf, cache)

    return res

@nb.njit(cache=True)
def A_qg(n, nf, order, cache):
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
        res = as1.Aqg1m1(n, cache)

    if order == (1, 0):
        res = as1.Aqg10(n, cache)

    if order == (1, 1):
        res = as1.Aqg11(n, cache)

    if order == (2, -1):
        res = as2.Aqg2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.Aqg20(n, nf, cache)

    if order == (2, 1):
        res = as2.Aqg21(n, nf, cache)

    if order == (2, 2):
        res = as2.Aqg22(n, nf, cache)

    if order == (2, 3):
        res = as2.Aqg23(n, nf, cache)

    return res

@nb.njit(cache=True)
def A_qq(n, nf, order, cache):
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
        res = as1.Aqq1m1(n, cache)

    if order == (1, 0):
        res = as1.Aqq10(n, cache)

    if order == (1, 1):
        res = as1.Aqq11(n, cache)

    if order == (2, -1):
        res = as2.Aqq2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.Aqq20(n, nf, cache)

    if order == (2, 1):
        res = as2.Aqq21(n, nf, cache)

    if order == (2, 2):
        res = as2.Aqq22(n, nf, cache)

    if order == (2, 3):
        res = as2.Aqq23(n, nf, cache)

    return res

@nb.njit(cache=True)
def A_qQ2(n, nf, order, cache):
    if order == (1, -1):
        res = 0.0

    if order == (1, 0):
        res = 0.0

    if order == (1, 1):
        res = 0.0

    if order == (2, -1):
        res = as2.AqQ2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.AqQ20(n, nf, cache)

    if order == (2, 1):
        res = as2.AqQ21(n, nf, cache)

    if order == (2, 2):
        res = as2.AqQ22(n, nf, cache)

    if order == (2, 3):
        res = as2.AqQ23(n, nf, cache)

    return res


@nb.njit(cache=True)
def A_qQ2bar(n, nf, order, cache):
    if order == (1, -1):
        res = 0.0

    if order == (1, 0):
        res = 0.0

    if order == (1, 1):
        res = 0.0

    if order == (2, -1):
        res = as2.AqQbar2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.AqQbar20(n, nf, cache)

    if order == (2, 1):
        res = as2.AqQbar21(n, nf, cache)

    if order == (2, 2):
        res = as2.AqQbar22(n, nf, cache)

    if order == (2, 3):
        res = as2.AqQbar23(n, nf, cache)

    return res

@nb.njit(cache=True)
def A_qqbar(n, nf, order, cache):
    if order == (1, -1):
        res = 0.0

    if order == (1, 0):
        res = 0.0

    if order == (1, 1):
        res = 0.0

    if order == (2, -1):
        res = as2.Aqqbar2m1(n, nf, cache)

    if order == (2, 0):
        res = as2.Aqqbar20(n, nf, cache)

    if order == (2, 1):
        res = as2.Aqqbar21(n, nf, cache)

    if order == (2, 2):
        res = as2.Aqqbar22(n, nf, cache)

    if order == (2, 3):
        res = as2.Aqqbar23(n, nf, cache)

    return res


# @nb.njit(cache=True)
# def A_entries(n, nf, order, cache):
#     r"""Compute the beam function matching kernel at the given order.

#     Parameters
#     ----------
#     n : complex
#         Mellin moment
#     cache: numpy.ndarray
#         Harmonic sum cache

#     Returns
#     -------
#     numpy.ndarray
#         `

#     """

#     Agg = A_gg(n, nf, order, cache)
#     Agq = A_gq(n, nf, order, cache)
#     Aqg = A_qg(n, nf, order, cache)
#     Aqq = A_qq(n, nf, order, cache)
#     AqQ2 = A_qQ2(n, nf, order, cache)
#     Aqqbar = A_qqbar(n, nf, order, cache)
#     AqQ2bar = A_qQ2bar(n, nf, order, cache)

#     A_S = np.array(
#         [
#             [Agg, Agq, Agq, Agq, Agq],
#             [Aqg, Aqq, Aqqbar, AqQ2, AqQ2bar],
#             [Aqg, Aqqbar, Aqq, AqQ2bar, AqQ2],
#             [Aqg, AqQ2, AqQ2bar, Aqq, Aqqbar],
#             [Aqg, AqQ2bar, AqQ2bar, Aqqbar, Aqq],
#         ],
#         np.complex_,
#     )
#     return A_S


