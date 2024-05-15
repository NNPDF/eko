"""SCET 1 kernels"""

import numba as nb
import numpy as np

from ..harmonics import cache as c
from . import k_space as k
from . import tau_space as tau

@nb.njit(cache=True)
def A_entries(n, nf, order, space, cache):
    r"""Compute the beam function matching kernel at the given order.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        `

    """
    if space=='k':
        Agg = k.A_gg(n, nf, order, cache)
        Agq = k.A_gq(n, nf, order, cache)
        Aqg = k.A_qg(n, nf, order, cache)
        Aqq = k.A_qq(n, nf, order, cache)
        AqQ2 = k.A_qQ2(n, nf, order, cache)
        Aqqbar = k.A_qqbar(n, nf, order, cache)
        AqQ2bar = k.A_qQ2bar(n, nf, order, cache)

    if space=='tau':
        Agg = tau.A_gg(n, nf, order, cache)
        Agq = tau.A_gq(n, nf, order, cache)
        Aqg = tau.A_qg(n, nf, order, cache)
        Aqq = tau.A_qq(n, nf, order, cache)
        AqQ2 = tau.A_qQ2(n, nf, order, cache)
        Aqqbar = tau.A_qqbar(n, nf, order, cache)
        AqQ2bar = tau.A_qQ2bar(n, nf, order, cache)

    A_S = np.array(
        [
            [Agg, Agq, Agq, Agq, Agq],
            [Aqg, Aqq, Aqqbar, AqQ2, AqQ2bar],
            [Aqg, Aqqbar, Aqq, AqQ2bar, AqQ2],
            [Aqg, AqQ2, AqQ2bar, Aqq, Aqqbar],
            [Aqg, AqQ2bar, AqQ2bar, Aqqbar, Aqq],
        ],
        np.complex_,
    )
    return A_S


@nb.njit(cache=True)
def SCET_I_entry(order, space, nf, n):
    r"""Compute the tower of the singlet |OME|.

    Parameters
    ----------
    matching_order : tuple(int,int)
        perturbative matching order
    n : complex
        Mellin variable
    nf: int
        number of active flavor below threshold
    L : float
        :math:``\ln(\mu_F^2 / m_h^2)``

    Returns
    -------
    numpy.ndarray
        singlet |OME|

    """
    cache = c.reset()
    A = np.zeros((5, 5), np.complex_)
    A = k.A_entries(n, nf, order, space, cache)

    return A