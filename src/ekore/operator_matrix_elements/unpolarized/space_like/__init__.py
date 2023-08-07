"""The unpolarized, space-like |OME|."""

import numba as nb
import numpy as np

from ....harmonics import cache as c
from . import as1, as2, as3


@nb.njit(cache=True)
def A_singlet(matching_order, n, nf, L, is_msbar):
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
    is_msbar: bool
        add the |MSbar| contribution

    Returns
    -------
    numpy.ndarray
        singlet |OME|

    """
    cache = c.reset()
    A_s = np.zeros((matching_order[0], 3, 3), np.complex_)
    if matching_order[0] >= 1:
        A_s[0] = as1.A_singlet(n, cache, L)
    if matching_order[0] >= 2:
        A_s[1] = as2.A_singlet(n, cache, L, is_msbar)
    if matching_order[0] >= 3:
        A_s[2] = as3.A_singlet(n, cache, nf, L)
    return A_s


@nb.njit(cache=True)
def A_non_singlet(matching_order, n, nf, L):
    r"""Compute the tower of the non-singlet |OME|.

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
        non-singlet |OME|

    """
    cache = c.reset()
    A_ns = np.zeros((matching_order[0], 2, 2), np.complex_)
    if matching_order[0] >= 1:
        A_ns[0] = as1.A_ns(n, cache, L)
    if matching_order[0] >= 2:
        A_ns[1] = as2.A_ns(n, cache, L)
    if matching_order[0] >= 3:
        A_ns[2] = as3.A_ns(n, cache, nf, L)
    return A_ns
