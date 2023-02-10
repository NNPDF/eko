"""
This module defines the |OME| for the non-trivial matching conditions in the
|VFNS| evolution.
"""

import numba as nb
import numpy as np

from ....harmonics import cache as c
from . import as1, as2, as3


@nb.njit(cache=True)
def A_singlet(matching_order, n, nf, L, is_msbar):
    r"""
    Computes the tower of the singlet |OME|.

    Parameters
    ----------
        matching_order : tuple(int,int)
            perturbative matching_order
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
        A_singlet : numpy.ndarray
            singlet |OME|

    """
    A_s = np.zeros((matching_order[0], 3, 3), np.complex_)

    cache = c.reset()
    if matching_order[0] >= 1:
        A_s[0] = as1.A_singlet(n, L, cache, True)
    if matching_order[0] >= 2:
        A_s[1] = as2.A_singlet(n, L, cache, True, is_msbar)
    if matching_order[0] >= 3:
        A_s[2] = as3.A_singlet(n, nf, L, cache)
    return A_s


@nb.njit(cache=True)
def A_non_singlet(matching_order, n, nf, L):
    r"""
    Computes the tower of the non-singlet |OME|

    Parameters
    ----------
        matching_order : tuple(int,int)
            perturbative matching_order
        n : complex
            Mellin variable
        nf: int
            number of active flavor below threshold
        L : float
            :math:``\ln(\mu_F^2 / m_h^2)``

    Returns
    -------
        A_non_singlet : numpy.ndarray
            non-singlet |OME|

    """
    A_ns = np.zeros((matching_order[0], 2, 2), np.complex_)

    cache = c.reset()
    if matching_order[0] >= 1:
        A_ns[0] = as1.A_ns(n, L, cache, False)
    if matching_order[0] >= 2:
        A_ns[1] = as2.A_ns(n, L, cache, False)
    if matching_order[0] >= 3:
        A_ns[2] = as3.A_ns(n, nf, L, cache, False)
    return A_ns
