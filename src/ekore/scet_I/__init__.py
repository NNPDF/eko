"""SCET 1 kernels"""

import numba as nb
import numpy as np

from ..harmonics import cache as c
from .k_space import as1 as k_as1
from .tau_space import as1 as tau_as1

@nb.njit(cache=True)
def SCET_I_entry(order, space, n):
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
    if space=='k':
        A = k_as1.A_entries(n, order, cache)
    if space=='tau':
        A = tau_as1.A_entries(n, order, cache)

    return A