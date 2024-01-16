"""SCET 1 kernels"""

import numba as nb
import numpy as np

from ..harmonics import cache as c
from . import as1

@nb.njit(cache=True)
def SCET_I_entry(order, n):
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
    A = as1.A_entries(n, order, cache)
    return A