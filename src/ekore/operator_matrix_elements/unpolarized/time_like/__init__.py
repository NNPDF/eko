r"""The unpolarized, time-like |OME|."""

import numba as nb
import numpy as np

from . import as1


@nb.njit(cache=True)
def A_non_singlet(matching_order, _N, _L):
    r"""Compute the non-singlet |OME|.

    Parameters
    ----------
    matching_order : tuple(int, int)
        perturbative matching order

    Returns
    -------
    numpy.ndarray
        non-singlet |OME|
    """
    A_ns = np.zeros((matching_order[0], 2, 2), np.complex128)
    A_ns[0] = as1.A_ns()
    return A_ns


@nb.njit(cache=True)
def A_singlet(matching_order, N, L):
    r"""Compute the singlet |OME|.

    Parameters
    ----------
    matching_order : tuple(int, int)
        perturbative matching order
    N : complex
        Mellin moment
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    numpy.ndarray
        singlet |OME|
    """
    A_s = np.zeros((matching_order[0], 3, 3), np.complex128)
    A_s[0] = as1.A_singlet(N, L)
    return A_s
