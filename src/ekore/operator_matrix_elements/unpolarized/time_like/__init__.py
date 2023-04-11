r"""The unpolarized, time-like |OME|."""

import numba as nb
import numpy as np

from . import as1


@nb.njit(cache=True)
def A_non_singlet(matching_order, N, L):
    r"""Compute the non-singlet |OME|.

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
    A_non_singlet : numpy.ndarray
        non-singlet |OME|

    """
    if matching_order[0] > 1:
        raise Exception("Time-like matching conditions are only known upto NLO")
    A_ns = np.zeros((matching_order[0], 2, 2), np.complex_)
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
    A_singlet : numpy.ndarray
        singlet |OME|

    """
    if matching_order[0] > 1:
        raise Exception("Time-like matching conditions are only known upto NLO")
    A_singlet = np.zeros((matching_order[0], 3, 3), np.complex_)
    A_singlet[0] = as1.A_singlet(N, L)
    return A_singlet
