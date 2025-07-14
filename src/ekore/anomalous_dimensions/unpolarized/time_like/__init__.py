"""The unpolarized, time-like Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from ....harmonics import cache as c
from . import as1, as2, as3


@nb.njit(cache=True)
def gamma_ns(order, mode, n, nf):
    r"""Compute the tower of the non-singlet anomalous dimensions.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    mode : 10201 | 10101 | 10200
        sector identifier
    n : complex
        Mellin variable
    nf : int
        Number of active flavors

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions
    """
    cache = c.reset()
    gamma_ns = np.zeros(order[0], np.complex128)
    gamma_ns[0] = as1.gamma_ns(n, cache)
    if order[0] >= 2:
        gamma_ns_1 = 0.0
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, cache)
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, cache)
        gamma_ns[1] = gamma_ns_1
    if order[0] >= 3:
        gamma_ns_2 = 0.0
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf, cache)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf, cache)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf, cache)
        gamma_ns[2] = gamma_ns_2
    return gamma_ns


@nb.njit(cache=True)
def gamma_singlet(order, n, nf):
    r"""Compute the tower of the singlet anomalous dimensions' matrices.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    n : complex
        Mellin variable
    nf : int
        Number of active flavors

    Returns
    -------
    numpy.ndarray
        singlet anomalous dimensions matrices
    """
    cache = c.reset()
    gamma_s = np.zeros((order[0], 2, 2), np.complex128)
    gamma_s[0] = as1.gamma_singlet(n, nf, cache)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf, cache)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf, cache)
    return gamma_s
