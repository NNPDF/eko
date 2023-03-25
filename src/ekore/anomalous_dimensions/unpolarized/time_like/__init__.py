"""The unpolarized time-like Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

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
    gamma_ns = np.zeros(order[0], np.complex_)
    gamma_ns[0] = as1.gamma_ns(n)
    if order[0] >= 2:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf)
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf)
        gamma_ns[1] = gamma_ns_1
    if order[0] >= 3:
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf)
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
    gamma_s = np.zeros((order[0], 2, 2), np.complex_)
    gamma_s[0] = as1.gamma_singlet(n, nf)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf)
    return gamma_s
