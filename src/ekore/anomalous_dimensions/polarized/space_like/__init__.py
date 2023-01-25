r"""This module contains the polarized spacelike Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
"""

import numba as nb
import numpy as np

from .... import harmonics
from . import as1, as2, as3


def compute_cache(n, pto, is_singlet):
    """Compute the harmonic cache for polarized anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin variable
    pto : int
        perturbative order
    is_singlet: bool
        True for singlet like quantities

    Returns
    -------
    list
        harmonics cache

    """
    max_weight = pto if pto != 3 else 4
    cache = harmonics.sx(n, max_weight)
    # TODO: fix this cache to contain all the harmonics needed
    # if pto == 1:
    #     return [harmonics.S1(n)]
    # if pto == 2:
    #     sx = harmonics.sx(n, max_weight=2)
    #     cache = [sx[0], [sx[1]]]
    #     if is_singlet:
    #         # S1, S2, Sm21
    #         Sm1 = harmonics.Sm1(n, cache[0], is_singlet)
    #         Sm21 = harmonics.Sm21(n, cache[0], Sm1, is_singlet)
    #         cache.append([0, Sm21])
    # if pto == 3:
    #     if is_singlet:
    #         # S1, S2, S3, Sm21, S4
    #         sx = harmonics.sx(n, max_weight=4)
    #         Sm1 = harmonics.Sm1(n, sx[0], is_singlet)
    #         Sm21 = harmonics.Sm21(n, sx[0], Sm1, is_singlet)
    #         cache = [sx[0], [sx[1], 0], [sx[2], Sm21], sx[3]]
    #     else:
    #         # TODO: this ordering will not be good when calling nsm,nsp...
    #         # S1, S2,Sm2, S3, Sm21, Sm3
    #         sx = harmonics.sx(n, max_weight=3)
    #         smx = harmonics.smx(n, 3, is_singlet)
    #         Sm21 = harmonics.Sm21(n, sx[0], smx[0], is_singlet)
    #         cache = [sx[0], [sx[1], smx[1]], [sx[2], Sm21, smx[2]]]
    return cache


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
    # cache the s-es
    sx = compute_cache(n, order + 1, False)
    # now combine
    gamma_ns = np.zeros(order[0], np.complex_)
    gamma_ns[0] = as1.gamma_ns(n, sx[0])
    # NLO and beyond
    if order[0] >= 2:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, sx)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, sx)
        else:
            raise NotImplementedError("Non-singlet sector is not implemented")
        gamma_ns[1] = gamma_ns_1
    if order[0] >= 3:
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf, sx)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf, sx)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf, sx)
        gamma_ns[2] = gamma_ns_2
    if order[0] >= 4:
        raise NotImplementedError("Polarized beyond NNLO is not yet implemented")
    return gamma_ns


@nb.njit(cache=True)
def gamma_singlet(order, n, nf):
    r"""Compute the tower of the singlet anomalous dimensions matrices.

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
    # cache the s-es
    sx = compute_cache(n, order + 1, True)

    gamma_s = np.zeros((order[0], 2, 2), np.complex_)
    gamma_s[0] = as1.gamma_singlet(n, sx[0], nf)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf, sx)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf, sx)
    if order[0] >= 4:
        raise NotImplementedError("Polarized beyond NNLO is not yet implemented")
    return gamma_s
