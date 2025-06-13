r"""The polarized, space-like Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
"""

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
    # NLO and beyond
    if order[0] >= 2:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, cache)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, cache)
        else:
            raise NotImplementedError("Non-singlet sector is not implemented")
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
    cache = c.reset()
    gamma_s = np.zeros((order[0], 2, 2), np.complex128)
    gamma_s[0] = as1.gamma_singlet(n, cache, nf)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf, cache)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf, cache)
    if order[0] >= 4:
        raise NotImplementedError("Polarized beyond NNLO is not yet implemented")
    return gamma_s
