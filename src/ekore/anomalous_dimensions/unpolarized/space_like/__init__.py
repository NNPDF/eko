r"""The unpolarized, space-like Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
The 3-loop references for the non-singlet :cite:`Moch:2004pa`
and singlet :cite:`Vogt:2004mw` case contain also the lower
order results. The results are also determined in Mellin space in
terms of the anomalous dimensions (note the additional sign!)

.. math::
    \gamma(N) = - \mathcal{M}[\mathbf{P}(x)](N)
"""

import numba as nb
import numpy as np

from .... import harmonics
from ....harmonics import cache as c
from . import aem1, aem2, as1, as2, as3, as4


@nb.njit(cache=True)
def gamma_ns(order, mode, n, nf):
    r"""Computes the tower of the non-singlet anomalous dimensions

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

    cache = c.reset()
    gamma_ns[0] = as1.gamma_ns(n, cache)
    # NLO and beyond
    if order[0] >= 2:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, cache, False)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, cache, False)
        else:
            raise NotImplementedError("Non-singlet sector is not implemented")
        gamma_ns[1] = gamma_ns_1
    # NNLO and beyond
    if order[0] >= 3:
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf, cache, False)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf, cache, False)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf, cache, False)
        gamma_ns[2] = gamma_ns_2
    # N3LO
    if order[0] >= 4:
        if mode == 10101:
            gamma_ns_3 = as4.gamma_nsp(n, nf, cache, False)
        elif mode == 10201:
            gamma_ns_3 = as4.gamma_nsm(n, nf, cache, False)
        elif mode == 10200:
            gamma_ns_3 = as4.gamma_nsv(n, nf, cache, False)
        gamma_ns[3] = gamma_ns_3
    return gamma_ns


@nb.njit(cache=True)
def gamma_singlet(order, n, nf):
    r"""Computes the tower of the singlet anomalous dimensions matrices

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
    gamma_s = np.zeros((order[0], 2, 2), np.complex_)
    gamma_s[0] = as1.gamma_singlet(n, nf, cache)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf, cache, True)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf, cache, True)
    if order[0] >= 4:
        gamma_s[3] = as4.gamma_singlet(n, nf, cache, True)
    return gamma_s
