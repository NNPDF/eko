# -*- coding: utf-8 -*-
r"""This module contains the scale variation operator for the expanded scheme (``ModSV=expanded``).

The expressions can be obtained using Eqs. (3.33) and (3.38) of :cite:`AbdulKhalek:2019ihb`.
Note, however, that our definition of the anomalous dimensions :math:`\gamma`
includes a further minus sign with resepect to :cite:`AbdulKhalek:2019ihb`, as well as
our definition of the coefficients of the beta function :math:`\beta_k`
(compare Eq. (3.3) of :cite:`AbdulKhalek:2019ihb`).
This effectively introduces a minus signs on terms which include a odd number of :math:`\gamma_j` and :math:`\beta_k` .
"""


import numba as nb
import numpy as np

from .. import beta


@nb.njit(cache=True)
def gamma_1_variation(gamma, L):
    r"""Computes the |NLO| anomalous dimension variation.

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    L : float
        logarithmic ratio of factorization and renormalization scale

    Returns
    -------
    gamma_1 : complex
        variation to :math:`\gamma^{(1)}`
    """
    return L * gamma[0]


@nb.njit(cache=True)
def gamma_2_variation(gamma, L, beta0, g0e2):
    r"""Computes the |NNLO| anomalous dimension variation.

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    L : float
        logarithmic ratio of factorization and renormalization scale
    beta0: float
        :math:`\beta_0`
    g0e2: complex or numpy.ndarray
        :math:`\left(\gamma^{(0)}\right)^2`

    Returns
    -------
    gamma_2 : complex
        variation to :math:`\gamma^{(2)}`
    """
    return gamma[1] * L + 1.0 / 2.0 * (beta0 * gamma[0] + g0e2) * L**2


@nb.njit(cache=True)
def gamma_3_variation(gamma, L, beta0, beta1, g0e2, g0e3, g1g0, g0g1):
    r"""Computes the |N3LO| anomalous dimension variation.

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    L : float
        logarithmic ratio of factorization and renormalization scale
    beta0: float
        :math:`\beta_0`
    beta0: float
        :math:`\beta_1`
    g0e2: complex or numpy.ndarray
        :math:`\left(\gamma^{(0)}\right)^2`
    g0e3: complex or numpy.ndarray
        :math:`\left(\gamma^{(0)}\right)^3`
    g1g0: complex or numpy.ndarray
        :math:`\gamma^{(1)} \gamma^{(0)}`
    g0g1: complex or numpy.ndarray
        :math:`\gamma^{(0)} \gamma^{(1)} `

    Returns
    -------
    gamma_3 : complex
        variation to :math:`\gamma^{(3)}`
    """
    return (
        gamma[2] * L
        + (1.0 / 2.0)
        * (beta1 * gamma[0] + 2.0 * beta0 * gamma[1] + g1g0 + g0g1)
        * L**2
        + (1.0 / 6.0)
        * (2.0 * beta0**2 * gamma[0] + 3.0 * beta0 * g0e2 + g0e3)
        * L**3
    )


@nb.njit(cache=True)
def non_singlet_variation(gamma, a_s, order, nf, L):
    """Non-singlet scale variation dispatcher.

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    a_s :  float
        target coupling value
    order : int
        perturbation order
    nf : int
        number of active flavors
    L : float
        logarithmic ratio of factorization and renormalization scale

    Returns
    -------
    sv_ker : numpy.ndarray
        scale varion kernel
    """
    sv_ker = 1.0
    if order >= 1:
        sv_ker += a_s * gamma_1_variation(gamma, L)
    if order >= 2:
        beta0 = beta.beta_0(nf)
        sv_ker += a_s**2 * gamma_2_variation(gamma, L, beta0, gamma[0] ** 2)
    if order >= 3:
        beta1 = beta.beta(1, nf)
        g0g1 = gamma[0] * gamma[1]
        sv_ker += a_s**3 * gamma_3_variation(
            gamma, L, beta0, beta1, gamma[0] ** 2, gamma[0] ** 3, g0g1, g0g1
        )
    return sv_ker


@nb.njit(cache=True)
def singlet_variation(gamma, a_s, order, nf, L):
    """Singlet scale cariation dispatcher

    Parameters
    ----------
    gamma : numpy.ndarray
        anomalous dimensions
    a_s :  float
        target coupling value
    order : int
        perturbation order
    nf : int
        number of active flavors
    L : float
        logarithmic ratio of factorization and renormalization scale

    Returns
    -------
    sv_ker : numpy.ndarray
        scale varion kernel
    """
    sv_ker = np.eye(2, dtype=np.complex_)
    gamma = np.ascontiguousarray(gamma)
    if order >= 1:
        sv_ker += a_s * gamma_1_variation(gamma, L)
    if order >= 2:
        beta0 = beta.beta_0(nf)
        gamma0e2 = gamma[0] @ gamma[0]
        sv_ker += a_s**2 * gamma_2_variation(gamma, L, beta0, gamma0e2)
    if order >= 3:
        beta1 = beta.beta(1, nf)
        gamma0e3 = gamma0e2 @ gamma[0]
        # here the product is not commutative
        g1g0 = gamma[1] @ gamma[0]
        g0g1 = gamma[0] @ gamma[1]
        sv_ker += a_s**3 * gamma_3_variation(
            gamma, L, beta0, beta1, gamma0e2, gamma0e3, g1g0, g0g1
        )
    return sv_ker
