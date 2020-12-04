# -*- coding: utf-8 -*-
r"""
This module contains the Altarelli-Parisi splitting kernels.

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

import numpy as np

import numba as nb

from .harmonics import harmonic_S1 as S1

from . import lo
from . import nlo


@nb.njit("Tuple((c16[:,:],c16,c16,c16[:,:],c16[:,:]))(c16[:,:])", cache=True)
def exp_singlet(gamma_S):
    r"""
    Computes the exponential and the eigensystem of the singlet anomalous dimension matrix

    Parameters
    ----------
        gamma_S : numpy.ndarray
            singlet anomalous dimension matrix

    Returns
    -------
        exp : numpy.ndarray
            exponential of the singlet anomalous dimension matrix :math:`\gamma_{S}(N)`
        lambda_p : complex
            positive eigenvalue of the singlet anomalous dimension matrix
            :math:`\gamma_{S}(N)`
        lambda_m : complex
            negative eigenvalue of the singlet anomalous dimension matrix
            :math:`\gamma_{S}(N)`
        e_p : numpy.ndarray
            projector for the positive eigenvalue of the singlet anomalous
            dimension matrix :math:`\gamma_{S}(N)`
        e_m : numpy.ndarray
            projector for the negative eigenvalue of the singlet anomalous
            dimension matrix :math:`\gamma_{S}(N)`

    See Also
    --------
        eko.anomalous_dimensions.lo.gamma_singlet_0 : :math:`\gamma_{S}^{(0)}(N)`
        eko.anomalous_dimensions.nlo.gamma_singlet_1 : :math:`\gamma_{S}^{(1)}(N)`
    """
    # compute eigenvalues
    det = np.sqrt(
        np.power(gamma_S[0, 0] - gamma_S[1, 1], 2) + 4.0 * gamma_S[0, 1] * gamma_S[1, 0]
    )
    lambda_p = 1.0 / 2.0 * (gamma_S[0, 0] + gamma_S[1, 1] + det)
    lambda_m = 1.0 / 2.0 * (gamma_S[0, 0] + gamma_S[1, 1] - det)
    # compute projectors
    identity = np.identity(2, np.complex_)
    c = 1.0 / det
    e_p = +c * (gamma_S - lambda_m * identity)
    e_m = -c * (gamma_S - lambda_p * identity)
    exp = e_m * np.exp(lambda_m) + e_p * np.exp(lambda_p)
    return exp, lambda_p, lambda_m, e_p, e_m


@nb.njit("c16[:](u1,string,c16,u1)", cache=True)
def gamma_ns(order, mode, n, nf):
    r"""
    Computes the tower of the non-singlet anomalous dimensions

    Parameters
    ----------
        order : int
            perturbative order
        mode : "m" | "p" | "v"
            sector identifier
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions

    See Also
    --------
        eko.anomalous_dimensions.lo.gamma_ns_0 : :math:`\gamma_{ns}^{(0)}(N)`
        eko.anomalous_dimensions.nlo.gamma_nsp_1 : :math:`\gamma_{ns,+}^{(1)}(N)`
        eko.anomalous_dimensions.nlo.gamma_nsm_1 : :math:`\gamma_{ns,-}^{(1)}(N)`
    """
    # cache the s-es
    s1 = S1(n)
    gamma_ns = np.zeros(order + 1, np.complex_)
    gamma_ns[0] = lo.gamma_ns_0(n, s1)
    if order > 0:
        if mode == "p":
            gamma_ns_1 = nlo.gamma_nsp_1(n, nf)
        elif mode == "m":
            gamma_ns_1 = nlo.gamma_nsm_1(n, nf)
        gamma_ns[1] = gamma_ns_1
    return gamma_ns


@nb.njit("c16[:,:,:](u1,c16,u1)", cache=True)
def gamma_singlet(order, n, nf):
    r"""
    Computes the tower of the singlet anomalous dimensions matrices

    Parameters
    ----------
        order : int
            perturbative order
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_singlet : numpy.ndarray
            singlet anomalous dimensions matrices

    See Also
    --------
        eko.anomalous_dimensions.lo.gamma_singlet_0 : :math:`\gamma_{S}^{(0)}(N)`
        eko.anomalous_dimensions.nlo.gamma_singlet_1 : :math:`\gamma_{S}^{(1)}(N)`
    """
    # cache the s-es
    s1 = S1(n)
    gamma_singlet = np.zeros((order + 1, 2, 2), np.complex_)
    gamma_singlet[0] = lo.gamma_singlet_0(n, s1, nf)
    if order > 0:
        gamma_singlet[1] = nlo.gamma_singlet_1(n, nf)
    return gamma_singlet
