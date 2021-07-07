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

import numba as nb
import numpy as np

from . import harmonics, lo, nlo, nnlo


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
        eko.anomalous_dimensions.nnlo.gamma_singlet_2 : :math:`\gamma_{S}^{(2)}(N)`
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
        eko.anomalous_dimensions.nnlo.gamma_nsp_2 : :math:`\gamma_{ns,+}^{(2)}(N)`
        eko.anomalous_dimensions.nnlo.gamma_nsm_2 : :math:`\gamma_{ns,-}^{(2)}(N)`
        eko.anomalous_dimensions.nnlo.gamma_nsv_2 : :math:`\gamma_{ns,v}^{(2)}(N)`
    """
    # cache the s-es
    sx = np.full(1, harmonics.harmonic_S1(n))
    # now combine
    gamma_ns = np.zeros(order + 1, np.complex_)
    gamma_ns[0] = lo.gamma_ns_0(n, sx[0])
    # NLO and beyond
    if order >= 1:
        # TODO: pass the necessary harmonics to nlo gammas
        if mode == "p":
            gamma_ns_1 = nlo.gamma_nsp_1(n, nf)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in ["m", "v"]:
            gamma_ns_1 = nlo.gamma_nsm_1(n, nf)
        gamma_ns[1] = gamma_ns_1
    # NNLO and beyond
    if order >= 2:
        sx = np.append(sx, harmonics.harmonic_S2(n))
        sx = np.append(sx, harmonics.harmonic_S3(n))
        if mode == "p":
            gamma_ns_2 = -nnlo.gamma_nsp_2(n, nf, sx)
        elif mode == "m":
            gamma_ns_2 = -nnlo.gamma_nsm_2(n, nf, sx)
        elif mode == "v":
            gamma_ns_2 = -nnlo.gamma_nsv_2(n, nf, sx)
        gamma_ns[2] = gamma_ns_2
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
        eko.anomalous_dimensions.nnlo.gamma_singlet_2 : :math:`\gamma_{S}^{(2)}(N)`
    """
    # cache the s-es
    sx = np.full(1, harmonics.harmonic_S1(n))
    if order >= 1:
        sx = np.append(sx, harmonics.harmonic_S2(n))
        sx = np.append(sx, harmonics.harmonic_S3(n))

    gamma_singlet = np.zeros((order + 1, 2, 2), np.complex_)
    gamma_singlet[0] = lo.gamma_singlet_0(n, sx[0], nf)
    if order >= 1:
        gamma_singlet[1] = nlo.gamma_singlet_1(n, nf)
    if order == 2:
        sx = np.append(sx, harmonics.harmonic_S4(n))
        gamma_singlet[2] = -nnlo.gamma_singlet_2(n, nf, sx)
    return gamma_singlet
