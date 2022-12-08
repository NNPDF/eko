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

from .. import basis_rotation as br
from .. import harmonics
from . import aem1, aem2, as1, as1aem1, as2, as3, as4, asp1


@nb.njit(cache=True)
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
        eko.anomalous_dimensions.as1.gamma_singlet : :math:`\gamma_{S}^{(0)}(N)`
        eko.anomalous_dimensions.as2.gamma_singlet : :math:`\gamma_{S}^{(1)}(N)`
        eko.anomalous_dimensions.as3.gamma_singlet : :math:`\gamma_{S}^{(2)}(N)`
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


@nb.njit(cache=True)
def gamma_ns(order, mode, n, nf, p):
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
    p : Boolean
        Polarised (True) or un-Polarised (False)

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions

    See Also
    --------
    eko.anomalous_dimensions.as1.gamma_ns : :math:`\gamma_{ns}^{(0)}(N)`
    eko.anomalous_dimensions.as2.gamma_nsp : :math:`\gamma_{ns,+}^{(1)}(N)`
    eko.anomalous_dimensions.as2.gamma_nsm : :math:`\gamma_{ns,-}^{(1)}(N)`
    eko.anomalous_dimensions.as3.gamma_nsp : :math:`\gamma_{ns,+}^{(2)}(N)`
    eko.anomalous_dimensions.as3.gamma_nsm : :math:`\gamma_{ns,-}^{(2)}(N)`
    eko.anomalous_dimensions.as3.gamma_nsv : :math:`\gamma_{ns,v}^{(2)}(N)`
    eko.anomalous_dimensions.as4.gamma_nsp : :math:`\gamma_{ns,+}^{(3)}(N)`
    eko.anomalous_dimensions.as4.gamma_nsm : :math:`\gamma_{ns,-}^{(3)}(N)`
    eko.anomalous_dimensions.as4.gamma_nsv : :math:`\gamma_{ns,v}^{(3)}(N)`

    """
    # cache the s-es
    if order[0] >= 4:
        full_sx_cache = harmonics.compute_cache(n, 5, is_singlet=False)
        sx = np.array(
            [
                full_sx_cache[0][0],
                full_sx_cache[1][0],
                full_sx_cache[2][0],
                full_sx_cache[3][0],
            ]
        )
    else:
        sx = harmonics.sx(n, max_weight=order[0] + 1)
    # now combine
    gamma_ns = np.zeros(order[0], np.complex_)
    if p == False:
        gamma_ns[0] = as1.gamma_ns(n, sx[0])
    elif p == True:
        gamma_ns[0] = asp1.gamma_pns(n, sx[0])
    # NLO and beyondd
    if order[0] >= 2 and p == False:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, sx)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, sx)
        else:
            raise NotImplementedError("Non-singlet sector is not implemented")
        gamma_ns[1] = gamma_ns_1

    elif order[0] >= 2 and p == True:
        raise Exception("Polarised case is not known at this order")

    # NNLO and beyond
    if order[0] >= 3 and p == False:
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf, sx)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf, sx)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf, sx)
        gamma_ns[2] = gamma_ns_2

    elif order[0] >= 3 and p == True:
        raise Exception("Polarised case is not known at this order")

    # N3LO
    if order[0] >= 4 and p == False:
        if mode == 10101:
            gamma_ns_3 = as4.gamma_nsp(n, nf, full_sx_cache)
        elif mode == 10201:
            gamma_ns_3 = as4.gamma_nsm(n, nf, full_sx_cache)
        elif mode == 10200:
            gamma_ns_3 = as4.gamma_nsv(n, nf, full_sx_cache)
        gamma_ns[3] = gamma_ns_3

    elif order[0] >= 4 and p == True:
        raise Exception("Polarised case is not known at this order")

    return gamma_ns


@nb.njit(cache=True)
def gamma_singlet(order, n, nf, p):
    r"""Computes the tower of the singlet anomalous dimensions matrices

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    p : Boolean
        Polarised (True) or un-Polarised (False)
    Returns
    -------
    numpy.ndarray
        singlet anomalous dimensions matrices

    See Also
    --------
    eko.anomalous_dimensions.as1.gamma_singlet : :math:`\gamma_{S}^{(0)}(N)`
    eko.anomalous_dimensions.as2.gamma_singlet : :math:`\gamma_{S}^{(1)}(N)`
    eko.anomalous_dimensions.as3.gamma_singlet : :math:`\gamma_{S}^{(2)}(N)`
    eko.anomalous_dimensions.as4.gamma_singlet : :math:`\gamma_{S}^{(3)}(N)`

    """
    # cache the s-es
    if order[0] >= 4:
        full_sx_cache = harmonics.compute_cache(n, 5, is_singlet=False)
        sx = np.array(
            [
                full_sx_cache[0][0],
                full_sx_cache[1][0],
                full_sx_cache[2][0],
                full_sx_cache[3][0],
            ]
        )
    elif order[0] >= 3:
        # here we need only S1,S2,S3,S4
        sx = harmonics.sx(n, max_weight=order[0] + 1)
    else:
        sx = harmonics.sx(n, max_weight=order[0])

    gamma_s = np.zeros((order[0], 2, 2), np.complex_)

    if p == False:
        gamma_s[0] = as1.gamma_singlet(n, sx[0], nf)
    elif p == True:
        gamma_s[0] = asp1.gamma_psinglet(n, sx[0], nf)

    if order[0] >= 2 and p == False:
        gamma_s[1] = as2.gamma_singlet(n, nf, sx)
    elif order[0] >= 2 and p == True:
        raise Exception("Polarised case is not known at this order")

    if order[0] >= 3 and p == False:
        gamma_s[2] = as3.gamma_singlet(n, nf, sx)
    elif order[0] >= 3 and p == True:
        raise Exception("Polarised case is not known at this order")

    if order[0] >= 4 and p == False:
        gamma_s[3] = as4.gamma_singlet(n, nf, full_sx_cache)
    elif order[0] >= 4 and p == True:
        raise Exception("Polarised case is not known at this order")

    return gamma_s
