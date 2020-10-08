# -*- coding: utf-8 -*-
"""
Colletion of non-singlet EKOs.
"""

import numpy as np

import numba as nb

from .. import beta

from . import evolution_integrals as ei
from . import utils


@nb.njit("c16(c16[:],f8,f8,u1)", cache=True)
def lo_exact(gamma_ns, a1, a0, nf):
    """
    Non-singlet leading order exact EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        e_ns^0 : complex
            non-singlet leading order exact EKO
    """
    return np.exp(gamma_ns[0] * ei.j00(a1, a0, nf))


@nb.njit("c16(c16[:],f8,f8,u1)", cache=True)
def nlo_exact(gamma_ns, a1, a0, nf):
    """
    Non-singlet next-to-leading order exact EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order exact EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j01_exact(a1, a0, nf) + gamma_ns[1] * ei.j11_exact(a1, a0, nf)
    )


@nb.njit("c16(c16[:],f8,f8,u1)", cache=True)
def nlo_expanded(gamma_ns, a1, a0, nf):
    """
    Non-singlet next-to-leading order expanded EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order expanded EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j01_expanded(a1, a0, nf)
        + gamma_ns[1] * ei.j11_expanded(a1, a0, nf)
    )


@nb.njit("c16(c16[:],f8,f8,u1,u4)", cache=True)
def nlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    Non-singlet next-to-leading order truncated EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    b1 = beta.b(1, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (1.0 + ei.j11_expanded(ah, al, nf) * (gamma_ns[1] - b1 * gamma_ns[0]))
        al = ah
    return e


@nb.njit("c16(c16[:],f8,f8,u1,u4)", cache=True)
def nlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    Non-singlet next-to-leading order ordered-truncated EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order ordered-truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    beta0 = beta.beta(0, nf)
    b1 = beta.b(1, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= (
            e0
            * (1.0 + ah / beta0 * (gamma_ns[1] - b1 * gamma_ns[0]))
            / (1.0 + al / beta0 * (gamma_ns[1] - b1 * gamma_ns[0]))
        )
        al = ah
    return e


@nb.njit("c16(u1,string,c16[:],f8,f8,u1,u4)", cache=True)
def dispatcher(order, method, gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
        order : int
            perturbation order
        method : str
            method
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_ns : complex
            non-singlet EKO
    """
    # use always exact in LO
    if order == 0:
        return lo_exact(gamma_ns, a1, a0, nf)
    # NLO
    if method in ["iterate-expanded", "decompose-expanded", "perturbative-expanded"]:
        return nlo_expanded(gamma_ns, a1, a0, nf)
    elif method == "truncated":
        return nlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
    elif method == "ordered-truncated":
        return nlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
    # if method in ["iterate-exact", "decompose-exact", "perturbative-exact"]:
    return nlo_exact(gamma_ns, a1, a0, nf)
