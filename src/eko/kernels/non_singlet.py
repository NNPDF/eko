# -*- coding: utf-8 -*-

import numpy as np

import numba as nb

from .. import strong_coupling as sc

from . import evolution_integrals as ei
from . import utils


@nb.njit
def lo_exact(gamma_ns, a1, a0, nf):
    """
    Non-singlet leading order exact EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions

    Returns
    -------
        e_ns^0 : complex
            non-singlet leading order exact EKO
    """
    return np.exp(gamma_ns[0] * ei.j00(a1, a0, nf))


@nb.njit
def nlo_exact(gamma_ns, a1, a0, nf):
    """
    Non-singlet next-to-leading order exact EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order exact EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j01_exact(a1, a0, nf) + gamma_ns[1] * ei.j11_exact(a1, a0, nf)
    )


@nb.njit
def nlo_expanded(gamma_ns, a1, a0, nf):
    """
    Non-singlet next-to-leading order expanded EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order expanded EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j01_expanded(a1, a0, nf)
        + gamma_ns[1] * ei.j11_expanded(a1, a0, nf)
    )


@nb.njit
def nlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    Non-singlet next-to-leading order truncated EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    b1 = sc.b(1, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (1.0 + ei.j11_expanded(ah, al, nf) * (gamma_ns[1] - b1 * gamma_ns[0]))
        al = ah
    return e


def nlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    Non-singlet next-to-leading order ordered-truncated EKO

    Parameters
    ----------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions

    Returns
    -------
        e_ns^1 : complex
            non-singlet next-to-leading order ordered-truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    beta0 = sc.beta(0, nf)
    b1 = sc.b(1, nf)
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


def dispatcher_lo(_method):
    return lo_exact


def dispatcher_nlo(method):
    if method in ["iterate-exact", "decompose-exact", "perturbative-exact"]:
        return nlo_exact
    if method in ["iterate-expanded", "decompose-expanded", "perturbative-expanded"]:
        return nlo_expanded
    if method == "truncated":
        return nlo_truncated
    if method == "ordered-truncated":
        return nlo_ordered_truncated
    raise ValueError(f"Unknown method: {method}")
