# -*- coding: utf-8 -*-

import numpy as np

import numba as nb

from eko import strong_coupling as sc
import eko.anomalous_dimensions as ad
from . import evolution_integrals as ei

@nb.njit
def lo_exact(gamma_singlet, a1, a0, nf):
    """
    Singlet leading order exact EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^0 : float
            singlet leading order exact EKO
    """
    return ad.exp_singlet(gamma_singlet[0] * ei.j00(a1, a0, nf))[0]


@nb.njit
def nlo_decompose_exact(gamma_singlet, a1, a0, nf):
    """
    Singlet next-to-leading order "exact decomposed" EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^1 : complex
            non-singlet next-to-leading order "exact decomposed" EKO
    """
    return ad.exp_singlet(
        gamma_singlet[0] * ei.j01_exact(a1, a0, nf)
        + gamma_singlet[1] * ei.j11_exact(a1, a0, nf)
    )[0]


@nb.njit
def nlo_decompose_expanded(gamma_singlet, a1, a0, nf):
    """
    Singlet next-to-leading order "expanded decomposed" EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^1 : complex
            non-singlet next-to-leading order "expanded decomposed" EKO
    """
    return ad.exp_singlet(
        gamma_singlet[0] * ei.j01_expanded(a1, a0, nf)
        + gamma_singlet[1] * ei.j11_expanded(a1, a0, nf)
    )[0]


@nb.njit
def nlo_iterate(gamma_singlet, a1, a0, nf, ev_op_iterations):
    """
    Singlet next-to-leading order iterated (exact) EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^1 : complex
            non-singlet next-to-leading order iterated (exact) EKO
    """
    a_steps = np.linspace(a0, a1, ev_op_iterations)
    beta0 = sc.beta(0, nf)
    beta1 = sc.beta(1, nf)
    e = np.identity(2, np.complex_)
    al = a_steps[0]
    for ah in a_steps[1:]:
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        ln = (
            (gamma_singlet[0] * a_half + gamma_singlet[1] * a_half ** 2)
            / (beta0 * a_half ** 2 + beta1 * a_half ** 3)
            * delta_a
        )
        ek = ad.exp_singlet(ln)[0]
        e = ek @ e
    return e


@nb.njit
def nlo_r_exact(gamma_singlet, nf, ev_op_max_order):
    """
    Compute singlet R vector for perturbative-exact mode.

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        r : np.ndarray
            R vector
    """
    r = np.zeros((ev_op_max_order + 1, 2, 2), np.complex_)  # k = 0 .. max_order
    beta0 = sc.beta(0, nf)
    r[0] = gamma_singlet[0] / beta0
    b1 = sc.b(1, nf)
    r[1] = gamma_singlet[1] / beta0 - b1 * gamma_singlet[0]
    # fill rest
    for kk in range(2, ev_op_max_order):
        r[kk] = -b1 * r[kk - 1]
    return r


@nb.njit
def nlo_r_expanded(gamma_singlet, nf, ev_op_max_order):
    """
    Compute singlet R vector for perturbative-expanded mode.

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        r : np.ndarray
            R vector
    """
    r_k = np.zeros((ev_op_max_order + 1, 2, 2), np.complex_)  # k = 0 .. max_order
    beta0 = sc.beta(0, nf)
    r_k[0] = gamma_singlet[0] / beta0
    b1 = sc.b(1, nf)
    r_k[1] = gamma_singlet[1] / beta0 - b1 * gamma_singlet[0]
    return r_k


@nb.njit
def nlo_u(r, ev_op_max_order):
    """
    Compute singlet U vector.

    Parameters
    ----------
        r : numpy.ndarray
            singlet R vector

    Returns
    -------
        u : np.ndarray
            U vector
    """
    # TODO find a way to numba compile this fnc (depends on sc.beta)
    u = np.zeros((ev_op_max_order + 1, 2, 2), np.complex_)  # k = 0 .. max_order
    # init
    u[0] = np.identity(2, np.complex_)
    _, r_p, r_m, e_p, e_m = ad.exp_singlet(r[0])
    for kk in range(1, ev_op_max_order + 1):
        rp = np.zeros((2, 2), np.complex_)
        for jj in range(kk):
            rp += r[kk - jj] @ u[jj]
        u[kk] = (
            (e_m @ rp @ e_m + e_p @ rp @ e_p) / kk
            + ((e_p @ rp @ e_m) / (r_m - r_p + kk))
            + ((e_m @ rp @ e_p) / (r_p - r_m + kk))
        )
    return u


@nb.njit
def nlo_perturbative_exact(
    gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
):
    """
    Singlet next-to-leading order pertubative-exact EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^1 : complex
            non-singlet next-to-leading order perturbative-exact EKO
    """


@nb.njit
def nlo_perturbative_expanded(
    gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
):
    """
    Singlet next-to-leading order pertubative-expanded EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^1 : complex
            non-singlet next-to-leading order perturbative-expanded EKO
    """


@nb.njit
def nlo_truncated(gamma_singlet, a1, a0, nf, ev_op_iterations):
    """
    Singlet next-to-leading order truncated EKO

    Parameters
    ----------
        gamma_singlet : list(numpy.ndarray)
            singlet anomalous dimensions matrices

    Returns
    -------
        e_s^1 : complex
            non-singlet next-to-leading order truncated EKO
    """
