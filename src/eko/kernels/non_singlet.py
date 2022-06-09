# -*- coding: utf-8 -*-
"""
Colletion of non-singlet EKOs.
"""

import numba as nb
import numpy as np

from .. import beta
from . import as4_evolution_integrals as as4_ei
from . import evolution_integrals as ei
from . import utils

nb.njit(cache=True)


def U_vec(gamma_ns, nf):
    r"""Compute the elements of the non-singlet U vector.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimension
    nf : int
        number of active flavors

    Returns
    -------
    np.ndarray
        U vector
    """
    order = gamma_ns.size - 1
    U = np.zeros(order + 1)

    beta0 = beta.beta(0, nf)
    R0 = gamma_ns[0] / beta0
    U[0] = 1.0
    if order >= 1:
        b1 = beta.b(1, nf)
        R1 = gamma_ns[1] / beta0 - b1 * R0
        U[1] = R1
    if order >= 2:
        b2 = beta.b(2, nf)
        R2 = gamma_ns[2] / beta0 - b1 * R1 - b2 * R0
        U[2] = 0.5 * (R2 + U[1] * R1)
    if order == 3:
        b3 = beta.b(3, nf)
        R3 = gamma_ns[3] / beta0 - b1 * R2 - b2 * R1 - b3 * R0
        U[3] = 1 / 3 * (R3 + R2 * U[1] + R1 * U[2])
    return U


@nb.njit(cache=True)
def lo_exact(gamma_ns, a1, a0, nf):
    """
    |LO| non-singlet exact EKO

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
            |LO| non-singlet exact EKO
    """
    return np.exp(gamma_ns[0] * ei.j00(a1, a0, nf))


@nb.njit(cache=True)
def nlo_exact(gamma_ns, a1, a0, nf):
    """
    |NLO| non-singlet exact EKO

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
            |NLO| non-singlet exact EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j01_exact(a1, a0, nf) + gamma_ns[1] * ei.j11_exact(a1, a0, nf)
    )


@nb.njit(cache=True)
def nlo_expanded(gamma_ns, a1, a0, nf):
    """
    |NLO| non-singlet expanded EKO

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
            |NLO| non-singlet expanded EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j01_expanded(a1, a0, nf)
        + gamma_ns[1] * ei.j11_expanded(a1, a0, nf)
    )


@nb.njit(cache=True)
def nlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    |NLO| non-singlet truncated EKO

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
            |NLO| non-singlet truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U0, U1 = U_vec(gamma_ns, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (U0 + (ah - al) * U1)
        al = ah
    return e


@nb.njit(cache=True)
def nlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    |NLO| non-singlet ordered-truncated EKO

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
            |NLO| non-singlet ordered-truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U0, U1 = U_vec(gamma_ns, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (U0 + ah * U1) / (U0 + al * U1)
        al = ah
    return e


@nb.njit(cache=True)
def nnlo_exact(gamma_ns, a1, a0, nf):
    """
    |NNLO| non-singlet exact EKO

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
        e_ns^2 : complex
            |NNLO| non-singlet exact EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j02_exact(a1, a0, nf)
        + gamma_ns[1] * ei.j12_exact(a1, a0, nf)
        + gamma_ns[2] * ei.j22_exact(a1, a0, nf)
    )


@nb.njit(cache=True)
def nnlo_expanded(gamma_ns, a1, a0, nf):
    """
    |NNLO| non-singlet expanded EKO

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
        e_ns^2 : complex
            |NNLO| non-singlet expanded EKO
    """
    return np.exp(
        gamma_ns[0] * ei.j02_expanded(a1, a0, nf)
        + gamma_ns[1] * ei.j12_expanded(a1, a0, nf)
        + gamma_ns[2] * ei.j22_expanded(a1, a0, nf)
    )


@nb.njit(cache=True)
def nnlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    |NNLO| non-singlet truncated EKO

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
        e_ns^2 : complex
            |NNLO| non-singlet truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U0, U1, U2 = U_vec(gamma_ns, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (
            U0
            + U1 * (ah - al)
            + U2 * ah**2
            - ah * al * U1**2
            + al**2 * (U1**2 - U2)
        )
        al = ah
    return e


@nb.njit(cache=True)
def nnlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """
    |NNLO| non-singlet ordered truncated EKO

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
        e_ns^2 : complex
            |NNLO| non-singlet ordered truncated EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U0, U1, U2 = U_vec(gamma_ns, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (U0 + ah * U1 + ah**2 * U2) / (U0 + al * U1 + al**2 * U2)
        al = ah
    return e


@nb.njit(cache=True)
def n3lo_expanded(gamma_ns, a1, a0, nf):
    """|N3LO| non-singlet expanded EKO

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
    complex
        |N3LO| non-singlet expanded EKO

    """
    return np.exp(
        gamma_ns[0] * as4_ei.j03_expanded(a1, a0, nf)
        + gamma_ns[1] * as4_ei.j13_expanded(a1, a0, nf)
        + gamma_ns[2] * as4_ei.j23_expanded(a1, a0, nf)
        + gamma_ns[3] * as4_ei.j33_expanded(a1, a0, nf)
    )


@nb.njit(cache=True)
def n3lo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """|N3LO| non-singlet truncated EKO

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
    complex
        |N3LO| non-singlet truncated EKO

    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U0, U1, U2, U3 = U_vec(gamma_ns, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= e0 * (
            U0
            + U1 * (ah - al)
            + U2 * ah**2
            - ah * al * U1**2
            + al**2 * (U1**2 - U2)
            + ah**3 * U3
            - ah**2 * al * U2 * U1
            + ah * al**2 * U1 * (U1**2 - U2)
            - al**3 * (U1**3 - 2 * U1 * U2 + U3)
        )
        al = ah
    return e


@nb.njit(cache=True)
def n3lo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations):
    """|N3LO| non-singlet ordered truncated EKO

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
    complex
        |N3LO| non-singlet ordered truncated EKO

    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U0, U1, U2, U3 = U_vec(gamma_ns, nf)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, nf)
        e *= (
            e0
            * (U0 + ah * U1 + ah**2 * U2 + ah**3 * U3)
            / (U0 + al * U1 + al**2 * U2 + al**3 * U3)
        )
        al = ah
    return e


@nb.njit(cache=True)
def n3lo_exact(gamma_ns, a1, a0, nf):
    """|N3LO| non-singlet exact EKO

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
    complex
        |N3LO| non-singlet exact EKO

    """
    beta0 = beta.beta(0, nf)
    b_list = [
        beta.b(1, nf),
        beta.b(2, nf),
        beta.b(3, nf),
    ]
    roots = as4_ei.roots(b_list)
    j00 = ei.j00(a1, a0, nf)
    j13 = as4_ei.j13_exact(a1, a0, beta0, b_list, roots)
    j23 = as4_ei.j23_exact(a1, a0, beta0, b_list, roots)
    j33 = as4_ei.j33_exact(a1, a0, beta0, b_list, roots)
    return np.exp(
        gamma_ns[0] * as4_ei.j03_exact(j00, j13, j23, j33, b_list)
        + gamma_ns[1] * j13
        + gamma_ns[2] * j23
        + gamma_ns[3] * j33
    )


@nb.njit(cache=True)
def dispatcher(
    order, method, gamma_ns, a1, a0, nf, ev_op_iterations
):  # pylint: disable=too-many-return-statements
    """Determine used kernel and call it.

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
    complex
        non-singlet EKO

    """
    # use always exact in LO
    if order == 0:
        return lo_exact(gamma_ns, a1, a0, nf)
    # NLO
    if order == 1:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return nlo_expanded(gamma_ns, a1, a0, nf)
        if method == "truncated":
            return nlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
        if method == "ordered-truncated":
            return nlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
        # if method in ["iterate-exact", "decompose-exact", "perturbative-exact"]:
        return nlo_exact(gamma_ns, a1, a0, nf)
    # NNLO
    if order == 2:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return nnlo_expanded(gamma_ns, a1, a0, nf)
        if method == "truncated":
            return nnlo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
        if method == "ordered-truncated":
            return nnlo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
        return nnlo_exact(gamma_ns, a1, a0, nf)
    # N3LO
    if order == 3:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return n3lo_expanded(gamma_ns, a1, a0, nf)
        if method == "truncated":
            return n3lo_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
        if method == "ordered-truncated":
            return n3lo_ordered_truncated(gamma_ns, a1, a0, nf, ev_op_iterations)
        return n3lo_exact(gamma_ns, a1, a0, nf)
    raise NotImplementedError("Selected order is not implemented")
