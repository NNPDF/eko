"""Collection of non-singlet EKOs."""

import numba as nb
import numpy as np

from .. import beta
from . import as4_evolution_integrals as as4_ei
from . import evolution_integrals as ei
from . import utils


@nb.njit(cache=True)
def U_vec(gamma_ns, beta, order):
    r"""Compute the elements of the non-singlet U vector.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimension
    beta : list
        list of the values of the beta functions
    order : int
        perturbative order

    Returns
    -------
    np.ndarray
        U vector

    """
    U = np.zeros(order[0], dtype=np.complex_)
    beta0 = beta[0]
    R0 = gamma_ns[0] / beta0
    U[0] = 1.0
    if order[0] >= 2:
        b1 = beta[1] / beta[0]
        R1 = gamma_ns[1] / beta0 - b1 * R0
        U[1] = R1
    if order[0] >= 3:
        b2 = beta[2] / beta[0]
        R2 = gamma_ns[2] / beta0 - b1 * R1 - b2 * R0
        U[2] = 0.5 * (R2 + U[1] * R1)
    if order[0] >= 4:
        b3 = beta[3] / beta[0]
        R3 = gamma_ns[3] / beta0 - b1 * R2 - b2 * R1 - b3 * R0
        U[3] = 1 / 3 * (R3 + R2 * U[1] + R1 * U[2])
    return U


@nb.njit(cache=True)
def lo_exact(gamma_ns, a1, a0, beta):
    """|LO| non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    e_ns^0 : complex
        |LO| non-singlet exact EKO
    """
    beta0 = beta[0]
    return np.exp(gamma_ns[0] * ei.j12(a1, a0, beta0))


@nb.njit(cache=True)
def nlo_exact(gamma_ns, a1, a0, beta):
    """|NLO| non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    e_ns^1 : complex
        |NLO| non-singlet exact EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return np.exp(
        gamma_ns[0] * ei.j13_exact(a1, a0, beta0, b_vec)
        + gamma_ns[1] * ei.j23_exact(a1, a0, beta0, b_vec)
    )


@nb.njit(cache=True)
def nlo_expanded(gamma_ns, a1, a0, beta):
    """|NLO| non-singlet expanded EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    e_ns^1 : complex
        |NLO| non-singlet expanded EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return np.exp(
        gamma_ns[0] * ei.j13_expanded(a1, a0, beta0, b_vec)
        + gamma_ns[1] * ei.j23_expanded(a1, a0, beta0)
    )


@nb.njit(cache=True)
def nnlo_exact(gamma_ns, a1, a0, beta):
    """|NNLO| non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    e_ns^2 : complex
        |NNLO| non-singlet exact EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return np.exp(
        gamma_ns[0] * ei.j14_exact(a1, a0, beta0, b_vec)
        + gamma_ns[1] * ei.j24_exact(a1, a0, beta0, b_vec)
        + gamma_ns[2] * ei.j34_exact(a1, a0, beta0, b_vec)
    )


@nb.njit(cache=True)
def nnlo_expanded(gamma_ns, a1, a0, beta):
    """|NNLO| non-singlet expanded EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    e_ns^2 : complex
        |NNLO| non-singlet expanded EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return np.exp(
        gamma_ns[0] * ei.j14_expanded(a1, a0, beta0, b_vec)
        + gamma_ns[1] * ei.j24_expanded(a1, a0, beta0, b_vec)
        + gamma_ns[2] * ei.j34_expanded(a1, a0, beta0)
    )


@nb.njit(cache=True)
def n3lo_expanded(gamma_ns, a1, a0, nf):
    """|N3LO| non-singlet expanded EKO.

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
    beta0 = beta.beta_qcd((2, 0), nf)
    b_list = [
        beta.b_qcd((3, 0), nf),
        beta.b_qcd((4, 0), nf),
        beta.b_qcd((5, 0), nf),
    ]
    j12 = ei.j12(a1, a0, beta0)
    j13 = as4_ei.j13_expanded(a1, a0, beta0, b_list)
    j23 = as4_ei.j23_expanded(a1, a0, beta0, b_list)
    j33 = as4_ei.j33_expanded(a1, a0, beta0)
    return np.exp(
        gamma_ns[0] * as4_ei.j03_expanded(j12, j13, j23, j33, b_list)
        + gamma_ns[1] * j13
        + gamma_ns[2] * j23
        + gamma_ns[3] * j33
    )


@nb.njit(cache=True)
def n3lo_exact(gamma_ns, a1, a0, nf):
    """|N3LO| non-singlet exact EKO.

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
    beta0 = beta.beta_qcd((2, 0), nf)
    b_list = [
        beta.b_qcd((3, 0), nf),
        beta.b_qcd((4, 0), nf),
        beta.b_qcd((5, 0), nf),
    ]
    roots = as4_ei.roots(b_list)
    j12 = ei.j12(a1, a0, beta0)
    j13 = as4_ei.j13_exact(a1, a0, beta0, b_list, roots)
    j23 = as4_ei.j23_exact(a1, a0, beta0, b_list, roots)
    j33 = as4_ei.j33_exact(a1, a0, beta0, b_list, roots)
    return np.exp(
        gamma_ns[0] * as4_ei.j03_exact(j12, j13, j23, j33, b_list)
        + gamma_ns[1] * j13
        + gamma_ns[2] * j23
        + gamma_ns[3] * j33
    )


@nb.njit(cache=True)
def eko_ordered_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations):
    """|NLO|, |NNLO| or |N3LO| non-singlet ordered truncated EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions
    order : tuple(int,int)
        perturbative order
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    complex
        non-singlet ordered truncated EKO

    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U = U_vec(gamma_ns, beta, order)
    e = 1.0
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, beta)
        num, den = 0, 0
        for i in range(order[0]):
            num += U[i] * ah**i
            den += U[i] * al**i
        e *= e0 * num / den
        al = ah
    return e


@nb.njit(cache=True)
def eko_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations):
    """|NLO|, |NNLO| or |N3LO| non-singlet truncated EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions
    order : tuple(int,int)
        perturbative order
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    complex
        non-singlet truncated EKO

    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    U = U_vec(gamma_ns, beta, order)
    e = 1.0
    al = a_steps[0]
    fact = U[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_ns, ah, al, beta)
        if order[0] >= 2:
            fact += U[1] * (ah - al)
        if order[0] >= 3:
            fact += +U[2] * ah**2 - ah * al * U[1] ** 2 + al**2 * (U[1] ** 2 - U[2])
        if order[0] >= 4:
            fact += (
                +(ah**3) * U[3]
                - ah**2 * al * U[2] * U[1]
                + ah * al**2 * U[1] * (U[1] ** 2 - U[2])
                - al**3 * (U[1] ** 3 - 2 * U[1] * U[2] + U[3])
            )
        e *= e0 * fact
        al = ah
    return e


@nb.njit(cache=True)
def dispatcher(
    order, method, gamma_ns, a1, a0, nf, ev_op_iterations
):  # pylint: disable=too-many-return-statements
    """Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
    order : tuple(int,int)
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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    # use always exact in LO
    if order[0] == 1:
        return lo_exact(gamma_ns, a1, a0, betalist)
    if method == "ordered-truncated":
        return eko_ordered_truncated(
            gamma_ns, a1, a0, betalist, order, ev_op_iterations
        )
    if method == "truncated":
        return eko_truncated(gamma_ns, a1, a0, betalist, order, ev_op_iterations)

    # NLO
    if order[0] == 2:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return nlo_expanded(gamma_ns, a1, a0, betalist)
        # if method in ["iterate-exact", "decompose-exact", "perturbative-exact"]:
        return nlo_exact(gamma_ns, a1, a0, betalist)
    # NNLO
    if order[0] == 3:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return nnlo_expanded(gamma_ns, a1, a0, betalist)
        return nnlo_exact(gamma_ns, a1, a0, betalist)
    # N3LO
    if order[0] == 4:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return n3lo_expanded(gamma_ns, a1, a0, nf)
        return n3lo_exact(gamma_ns, a1, a0, nf)
    raise NotImplementedError("Selected order is not implemented")
