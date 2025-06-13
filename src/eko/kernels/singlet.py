"""Collection of singlet EKOs."""

import numba as nb
import numpy as np

from ekore import anomalous_dimensions as ad

from .. import beta
from . import EvoMethods
from . import as4_evolution_integrals as as4_ei
from . import evolution_integrals as ei


@nb.njit(cache=True)
def lo_exact(gamma_singlet, a1, a0, beta):
    """Singlet leading order exact EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    numpy.ndarray
        singlet leading order exact EKO
    """
    return ad.exp_matrix_2D(gamma_singlet[0] * ei.j12(a1, a0, beta[0]))[0]


@nb.njit(cache=True)
def nlo_decompose(gamma_singlet, j01, j11):
    """Singlet next-to-leading order decompose EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    j01 : float
        |LO|-|NLO| evolution integral
    j11 : float
        |NLO|-|NLO| evolution integral

    Returns
    -------
    numpy.ndarray
        singlet next-to-leading order decompose EKO
    """
    return ad.exp_matrix_2D(gamma_singlet[0] * j01 + gamma_singlet[1] * j11)[0]


@nb.njit(cache=True)
def nlo_decompose_exact(gamma_singlet, a1, a0, beta):
    """Singlet next-to-leading order decompose-exact EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    numpy.ndarray
        singlet next-to-leading order decompose-exact EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return nlo_decompose(
        gamma_singlet,
        ei.j13_exact(a1, a0, beta0, b_vec),
        ei.j23_exact(a1, a0, beta0, b_vec),
    )


@nb.njit(cache=True)
def nlo_decompose_expanded(gamma_singlet, a1, a0, beta):
    """Singlet next-to-leading order decompose-expanded EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    numpy.ndarray
        singlet next-to-leading order decompose-expanded EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return nlo_decompose(
        gamma_singlet,
        ei.j13_expanded(a1, a0, beta0, b_vec),
        ei.j23_expanded(a1, a0, beta0),
    )


@nb.njit(cache=True)
def nnlo_decompose(gamma_singlet, j02, j12, j22):
    """Singlet next-to-next-to-leading order decompose EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    j02 : float
        LO-NNLO evolution integral
    j12 : float
        NLO-NNLO evolution integral
    j22 : float
        NNLO-NNLO evolution integral

    Returns
    -------
    numpy.ndarray
        singlet next-to-next-to-leading order decompose EKO
    """
    return ad.exp_matrix_2D(
        gamma_singlet[0] * j02 + gamma_singlet[1] * j12 + gamma_singlet[2] * j22
    )[0]


@nb.njit(cache=True)
def nnlo_decompose_exact(gamma_singlet, a1, a0, beta):
    """Singlet next-to-next-to-leading order decompose-exact EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    numpy.ndarray
        singlet next-to-next-to-leading order decompose-exact EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return nnlo_decompose(
        gamma_singlet,
        ei.j14_exact(a1, a0, beta0, b_vec),
        ei.j24_exact(a1, a0, beta0, b_vec),
        ei.j34_exact(a1, a0, beta0, b_vec),
    )


@nb.njit(cache=True)
def nnlo_decompose_expanded(gamma_singlet, a1, a0, beta):
    """Singlet next-to-next-to-leading order decompose-expanded EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions

    Returns
    -------
    numpy.ndarray
        singlet next-to-next-to-leading order decompose-expanded EKO
    """
    beta0 = beta[0]
    b_vec = [betas / beta0 for betas in beta]
    return nnlo_decompose(
        gamma_singlet,
        ei.j14_expanded(a1, a0, beta0, b_vec),
        ei.j24_expanded(a1, a0, beta0, b_vec),
        ei.j34_expanded(a1, a0, beta0),
    )


@nb.njit(cache=True)
def n3lo_decompose(gamma_singlet, j03, j13, j23, j33):
    """Singlet |N3LO| decompose EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    j03 : float
        |LO|-|N3LO| evolution integral
    j13 : float
        |NLO|-|N3LO| evolution integral
    j23 : float
        |NNLO|-|N3LO| evolution integral
    j33 : float
        |N3LO|-|N3LO| evolution integral

    Returns
    -------
    numpy.ndarray
        singlet |N3LO| decompose EKO
    """
    return ad.exp_matrix_2D(
        gamma_singlet[0] * j03
        + gamma_singlet[1] * j13
        + gamma_singlet[2] * j23
        + gamma_singlet[3] * j33
    )[0]


@nb.njit(cache=True)
def n3lo_decompose_exact(gamma_singlet, a1, a0, nf):
    """Singlet |N3LO| decompose-exact EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    nf : int
        number of active flavors

    Returns
    -------
    numpy.ndarray
        singlet |N3LO| decompose-exact EKO
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
    return n3lo_decompose(
        gamma_singlet, as4_ei.j03_exact(j12, j13, j23, j33, b_list), j13, j23, j33
    )


@nb.njit(cache=True)
def n3lo_decompose_expanded(gamma_singlet, a1, a0, nf):
    """Singlet |N3LO| decompose-expanded EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    nf : int
        number of active flavors

    Returns
    -------
    numpy.ndarray
        singlet |N3LO| decompose-expanded EKO
    """
    beta0 = beta.beta_qcd((2, 0), nf)
    b_list = [
        beta.b_qcd((3, 0), nf),
        beta.b_qcd((4, 0), nf),
        beta.b_qcd((5, 0), nf),
    ]
    j12 = ei.j12(a1, a0, nf)
    j13 = as4_ei.j13_expanded(a1, a0, beta0, b_list)
    j23 = as4_ei.j23_expanded(a1, a0, beta0, b_list)
    j33 = as4_ei.j33_expanded(a1, a0, beta0)
    return n3lo_decompose(
        gamma_singlet, as4_ei.j03_expanded(j12, j13, j23, j33, b_list), j13, j23, j33
    )


@nb.njit(cache=True)
def eko_iterate(gamma_singlet, a1, a0, beta_vec, order, ev_op_iterations):
    """Singlet |NLO|, |NNLO| or |N3LO| iterated (exact) EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta_vec : list
        list of the values of the beta functions
    order : tuple(int,int)
        perturbative order
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    numpy.ndarray
        singlet iterated (exact) EKO
    """
    a_steps = np.geomspace(a0, a1, 1 + ev_op_iterations)
    e = np.identity(2, np.complex128)
    al = a_steps[0]
    for ah in a_steps[1:]:
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        gamma_summed = np.zeros((2, 2), dtype=np.complex128)
        beta_summed = 0
        for i in range(order[0]):
            gamma_summed += gamma_singlet[i] * a_half**i
            beta_summed += beta_vec[i] * a_half ** (i + 1)
        ln = gamma_summed / beta_summed * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix_2D(ln)[0])
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def r_vec(gamma_singlet, beta, ev_op_max_order, order, is_exact):
    r"""Compute singlet R vector for perturbative mode.

    .. math::
        \frac{d}{da_s} \dSV{1}{a_s} &= \frac{\mathbf R (a_s)}{a_s} \cdot \dSV{1}{a_s}\\
        \mathbf R (a_s) &= \sum\limits_{k=0} a_s^k \mathbf R_{k}

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    beta : list
        list of the values of the beta functions
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U
    order : tuple(int,int)
       perturbative order
    is_exact : boolean
        fill up r-vector?

    Returns
    -------
    np.ndarray
        R vector
    """
    r = np.zeros(
        (ev_op_max_order[0] + 1, 2, 2), dtype=np.complex128
    )  # k = 0 .. max_order
    beta0 = beta[0]
    # fill explicit elements
    r[0] = gamma_singlet[0] / beta0
    if order[0] > 1:
        b1 = beta[1] / beta0
        r[1] = gamma_singlet[1] / beta0 - b1 * r[0]
    if order[0] > 2:
        b2 = beta[2] / beta0
        r[2] = gamma_singlet[2] / beta0 - b1 * r[1] - b2 * r[0]
    if order[0] > 3:
        b3 = beta[3] / beta0
        r[3] = gamma_singlet[3] / beta0 - b1 * r[2] - b2 * r[1] - b3 * r[0]
    # fill rest
    if is_exact:
        if order[0] == 2:
            for kk in range(2, ev_op_max_order[0]):
                r[kk] = -b1 * r[kk - 1]
        elif order[0] == 3:
            for kk in range(3, ev_op_max_order[0]):
                r[kk] = -b1 * r[kk - 1] - b2 * r[kk - 2]
        elif order[0] == 4:
            for kk in range(4, ev_op_max_order[0] + 1):
                r[kk] = -b1 * r[kk - 1] - b2 * r[kk - 2] - b3 * r[kk - 3]
    return r


@nb.njit(cache=True)
def u_vec(r, ev_op_max_order):
    r"""Compute the elements of the singlet U vector.

    .. math::
        \ESk{n}{a_s}{a_s^0} &= \mathbf U (a_s) \ESk{0}{a_s}{a_s^0} {\mathbf U}^{-1} (a_s^0)\\
        \mathbf U (a_s) &= \mathbf I + \sum\limits_{k=1} a_s^k \mathbf U_k

    Parameters
    ----------
    r : numpy.ndarray
        singlet R vector
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U

    Returns
    -------
    numpy.ndarray
        U vector
    """
    u = np.zeros((ev_op_max_order[0], 2, 2), np.complex128)  # k = 0 .. max_order
    # init
    u[0] = np.identity(2, np.complex128)
    _, r_p, r_m, e_p, e_m = ad.exp_matrix_2D(r[0])
    e_p = np.ascontiguousarray(e_p)
    e_m = np.ascontiguousarray(e_m)
    for kk in range(1, ev_op_max_order[0]):
        # compute R'
        rp = np.zeros((2, 2), dtype=np.complex128)
        for jj in range(kk):
            rp += np.ascontiguousarray(r[kk - jj]) @ u[jj]
        # now compose U
        u[kk] = (
            (e_m @ rp @ e_m + e_p @ rp @ e_p) / kk
            + ((e_p @ rp @ e_m) / (r_m - r_p + kk))
            + ((e_m @ rp @ e_p) / (r_p - r_m + kk))
        )
    return u


@nb.njit(cache=True)
def sum_u(uvec, a):
    r"""Sum up the actual U operator.

    .. math::
        \mathbf U (a_s) = \mathbf I + \sum\limits_{k=1} a_s^k \mathbf U_k

    Parameters
    ----------
    uvec : numpy.ndarray
        U vector (elements)
    a : float
        strong coupling

    Returns
    -------
    numpy.ndarray
        sum
    """
    p = 1.0
    res = np.zeros((2, 2), dtype=np.complex128)
    for uk in uvec:
        res += p * uk
        p *= a
    # alternative implementation:
    # al_vec = al**(np.arange(len(uk)))
    # ul = np.sum(al_vec * uk.T,-1).T
    return res


@nb.njit(cache=True)
def eko_perturbative(
    gamma_singlet, a1, a0, beta, order, ev_op_iterations, ev_op_max_order, is_exact
):
    """Singlet |NLO|,|NNLO| or |N3LO| perturbative EKO, depending on which r is
    passed.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
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
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U
    is_exact : boolean
        fill up r-vector?

    Returns
    -------
    numpy.ndarray
        singlet perturbative EKO
    """
    r = r_vec(gamma_singlet, beta, ev_op_max_order, order, is_exact)
    uk = u_vec(r, ev_op_max_order)
    e = np.identity(2, np.complex128)
    # iterate elements
    a_steps = np.geomspace(a0, a1, 1 + ev_op_iterations)
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_singlet, ah, al, beta)
        uh = sum_u(uk, ah)
        ul = sum_u(uk, al)
        # join elements
        ek = np.ascontiguousarray(uh) @ np.ascontiguousarray(e0) @ np.linalg.inv(ul)
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def eko_truncated(gamma_singlet, a1, a0, beta, order):
    """Singlet |NLO|, |NNLO| or |N3LO| truncated EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta : list
        list of the values of the beta functions
    order : tuple(int,int)
        perturbative order

    Returns
    -------
    numpy.ndarray
        singlet truncated EKO
    """
    r = r_vec(gamma_singlet, beta, order, order, False)
    u = u_vec(r, order)
    u1 = np.ascontiguousarray(u[1])
    e0 = np.ascontiguousarray(lo_exact(gamma_singlet, a1, a0, beta))
    e = e0
    if order[0] >= 2:
        e += a1 * u1 @ e0 - a0 * e0 @ u1
    if order[0] >= 3:
        u2 = np.ascontiguousarray(u[2])
        e += +(a1**2) * u2 @ e0 - a1 * a0 * u1 @ e0 @ u1 + a0**2 * e0 @ (u1 @ u1 - u2)
    if order[0] >= 4:
        u3 = np.ascontiguousarray(u[3])
        e += (
            +(a1**3) * u3 @ e0
            - a1**2 * a0 * u2 @ e0 @ u1
            + a1 * a0**2 * u1 @ e0 @ (u1 @ u1 - u2)
            - a0**3 * e0 @ (u1 @ u1 @ u1 - u1 @ u2 - u2 @ u1 + u3)
        )
    return e


@nb.njit(cache=True)
def dispatcher(  # pylint: disable=too-many-return-statements
    order, method, gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
):
    """Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
    order :  tuple(int,int)
        perturbative order
    method : int
        method
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U

    Returns
    -------
    numpy.ndarray
        singlet EKO
    """
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    # for SV expanded we still fall in here, but we don't need to do anything
    if a1 == a0:
        return np.eye(len(gamma_singlet[0]), dtype=np.complex128)

    # use always exact in LO
    if order[0] == 1:
        return lo_exact(gamma_singlet, a1, a0, betalist)

    # Common method for NLO and NNLO
    if method in [EvoMethods.ITERATE_EXACT, EvoMethods.ITERATE_EXPANDED]:
        return eko_iterate(gamma_singlet, a1, a0, betalist, order, ev_op_iterations)
    if method == EvoMethods.PERTURBATIVE_EXACT:
        return eko_perturbative(
            gamma_singlet,
            a1,
            a0,
            betalist,
            order,
            ev_op_iterations,
            ev_op_max_order,
            True,
        )
    if method == EvoMethods.PERTURBATIVE_EXPANDED:
        return eko_perturbative(
            gamma_singlet,
            a1,
            a0,
            betalist,
            order,
            ev_op_iterations,
            ev_op_max_order,
            False,
        )
    if method in [EvoMethods.TRUNCATED, EvoMethods.ORDERED_TRUNCATED]:
        return eko_truncated(gamma_singlet, a1, a0, betalist, order)
    # These methods are scattered for nlo and nnlo
    if method == EvoMethods.DECOMPOSE_EXACT:
        if order[0] == 2:
            return nlo_decompose_exact(gamma_singlet, a1, a0, betalist)
        if order[0] == 3:
            return nnlo_decompose_exact(gamma_singlet, a1, a0, betalist)
        return n3lo_decompose_exact(gamma_singlet, a1, a0, nf)
    if method == EvoMethods.DECOMPOSE_EXPANDED:
        if order[0] == 2:
            return nlo_decompose_expanded(gamma_singlet, a1, a0, betalist)
        if order[0] == 3:
            return nnlo_decompose_expanded(gamma_singlet, a1, a0, betalist)
        return n3lo_decompose_expanded(gamma_singlet, a1, a0, nf)
    raise NotImplementedError("Selected method is not implemented")
