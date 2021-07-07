# -*- coding: utf-8 -*-
"""
Colletion of singlet EKOs.
"""

import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import evolution_integrals as ei
from . import utils


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1)", cache=True)
def lo_exact(gamma_singlet, a1, a0, nf):
    """
    Singlet leading order exact EKO

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
        e_s^0 : numpy.ndarray
            singlet leading order exact EKO
    """
    return ad.exp_singlet(gamma_singlet[0] * ei.j00(a1, a0, nf))[0]


@nb.njit("c16[:,:](c16[:,:,:],f8,f8)", cache=True)
def nlo_decompose(gamma_singlet, j01, j11):
    """
    Singlet next-to-leading order decompose EKO

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
        j01 : float
            LO-NLO evolution integral
        j11 : float
            NLO-NLO evolution integral

    Returns
    -------
        e_s^1 : numpy.ndarray
            singlet next-to-leading order decompose EKO
    """
    return ad.exp_singlet(gamma_singlet[0] * j01 + gamma_singlet[1] * j11)[0]


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1)", cache=True)
def nlo_decompose_exact(gamma_singlet, a1, a0, nf):
    """
    Singlet next-to-leading order decompose-exact EKO

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
        e_s^1 : numpy.ndarray
            singlet next-to-leading order decompose-exact EKO
    """
    return nlo_decompose(
        gamma_singlet, ei.j01_exact(a1, a0, nf), ei.j11_exact(a1, a0, nf)
    )


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1)", cache=True)
def nlo_decompose_expanded(gamma_singlet, a1, a0, nf):
    """
    Singlet next-to-leading order decompose-expanded EKO

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
        e_s^1 : numpy.ndarray
            singlet next-to-leading order decompose-expanded EKO
    """
    return nlo_decompose(
        gamma_singlet, ei.j01_expanded(a1, a0, nf), ei.j11_expanded(a1, a0, nf)
    )


@nb.njit("c16[:,:](c16[:,:,:],c16,c16,c16)", cache=True)
def nnlo_decompose(gamma_singlet, j02, j12, j22):
    """
    Singlet next-to-next-to-leading order decompose EKO

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
        j02 : float
            LO-NNLO evolution integral
        j12 : float
            NLO-NNLO evolution integral
        j22 : float
            NNLO-NNLO evolution integral
    Returns
    -------
        e_s^2 : numpy.ndarray
            singlet next-to-next-to-leading order decompose EKO
    """
    return ad.exp_singlet(
        gamma_singlet[0] * j02 + gamma_singlet[1] * j12 + gamma_singlet[2] * j22
    )[0]


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1)", cache=True)
def nnlo_decompose_exact(gamma_singlet, a1, a0, nf):
    """
    Singlet next-to-next-to-leading order decompose-exact EKO

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
        e_s^2 : numpy.ndarray
            singlet next-to-next-to-leading order decompose-exact EKO
    """
    return nnlo_decompose(
        gamma_singlet,
        ei.j02_exact(a1, a0, nf),
        ei.j12_exact(a1, a0, nf),
        ei.j22_exact(a1, a0, nf),
    )


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1)", cache=True)
def nnlo_decompose_expanded(gamma_singlet, a1, a0, nf):
    """
    Singlet next-to-next-to-leading order decompose-expanded EKO

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
        e_s^2 : numpy.ndarray
            singlet next-to-next-to-leading order decompose-expanded EKO
    """
    return nnlo_decompose(
        gamma_singlet,
        ei.j02_expanded(a1, a0, nf),
        ei.j12_expanded(a1, a0, nf),
        ei.j22_expanded(a1, a0, nf),
    )


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1, u1, u4)", cache=True)
def eko_iterate(gamma_singlet, a1, a0, nf, order, ev_op_iterations):
    """
    Singlet NLO or NNLO iterated (exact) EKO

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
        order : int
            perturbative order
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_s^{order} : numpy.ndarray
            singlet NLO or NNLO iterated (exact) EKO
    """
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    beta0 = beta.beta(0, nf)
    beta1 = beta.beta(1, nf)
    if order >= 2:
        beta2 = beta.beta(2, nf)
    e = np.identity(2, np.complex_)
    al = a_steps[0]
    for ah in a_steps[1:]:
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        if order == 1:
            ln = (
                (gamma_singlet[0] * a_half + gamma_singlet[1] * a_half ** 2)
                / (beta0 * a_half ** 2 + beta1 * a_half ** 3)
                * delta_a
            )
        elif order == 2:
            ln = (
                (
                    gamma_singlet[0] * a_half
                    + gamma_singlet[1] * a_half ** 2
                    + gamma_singlet[2] * a_half ** 3
                )
                / (beta0 * a_half ** 2 + beta1 * a_half ** 3 + beta2 * a_half ** 4)
                * delta_a
            )
        ek = np.ascontiguousarray(ad.exp_singlet(ln)[0])
        e = ek @ e
        al = ah
    return e


@nb.njit("c16[:,:,:](c16[:,:,:],u1,u1,u1,b1)", cache=True)
def r_vec(gamma_singlet, nf, ev_op_max_order, order, is_exact):
    r"""
    Compute singlet R vector for perturbative mode.

    .. math::
        \frac{d}{da_s} \dSV{1}{a_s} &= \frac{\mathbf R (a_s)}{a_s} \cdot \dSV{1}{a_s}\\
        \mathbf R (a_s) &= \sum\limits_{k=0} a_s^k \mathbf R_{k}

    Parameters
    ----------
        gamma_singlet : numpy.ndarray
            singlet anomalous dimensions matrices
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps
        order : int
            order order
        is_exact : boolean
            fill up r-vector?

    Returns
    -------
        r : np.ndarray
            R vector
    """
    r = np.zeros((ev_op_max_order + 1, 2, 2), np.complex_)  # k = 0 .. max_order
    beta0 = beta.beta(0, nf)
    b1 = beta.b(1, nf)
    b2 = beta.b(2, nf)
    # fill explicit elements
    r[0] = gamma_singlet[0] / beta0
    if order > 0:
        r[1] = gamma_singlet[1] / beta0 - b1 * r[0]
    if order > 1:
        r[2] = gamma_singlet[2] / beta0 - b1 * r[1] - b2 * r[0]
    # fill rest
    if is_exact:
        if order == 1:
            for kk in range(2, ev_op_max_order + 1):
                r[kk] = -b1 * r[kk - 1]
        elif order == 2:
            for kk in range(3, ev_op_max_order + 1):
                r[kk] = -b1 * r[kk - 1] - b2 * r[kk - 2]
    return r


@nb.njit("c16[:,:,:](c16[:,:,:],u1)", cache=True)
def u_vec(r, ev_op_max_order):
    r"""
    Compute the elements of the singlet U vector.

    .. math::
        \ESk{n}{a_s}{a_s^0} &= \mathbf U (a_s) \ESk{0}{a_s}{a_s^0} {\mathbf U}^{-1} (a_s^0)\\
        \mathbf U (a_s) &= \mathbf I + \sum\limits_{k=1} a_s^k \mathbf U_k

    Parameters
    ----------
        r : numpy.ndarray
            singlet R vector
        ev_op_max_order : int
            perturbative expansion order of U

    Returns
    -------
        u : np.ndarray
            U vector
    """
    u = np.zeros((ev_op_max_order + 1, 2, 2), np.complex_)  # k = 0 .. max_order
    # init
    u[0] = np.identity(2, np.complex_)
    _, r_p, r_m, e_p, e_m = ad.exp_singlet(r[0])
    e_p = np.ascontiguousarray(e_p)
    e_m = np.ascontiguousarray(e_m)
    for kk in range(1, ev_op_max_order + 1):
        # compute R'
        rp = np.zeros((2, 2), np.complex_)
        for jj in range(kk):
            rp += np.ascontiguousarray(r[kk - jj]) @ u[jj]
        # now compose U
        u[kk] = (
            (e_m @ rp @ e_m + e_p @ rp @ e_p) / kk
            + ((e_p @ rp @ e_m) / (r_m - r_p + kk))
            + ((e_m @ rp @ e_p) / (r_p - r_m + kk))
        )
    return u


@nb.njit("c16[:,:](c16[:,:,:],f8)", cache=True)
def sum_u(uvec, a):
    r"""
    Sums up the actual U operator.


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
        u : numpy.ndarray
            sum
    """
    p = 1.0
    res = np.zeros((2, 2), np.complex_)
    for uk in uvec:
        res += p * uk
        p *= a
    # alternative implementation:
    # al_vec = al**(np.arange(len(uk)))
    # ul = np.sum(al_vec * uk.T,-1).T
    return res


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1,u1,u4,u1,b1)", cache=True)
def eko_perturbative(
    gamma_singlet, a1, a0, nf, order, ev_op_iterations, ev_op_max_order, is_exact
):
    """
    Singlet NLO or NNLO order pertubative EKO, depending on which r is passed

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
        order : int
            perturbative order
        ev_op_iterations : int
            number of evolution steps
        ev_op_max_order : int
            perturbative expansion order of U
        is_exact : boolean
            fill up r-vector?

    Returns
    -------
        e_s^{order} : numpy.ndarray
            singlet NLO or NNLO order perturbative EKO
    """
    r = r_vec(gamma_singlet, nf, ev_op_max_order, order, is_exact)
    uk = u_vec(r, ev_op_max_order)
    e = np.identity(2, np.complex_)
    # iterate elements
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = lo_exact(gamma_singlet, ah, al, nf)
        uh = sum_u(uk, ah)
        ul = sum_u(uk, al)
        # join elements
        ek = np.ascontiguousarray(uh) @ np.ascontiguousarray(e0) @ np.linalg.inv(ul)
        e = ek @ e
        al = ah
    return e


@nb.njit("c16[:,:](c16[:,:,:],f8,f8,u1,u1,u4)", cache=True)
def eko_truncated(gamma_singlet, a1, a0, nf, order, ev_op_iterations):
    """
    Singlet NLO or NNLO truncated EKO

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
        order : int
            perturbative order
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_s^{order} : numpy.ndarray
            singlet NLO or NNLO truncated EKO
    """
    r = r_vec(gamma_singlet, nf, order, order, False)
    u = u_vec(r, order)
    u1 = np.ascontiguousarray(u[1])
    if order >= 2:
        u2 = np.ascontiguousarray(u[2])
    e = np.identity(2, np.complex_)
    # iterate elements
    a_steps = utils.geomspace(a0, a1, ev_op_iterations)
    al = a_steps[0]
    for ah in a_steps[1:]:
        e0 = np.ascontiguousarray(lo_exact(gamma_singlet, ah, al, nf))
        if order >= 1:
            ek = e0 + ah * u1 @ e0 - al * e0 @ u1
        if order >= 2:
            ek += (
                +(ah ** 2) * u2 @ e0
                - ah * al * u1 @ e0 @ u1
                + al ** 2 * e0 @ (u1 @ u1 - u2)
            )
        e = ek @ e
        al = ah
    return e


@nb.njit("c16[:,:](u1,string,c16[:,:,:],f8,f8,u1,u4,u1)", cache=True)
def dispatcher(  # pylint: disable=too-many-return-statements
    order, method, gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
):
    """
    Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
        method : str
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
        ev_op_max_order : int
            perturbative expansion order of U

    Returns
    -------
        e_s : numpy.ndarray
            singlet EKO
    """
    # use always exact in LO
    if order == 0:
        return lo_exact(gamma_singlet, a1, a0, nf)

    # Common method for NLO and NNLO
    if method in ["iterate-exact", "iterate-expanded"]:
        return eko_iterate(gamma_singlet, a1, a0, nf, order, ev_op_iterations)
    elif method == "perturbative-exact":
        return eko_perturbative(
            gamma_singlet, a1, a0, nf, order, ev_op_iterations, ev_op_max_order, True
        )
    elif method == "perturbative-expanded":
        return eko_perturbative(
            gamma_singlet, a1, a0, nf, order, ev_op_iterations, ev_op_max_order, False
        )
    elif method in ["truncated", "ordered-truncated"]:
        return eko_truncated(gamma_singlet, a1, a0, nf, order, ev_op_iterations)
    # These methods are scattered for nlo and nnlo
    elif method == "decompose-exact":
        if order == 1:
            return nlo_decompose_exact(gamma_singlet, a1, a0, nf)
        else:
            return nnlo_decompose_exact(gamma_singlet, a1, a0, nf)
    elif method == "decompose-expanded":
        if order == 1:
            return nlo_decompose_expanded(gamma_singlet, a1, a0, nf)
        else:
            return nnlo_decompose_expanded(gamma_singlet, a1, a0, nf)
    else:
        raise NotImplementedError("Selected method is not implemented")
