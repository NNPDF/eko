"""Collection of QED singlet EKOs."""
import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import singlet as s
from . import utils

# from .non_singlet_qed import contract_gammas


@nb.njit(cache=True)
def contract_gammas(gamma_singlet, aem):
    """Contract anomalous dimension along the QED axis.

    Parameters
    ----------
    gamma_ns : 2D numpy.ndarray
        non-singlet anomalous dimensions
    aem : float
        electromagnetic coupling value

    Returns
    -------
    gamma_ns : 1D numpy.ndarray
        non-singlet anomalous dimensions
    """
    vec_alphaem = np.array(
        [aem**i for i in range(gamma_singlet.shape[1])], dtype=np.complex_
    )
    mat_dim = gamma_singlet.shape[-1]
    # TODO : implement this contraction with numpy
    res = np.zeros((gamma_singlet.shape[0], mat_dim, mat_dim), dtype=np.complex_)
    for qcd in range(gamma_singlet.shape[0]):
        for qed in range(gamma_singlet.shape[1]):
            res[qcd] += gamma_singlet[qcd, qed, :, :] * vec_alphaem[qed]
    return res


@nb.njit(cache=True)
def eko_iterate(gamma_singlet, a1, a0, aem_list, nf, order, ev_op_iterations, dim):
    """Singlet QEDxQCD iterated (exact) EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target strong coupling value
    a0 : float
        initial strong coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors
    order : tuple(int,int)
        QCDxQED perturbative orders
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    e_s^{order} : numpy.ndarray
        singlet QEDxQCD iterated (exact) EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    e = np.identity(dim, np.complex_)
    betaQCD = np.zeros((order[0] + 1, order[1] + 1))
    for i in range(1, order[0] + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for step in range(1, ev_op_iterations + 1):
        ah = a_steps[step]
        al = a_steps[step - 1]
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        gamma = np.zeros((dim, dim), np.complex_)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += betaQCD[i, j] * a_half ** (i + 1) * aem_list[step - 1] ** j
                gamma += gamma_singlet[i, j] * a_half**i * aem_list[step - 1] ** j
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix(ln)[0])
        e = ek @ e
    return e


@nb.njit(cache=True)
def qed_lo(gamma_singlet, a1, a0, aem_list, nf, ev_op_iterations, dim):
    """Compute the QED leading order 'exact' evolution.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target strong coupling value
    a0 : float
        initial strong coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps
    dim : int
        dimension of the matrix sector

    Returns
    -------
    e_s^{order} : numpy.ndarray
        singlet QEDxQCD iterated (exact) EKO
    """
    return eko_iterate(
        gamma_singlet, a1, a0, aem_list, nf, (1, 2), ev_op_iterations, dim
    )


@nb.njit(cache=True)
def u_vec(r, ev_op_max_order, dim):
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
    u = np.zeros((ev_op_max_order[0], dim, dim), np.complex_)  # k = 0 .. max_order
    # init
    u[0] = np.identity(dim, np.complex_)
    _, w, e = ad.exp_matrix(r[0])
    for i in range(dim):
        e[i] = np.ascontiguousarray(e[i])
    for kk in range(1, ev_op_max_order[0]):
        # compute R'
        rp = np.zeros((dim, dim), dtype=np.complex_)
        for jj in range(kk):
            rp += np.ascontiguousarray(r[kk - jj]) @ u[jj]
        # now compose U
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    u[kk] += e[i] @ rp @ e[i] / kk
                else:
                    u[kk] += e[i] @ rp @ e[j] / (w[j] - w[i] + kk)
    return u


@nb.njit(cache=True)
def eko_perturbative(
    gamma_singlet,
    a1,
    a0,
    aem_list,
    nf,
    order,
    ev_op_iterations,
    ev_op_max_order,
    is_exact,
    dim,
):
    """Singlet |NLO|,|NNLO| or |N3LO| perturbative EKO, depending on which r is passed.

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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    e = np.identity(dim, np.complex_)
    # iterate elements
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    al = a_steps[0]
    for step, ah in enumerate(a_steps[1:]):
        betalist[0] += aem_list[step] * beta.beta_qcd((2, 1), nf)
        r = s.r_vec(
            contract_gammas(gamma_singlet, aem_list[step])[1:],
            betalist,
            ev_op_max_order,
            order,
            is_exact,
            dim,
        )
        uk = u_vec(r, ev_op_max_order, dim)
        e0 = qed_lo(gamma_singlet, ah, al, aem_list, nf, ev_op_iterations=1, dim=dim)
        uh = s.sum_u(uk, ah, dim)
        ul = s.sum_u(uk, al, dim)
        # join elements
        ek = np.ascontiguousarray(uh) @ np.ascontiguousarray(e0) @ np.linalg.inv(ul)
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def eko_truncated(gamma_singlet, a1, a0, aem_list, nf, order, ev_op_iterations, dim):
    """Singlet |NLO|, |NNLO| or |N3LO| truncated EKO.

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
    order : tuple(int,int)
        perturbative order
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    numpy.ndarray
        singlet truncated EKO
    """
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    e = np.identity(dim, np.complex_)
    # iterate elements
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    al = a_steps[0]
    for step, ah in enumerate(a_steps[1:]):
        betalist[0] += aem_list[step] * beta.beta_qcd((2, 1), nf)
        r = s.r_vec(
            contract_gammas(gamma_singlet, aem_list[step])[1:],
            betalist,
            order,
            order,
            False,
            dim,
        )
        u = u_vec(r, order, dim)
        u1 = np.ascontiguousarray(u[1])
        e0 = np.ascontiguousarray(
            qed_lo(gamma_singlet, ah, al, aem_list, nf, ev_op_iterations=1, dim=dim)
        )
        if order[0] >= 2:
            ek = e0 + ah * u1 @ e0 - al * e0 @ u1
        if order[0] >= 3:
            u2 = np.ascontiguousarray(u[2])
            ek += (
                +(ah**2) * u2 @ e0
                - ah * al * u1 @ e0 @ u1
                + al**2 * e0 @ (u1 @ u1 - u2)
            )
        if order[0] >= 4:
            u3 = np.ascontiguousarray(u[3])
            ek += (
                +(ah**3) * u3 @ e0
                - ah**2 * al * u2 @ e0 @ u1
                + ah * al**2 * u1 @ e0 @ (u1 @ u1 - u2)
                - al**3 * e0 @ (u1 @ u1 @ u1 - u1 @ u2 - u2 @ u1 + u3)
            )
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_singlet,
    a1,
    a0,
    aem_list,
    nf,
    ev_op_iterations,
    ev_op_max_order,
):
    """Determine used kernel and call it.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative order
    method : str
        method
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U

    Returns
    -------
    e_s : numpy.ndarray
        singlet EKO
    """
    if method in ["iterate-exact", "iterate-expanded"]:
        return eko_iterate(
            gamma_singlet, a1, a0, aem_list, nf, order, ev_op_iterations, dim=4
        )
    if method == "perturbative-exact":
        return eko_perturbative(
            gamma_singlet,
            a1,
            a0,
            aem_list,
            nf,
            order,
            ev_op_iterations,
            ev_op_max_order,
            True,
            dim=4,
        )
    if method == "perturbative-expanded":
        return eko_perturbative(
            gamma_singlet,
            a1,
            a0,
            aem_list,
            nf,
            order,
            ev_op_iterations,
            ev_op_max_order,
            False,
            dim=4,
        )
    if method in ["truncated", "ordered-truncated"]:
        return eko_truncated(
            gamma_singlet, a1, a0, aem_list, nf, order, ev_op_iterations, dim=4
        )
    raise NotImplementedError("selected method is not implemented")
