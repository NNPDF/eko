"""Collection of QED non-singlet EKOs."""
import numba as nb
import numpy as np

from . import evolution_integrals_qed as ei
from . import utils
from .non_singlet import U_vec, lo_exact


@nb.njit(cache=True)
def as1aem1(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as1aem1) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^0 : complex
        O(as1aem1) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j12(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as1aem2(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """
    O(as1aem2) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^0 : complex
        O(as1aem2) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j12(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as2aem1_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as2aem1) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^1 : complex
        O(as2aem1) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j13_exact(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j23_exact(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as2aem1_expanded(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as2aem1) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^1 : complex
        O(as2aem1) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j13_expanded(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j23_expanded(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as2aem2_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as2aem2) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^1 : complex
        O(as2aem2) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j13_exact(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j23_exact(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as2aem2_expanded(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as2aem2) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^1 : complex
        O(as2aem2) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j13_expanded(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j23_expanded(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as3aem1_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as3aem1) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
            electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^2 : complex
        O(as3aem1) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j14_exact(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j24_exact(a1, a0, aem, nf)
        + gamma_ns[3, 0] * ei.j34_exact(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as3aem1_expanded(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """O(as3aem1) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
            electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^2 : complex
        O(as3aem1) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j14_expanded(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j24_expanded(a1, a0, aem, nf)
        + gamma_ns[3, 0] * ei.j34_expanded(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as3aem2_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """
    O(as3aem2) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^2 : complex
        O(as3aem2) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j14_exact(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j24_exact(a1, a0, aem, nf)
        + gamma_ns[3, 0] * ei.j34_exact(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def as3aem2_expanded(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """
    O(as3aem2) non-singlet exact EKO.

    Parameters
    ----------
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^2 : complex
        O(as3aem2) non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j14_expanded(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j24_expanded(a1, a0, aem, nf)
        + gamma_ns[3, 0] * ei.j34_expanded(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * np.log(mu2_from / mu2_to)
    )


@nb.njit(cache=True)
def eko_truncated(
    gamma_ns,
    a1,
    a0,
    nf,
    order,
    ev_op_iterations,
    aem_list,
    mu2_to,
    mu2_from,
):
    """|NLO|, |NNLO| or |N3LO| non-singlet truncated EKO.

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
    gamma_qcd = gamma_ns[1:, 0]
    U = U_vec(gamma_qcd, nf, order)
    e = 1.0
    al = a_steps[0]
    fact = U[0]
    for step in range(1, ev_op_iterations + 1):
        ah = a_steps[step]
        aem = aem_list[step - 1]
        gamma_qed = 0.0
        for j in range(order[1] + 1):
            gamma_qed += aem**j * gamma_qcd[0, j]
        e0 = lo_exact(gamma_qcd, ah, al, nf)
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
        fact += gamma_qed * np.log(mu2_from / mu2_to)
        e *= e0 * fact
        al = ah
    return e


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_ns,
    a1,
    a0,
    aem_list,
    alphaem_running,
    nf,
    ev_op_iterations,
    mu2_to,
    mu2_from,
):
    r"""Determine used kernel and call it.

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
    aem_list : numpy.ndarray
        electromagnetic coupling values
    alphaem_running : Bool
        running of alphaem
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    if method == "truncated":
        return eko_truncated(
            gamma_ns, a1, a0, nf, order, ev_op_iterations, aem_list, mu2_to, mu2_from
        )
    if not alphaem_running:
        aem = aem_list[0]
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return fixed_alphaem_expanded(
                order, gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from
            )
        return fixed_alphaem_exact(order, gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
    else:
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return running_alphaem_expanded(
                order, gamma_ns, a1, a0, aem, nf, ev_op_iterations, mu2_to, mu2_from
            )
        return running_alphaem_exact(
            order, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations, mu2_to, mu2_from
        )


@nb.njit(cache=True)
def fixed_alphaem_exact(order, gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
    """Compute exact solution for fixed alphaem.

    Parameters
    ----------
    order : tuple(int,int)
        perturbation order
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    if order[1] == 1:
        if order[0] == 1:
            return as1aem1(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
        if order[0] == 2:
            return as2aem1_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
        if order[0] == 3:
            return as3aem1_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
    if order[1] == 2:
        if order[0] == 1:
            return as1aem2(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
        if order[0] == 2:
            return as2aem2_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
        if order[0] == 3:
            return as3aem2_exact(gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from)
    raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def fixed_alphaem_expanded(order, gamma_ns, a1, a0, aem, nf):
    """Compute exact solution for fixed alphaem.

    Parameters
    ----------
    order : tuple(int,int)
        perturbation order
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    if order[1] == 1:
        if order[0] == 1:
            return as1aem1(gamma_ns, a1, a0, aem, nf)
        if order[0] == 2:
            return as2aem1_expanded(gamma_ns, a1, a0, aem, nf)
        if order[0] == 3:
            return as3aem1_expanded(gamma_ns, a1, a0, aem, nf)
    if order[1] == 2:
        if order[0] == 1:
            return as1aem2(gamma_ns, a1, a0, aem, nf)
        if order[0] == 2:
            return as2aem2_expanded(gamma_ns, a1, a0, aem, nf)
        if order[0] == 3:
            return as3aem2_expanded(gamma_ns, a1, a0, aem, nf)
    raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def running_alphaem_exact(
    order, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations, mu2_to, mu2_from
):
    """Compute exact solution for running alphaem.

    Parameters
    ----------
    order : tuple(int,int)
        perturbation order
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem_list : numpy.ndarray
        electromagnetic coupling values
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    res = 1.0
    # For the moment implemented in this way to make numba compile it
    # TODO : implement it with np.prod in a way that numba compiles it
    if order[1] == 1:
        if order[0] == 1:
            for step in range(1, ev_op_iterations + 1):
                res *= as1aem1(
                    gamma_ns,
                    a_steps[step],
                    a_steps[step - 1],
                    aem_list[step - 1],
                    nf,
                    mu2_to,
                    mu2_from,
                )
            return res
        if order[0] == 2:
            for step in range(1, ev_op_iterations + 1):
                res *= as2aem1_exact(
                    gamma_ns,
                    a_steps[step],
                    a_steps[step - 1],
                    aem_list[step - 1],
                    nf,
                    mu2_to,
                    mu2_from,
                )
            return res
        if order[0] == 3:
            for step in range(1, ev_op_iterations + 1):
                res *= as3aem1_exact(
                    gamma_ns,
                    a_steps[step],
                    a_steps[step - 1],
                    aem_list[step - 1],
                    nf,
                    mu2_to,
                    mu2_from,
                )
            return res
    if order[1] == 2:
        if order[0] == 1:
            for step in range(1, ev_op_iterations + 1):
                res *= as1aem2(
                    gamma_ns,
                    a_steps[step],
                    a_steps[step - 1],
                    aem_list[step - 1],
                    nf,
                    mu2_to,
                    mu2_from,
                )
            return res
        if order[0] == 2:
            for step in range(1, ev_op_iterations + 1):
                res *= as2aem2_exact(
                    gamma_ns,
                    a_steps[step],
                    a_steps[step - 1],
                    aem_list[step - 1],
                    nf,
                    mu2_to,
                    mu2_from,
                )
            return res
        if order[0] == 3:
            for step in range(1, ev_op_iterations + 1):
                res *= as3aem2_exact(
                    gamma_ns,
                    a_steps[step],
                    a_steps[step - 1],
                    aem_list[step - 1],
                    nf,
                    mu2_to,
                    mu2_from,
                )
            return res
    raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def running_alphaem_expanded(order, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations):
    """Compute exact solution for running alphaem.

    Parameters
    ----------
    order : tuple(int,int)
        perturbation order
    gamma_ns : numpy.ndarray
        non-singlet anomalous dimensions
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem_list : numpy.ndarray
        electromagnetic coupling values
    nf : int
        number of active flavors
    ev_op_iterations : int
        number of evolution steps

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    res = 1.0
    # For the moment implemented in this way to make numba compile it
    # TODO : implement it with np.prod in a way that numba compiles it
    if order[1] == 1:
        if order[0] == 1:
            for step in range(1, ev_op_iterations + 1):
                res *= as1aem1(
                    gamma_ns, a_steps[step], a_steps[step - 1], aem_list[step - 1], nf
                )
            return res
        if order[0] == 2:
            for step in range(1, ev_op_iterations + 1):
                res *= as2aem1_expanded(
                    gamma_ns, a_steps[step], a_steps[step - 1], aem_list[step - 1], nf
                )
            return res
        if order[0] == 3:
            for step in range(1, ev_op_iterations + 1):
                res *= as3aem1_expanded(
                    gamma_ns, a_steps[step], a_steps[step - 1], aem_list[step - 1], nf
                )
            return res
    if order[1] == 2:
        if order[0] == 1:
            for step in range(1, ev_op_iterations + 1):
                res *= as1aem2(
                    gamma_ns, a_steps[step], a_steps[step - 1], aem_list[step - 1], nf
                )
            return res
        if order[0] == 2:
            for step in range(1, ev_op_iterations + 1):
                res *= as2aem2_expanded(
                    gamma_ns, a_steps[step], a_steps[step - 1], aem_list[step - 1], nf
                )
            return res
        if order[0] == 3:
            for step in range(1, ev_op_iterations + 1):
                res *= as3aem2_expanded(
                    gamma_ns, a_steps[step], a_steps[step - 1], aem_list[step - 1], nf
                )
            return res
    raise NotImplementedError("Selected order is not implemented")
