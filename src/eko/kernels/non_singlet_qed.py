"""Collection of QED non-singlet EKOs."""
import numba as nb
import numpy as np

from .. import beta
from . import non_singlet as ns
from . import utils


@nb.njit(cache=True)
def contract_gammas(gamma_ns, aem):
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
        [aem**i for i in range(gamma_ns.shape[1])], dtype=np.complex_
    )
    return gamma_ns @ vec_alphaem


@nb.njit(cache=True)
def as0_exact(gamma_pure_qed, mu2_from, mu2_to):
    """Apply pure QED evolution to QCD kernel.

    Parameters
    ----------
    gamma_ns : float
        pure QED part of the AD
    mu2_from : float
        initial value of mu2
    mu2_from : float
        final value of mu2

    Returns
    -------
    exp : float
        pure QED evolution kernel

    """
    return np.exp(gamma_pure_qed * np.log(mu2_from / mu2_to))


@nb.njit(cache=True)
def as1_exact(gamma_ns, a1, a0, beta):
    """O(as1aem1) non-singlet exact EKO.

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
        O(as1aem1) non-singlet exact EKO
    """
    return ns.lo_exact(gamma_ns, a1, a0, beta)


@nb.njit(cache=True)
def as2_exact(gamma_ns, a1, a0, beta):
    """O(as2aem1) non-singlet exact EKO.

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
        O(as2aem1) non-singlet exact EKO
    """
    return ns.nlo_exact(gamma_ns, a1, a0, beta)


@nb.njit(cache=True)
def as2_expanded(gamma_ns, a1, a0, beta):
    """O(as2aem1) non-singlet exact EKO.

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
        O(as2aem1) non-singlet exact EKO
    """
    return ns.nlo_expanded(gamma_ns, a1, a0, beta)


@nb.njit(cache=True)
def as3_exact(gamma_ns, a1, a0, beta):
    """O(as3aem1) non-singlet exact EKO.

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
        O(as3aem1) non-singlet exact EKO
    """
    return ns.nnlo_exact(gamma_ns, a1, a0, beta)


@nb.njit(cache=True)
def as3_expanded(gamma_ns, a1, a0, beta):
    """O(as3aem1) non-singlet exact EKO.

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
        O(as3aem1) non-singlet exact EKO
    """
    return ns.nnlo_expanded(gamma_ns, a1, a0, beta)


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_ns,
    as_list,
    aem_half,
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
    mu2_from : float
        initial value of mu2
    mu2_from : float
        final value of mu2

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    betas = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    betamix = beta.beta_qcd((2, 1), nf)
    if not alphaem_running:
        aem = aem_half[0]
        betas[0] += aem * betamix
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        return fixed_alphaem(
            gamma_ns_list,
            as_list[-1],
            as_list[0],
            mu2_to,
            mu2_from,
            betas,
            order,
            ev_op_iterations,
            method,
        )
    mu2_steps = utils.geomspace(mu2_from, mu2_to, 1 + ev_op_iterations)
    res = 1.0
    for step in range(1, ev_op_iterations + 1):
        aem = aem_half[step - 1]
        betas[0] += aem * betamix
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        a1 = as_list[step]
        a0 = as_list[step - 1]
        mu2_from = mu2_steps[step - 1]
        mu2_to = mu2_steps[step]
        res *= fixed_alphaem(
            gamma_ns_list,
            a1,
            a0,
            mu2_to,
            mu2_from,
            betas,
            order,
            1,
            method,
        )
    return res


@nb.njit(cache=True)
def pure_qcd_exact(order, gamma_ns_list, a1, a0, betalist):
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
    mu2_from : float
        initial value of mu2
    mu2_from : float
        final value of mu2

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    if order[0] == 1:
        return as1_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 2:
        return as2_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 3:
        return as3_exact(gamma_ns_list[1:], a1, a0, betalist)
    else:
        raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def pure_qcd_expanded(order, gamma_ns_list, a1, a0, betalist):
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
    mu2_from : float
        initial value of mu2
    mu2_from : float
        final value of mu2

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    if order[0] == 1:
        return as1_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 2:
        return as2_expanded(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 3:
        return as3_expanded(gamma_ns_list[1:], a1, a0, betalist)
    else:
        raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def fixed_alphaem(
    gamma_ns_list, a1, a0, mu2_to, mu2_from, beta, order, ev_op_iterations, method
):
    """Select method to compute the QCD part.

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
    # TODO : in the case of aem running
    if method == "ordered-truncated":
        tmp = ns.eko_ordered_truncated(
            gamma_ns_list[1:], a1, a0, beta, order, ev_op_iterations
        )
    elif method == "truncated":
        tmp = ns.eko_truncated(gamma_ns_list[1:], a1, a0, beta, order, ev_op_iterations)
    elif method in [
        "iterate-expanded",
        "decompose-expanded",
        "perturbative-expanded",
    ]:
        tmp = pure_qcd_expanded(order, gamma_ns_list, a1, a0, beta)
    else:
        tmp = pure_qcd_exact(order, gamma_ns_list, a1, a0, beta)
    return tmp * as0_exact(gamma_ns_list[0], mu2_from, mu2_to)
