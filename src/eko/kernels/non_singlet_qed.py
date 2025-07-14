"""Collection of QED non-singlet EKOs."""

import numba as nb
import numpy as np

from .. import beta
from . import non_singlet as ns


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
        [aem**i for i in range(gamma_ns.shape[1])], dtype=np.complex128
    )
    return gamma_ns @ vec_alphaem


@nb.njit(cache=True)
def apply_qed(gamma_pure_qed, mu2_from, mu2_to):
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
def as4_exact(gamma_ns, a1, a0, beta):
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
    e_ns^3 : complex
        O(as4aem1) non-singlet exact EKO
    """
    return ns.n3lo_exact(gamma_ns, a1, a0, beta)


@nb.njit(cache=True)
def dispatcher(
    order,
    _method,
    gamma_ns,
    as_list,
    aem_half,
    alphaem_running,
    nf,
    ev_op_iterations,
    mu2_from,
    mu2_to,
):
    r"""Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
    order : tuple(int,int)
        perturbation order
    method : int
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
    return exact(
        order,
        gamma_ns,
        as_list,
        aem_half,
        nf,
        ev_op_iterations,
        mu2_from,
        mu2_to,
    )


@nb.njit(cache=True)
def fixed_alphaem_exact(order, gamma_ns, a1, a0, aem, nf, mu2_from, mu2_to):
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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    betalist[0] += aem * beta.beta_qcd((2, 1), nf)
    gamma_ns_list = contract_gammas(gamma_ns, aem)
    if order[0] == 1:
        qcd_only = as1_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 2:
        qcd_only = as2_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 3:
        qcd_only = as3_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 4:
        qcd_only = as4_exact(gamma_ns_list[1:], a1, a0, betalist)
    else:
        raise NotImplementedError("Selected order is not implemented")
    return qcd_only * apply_qed(gamma_ns_list[0], mu2_from, mu2_to)


@nb.njit(cache=True)
def exact(order, gamma_ns, as_list, aem_half, nf, ev_op_iterations, mu2_from, mu2_to):
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
    mu2_from : float
        initial value of mu2
    mu2_from : float
        final value of mu2

    Returns
    -------
    e_ns : complex
        non-singlet EKO
    """
    mu2_steps = np.geomspace(mu2_from, mu2_to, 1 + ev_op_iterations)
    res = 1.0
    for step in range(1, ev_op_iterations + 1):
        aem = aem_half[step - 1]
        a1 = as_list[step]
        a0 = as_list[step - 1]
        mu2_from = mu2_steps[step - 1]
        mu2_to = mu2_steps[step]
        res *= fixed_alphaem_exact(order, gamma_ns, a1, a0, aem, nf, mu2_from, mu2_to)
    return res
