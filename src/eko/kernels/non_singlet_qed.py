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
    aem_steps = 1 if not alphaem_running else ev_op_iterations
    a_steps = utils.geomspace(a0, a1, 1 + aem_steps)
    res = 1.0
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    for step in range(1, aem_steps + 1):
        aem = aem_list[step - 1]
        a1 = a_steps[step]
        a0 = a_steps[step - 1]
        betalist[0] += aem * beta.beta_qcd((2, 1), nf)
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        # Observe that in this way ordered_truncated and truncated are correct only in the case of
        # aem fixed. In order to handle also aem running, they have to be reimplemented in a similar
        # way w.r.t. the function eko_iterate in singlet_qed (the reason is that they involve an iteration
        # on an object that is aem dependent)
        res *= choose_method_qcd(
            gamma_ns_list[1:], a1, a0, betalist, order, ev_op_iterations, method
        )
    # TODO : we should divide also the mu_integral in steps in order to attach the
    # QED solution with aem running. For the moment we use the last value of aem.
    # For aem fixed nothing changes.
    res *= as0_exact(gamma_ns_list[0], mu2_from, mu2_to)
    return res


@nb.njit(cache=True)
def exact(order, gamma_ns, a1, a0, beta):
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
    if order[0] == 1:
        return ns.lo_exact(gamma_ns, a1, a0, beta)
    elif order[0] == 2:
        return ns.nlo_exact(gamma_ns, a1, a0, beta)
    elif order[0] == 3:
        return ns.nnlo_exact(gamma_ns, a1, a0, beta)
    else:
        raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def expanded(order, gamma_ns, a1, a0, beta):
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
    if order[0] == 1:
        return ns.lo_exact(gamma_ns, a1, a0, beta)
    elif order[0] == 2:
        return ns.nlo_expanded(gamma_ns, a1, a0, beta)
    elif order[0] == 3:
        return ns.nnlo_expanded(gamma_ns, a1, a0, beta)
    else:
        raise NotImplementedError("Selected order is not implemented")


@nb.njit(cache=True)
def choose_method_qcd(gamma_ns, a1, a0, beta, order, ev_op_iterations, method):
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
    if method == "ordered-truncated":
        return ns.eko_ordered_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)
    if method == "truncated":
        return ns.eko_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)
    if method in [
        "iterate-expanded",
        "decompose-expanded",
        "perturbative-expanded",
    ]:
        return expanded(order, gamma_ns, a1, a0, beta)
    return exact(order, gamma_ns, a1, a0, beta)
