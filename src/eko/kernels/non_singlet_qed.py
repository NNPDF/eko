"""Collection of QED non-singlet EKOs."""
import numba as nb
import numpy as np

from .. import beta
from . import non_singlet as ns
from . import utils
from .non_singlet import U_vec, lo_exact


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
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

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
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

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
    aem : float
            electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^2 : complex
        O(as3aem1) non-singlet exact EKO
    """
    return ns.nnlo_exact(gamma_ns, a1, a0, beta)


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
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^1 : complex
        O(as2aem1) non-singlet exact EKO
    """
    return ns.nlo_expanded(gamma_ns, a1, a0, beta)


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
    aem : float
            electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    e_ns^2 : complex
        O(as3aem1) non-singlet exact EKO
    """
    return ns.nnlo_expanded(gamma_ns, a1, a0, beta)


# For eko_truncated and eko_ordered_truncated is not sufficient
# to call the non_singlet ones with the new gamma, but they have to be reimplemented
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
    raise NotImplementedError("eko_truncated for qed is not implemented yet")
    # return ns.eko_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)


@nb.njit(cache=True)
def eko_ordered_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations):
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
    raise NotImplementedError("eko_ordered_truncated for qed is not implemented yet")
    # return ns.eko_ordered_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)


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
    if not alphaem_running:
        aem = aem_list[0]
        if method == "ordered-truncated":
            return fixed_alphaem_ordered_truncated(
                gamma_ns, a1, a0, aem, nf, order, mu2_to, mu2_from, ev_op_iterations
            )
        if method == "truncated":
            return fixed_alphaem_truncated(
                gamma_ns, a1, a0, aem, nf, order, mu2_to, mu2_from, ev_op_iterations
            )
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
        if method == "ordered-truncated":
            return running_alphaem_ordered_truncated(
                gamma_ns,
                a1,
                a0,
                aem_list,
                nf,
                order,
                mu2_to,
                mu2_from,
                ev_op_iterations,
            )
        if method == "truncated":
            return running_alphaem_truncated(
                gamma_ns,
                a1,
                a0,
                aem_list,
                nf,
                order,
                mu2_to,
                mu2_from,
                ev_op_iterations,
            )
        if method in [
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            return running_alphaem_expanded(
                order,
                gamma_ns,
                a1,
                a0,
                aem_list,
                nf,
                ev_op_iterations,
                mu2_to,
                mu2_from,
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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    betalist[0] += aem * beta.beta_qcd((2, 1), nf)
    gamma_ns_list = contract_gammas(gamma_ns, aem)
    if order[0] == 1:
        qcd_only = as1_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 2:
        qcd_only = as2_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 3:
        qcd_only = as3_exact(gamma_ns_list[1:], a1, a0, betalist)
    else:
        raise NotImplementedError("Selected order is not implemented")
    return qcd_only * apply_qed(gamma_ns_list[0], mu2_from, mu2_to)


@nb.njit(cache=True)
def fixed_alphaem_expanded(order, gamma_ns, a1, a0, aem, nf, mu2_to, mu2_from):
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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    betalist[0] += aem * beta.beta_qcd((2, 1), nf)
    gamma_ns_list = contract_gammas(gamma_ns, aem)
    if order[0] == 1:
        qcd_only = as1_exact(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 2:
        qcd_only = as2_expanded(gamma_ns_list[1:], a1, a0, betalist)
    elif order[0] == 3:
        qcd_only = as3_expanded(gamma_ns_list[1:], a1, a0, betalist)
    else:
        raise NotImplementedError("Selected order is not implemented")
    return qcd_only * apply_qed(gamma_ns_list[0], mu2_from, mu2_to)


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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    # For the moment implemented in this way to make numba compile it
    # TODO : implement it with np.prod in a way that numba compiles it
    for step in range(1, ev_op_iterations + 1):
        aem = aem_list[step - 1]
        a1 = a_steps[step]
        a0 = a_steps[step - 1]
        betalist[0] += aem * beta.beta_qcd((2, 1), nf)
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        if order[0] == 1:
            res *= as1_exact(gamma_ns_list[1:], a1, a0, betalist)
        elif order[0] == 2:
            res *= as2_exact(gamma_ns_list[1:], a1, a0, betalist)
        elif order[0] == 3:
            res *= as3_exact(gamma_ns_list[1:], a1, a0, betalist)
        else:
            raise NotImplementedError("Selected order is not implemented")
        res *= apply_qed(gamma_ns_list[0], mu2_from, mu2_to)
    return res


@nb.njit(cache=True)
def running_alphaem_expanded(
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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    # For the moment implemented in this way to make numba compile it
    # TODO : implement it with np.prod in a way that numba compiles it
    for step in range(1, ev_op_iterations + 1):
        aem = aem_list[step - 1]
        a1 = a_steps[step]
        a0 = a_steps[step - 1]
        betalist[0] += aem * beta.beta_qcd((2, 1), nf)
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        if order[0] == 1:
            res *= as1_exact(gamma_ns_list[1:], a1, a0, betalist)
        elif order[0] == 2:
            res *= as2_expanded(gamma_ns_list[1:], a1, a0, betalist)
        elif order[0] == 3:
            res *= as3_expanded(gamma_ns_list[1:], a1, a0, betalist)
        else:
            raise NotImplementedError("Selected order is not implemented")
        res *= apply_qed(gamma_ns_list[0], mu2_from, mu2_to)
    return res


@nb.njit(cache=True)
def fixed_alphaem_truncated(
    gamma_ns, a1, a0, aem, nf, order, mu2_to, mu2_from, ev_op_iterations
):
    """Compute truncated solution for fixed alphaem.

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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    betalist[0] += aem * beta.beta_qcd((2, 1), nf)
    gamma_ns_list = contract_gammas(gamma_ns, aem)
    qcd_only = eko_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)
    return qcd_only * apply_qed(gamma_ns_list[0], mu2_from, mu2_to)


@nb.njit(cache=True)
def running_alphaem_truncated(
    gamma_ns, a1, a0, aem_list, nf, order, mu2_to, mu2_from, ev_op_iterations
):
    """Compute truncated solution for running alphaem.

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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    # For the moment implemented in this way to make numba compile it
    # TODO : implement it with np.prod in a way that numba compiles it
    for step in range(1, ev_op_iterations + 1):
        aem = aem_list[step - 1]
        a1 = a_steps[step]
        a0 = a_steps[step - 1]
        betalist[0] += aem * beta.beta_qcd((2, 1), nf)
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        res *= eko_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)
        res *= apply_qed(gamma_ns_list[0], mu2_from, mu2_to)
    return res


@nb.njit(cache=True)
def fixed_alphaem_ordered_truncated(
    gamma_ns, a1, a0, aem, nf, order, mu2_to, mu2_from, ev_op_iterations
):
    """Compute ordered-truncated solution for fixed alphaem.

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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    betalist[0] += aem * beta.beta_qcd((2, 1), nf)
    gamma_ns_list = contract_gammas(gamma_ns, aem)
    qcd_only = eko_ordered_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)
    return qcd_only * apply_qed(gamma_ns_list[0], mu2_from, mu2_to)


@nb.njit(cache=True)
def running_alphaem_ordered_truncated(
    gamma_ns, a1, a0, aem_list, nf, order, mu2_to, mu2_from, ev_op_iterations
):
    """Compute ordered-truncated solution for running alphaem.

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
    betalist = [beta.beta_qcd((2 + i, 0), nf) for i in range(order[0])]
    # For the moment implemented in this way to make numba compile it
    # TODO : implement it with np.prod in a way that numba compiles it
    for step in range(1, ev_op_iterations + 1):
        aem = aem_list[step - 1]
        a1 = a_steps[step]
        a0 = a_steps[step - 1]
        betalist[0] += aem * beta.beta_qcd((2, 1), nf)
        gamma_ns_list = contract_gammas(gamma_ns, aem)
        res *= eko_ordered_truncated(gamma_ns, a1, a0, beta, order, ev_op_iterations)
        res *= apply_qed(gamma_ns_list[0], mu2_from, mu2_to)
    return res
