# -*- coding: utf-8 -*-
"""Collection of QED non-singlet EKOs."""
import numba as nb
import numpy as np

from .. import beta
from . import evolution_integrals as ei
from . import non_singlet, utils


@nb.njit(cache=True)
def lo_aem1_exact(gamma_ns, a1, a0, aem, nf):  # lo refers to the order in as1
    """
    O(as1aem1) non-singlet exact EKO.

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
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j00_qed(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * ei.jm10(a1, a0, aem, nf)
    )


@nb.njit(cache=True)
def lo_aem2_exact(gamma_ns, a1, a0, aem, nf):  # lo refers to the order in as1
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
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j00_qed(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * ei.jm10(a1, a0, aem, nf)
    )


@nb.njit(cache=True)
def nlo_aem1_exact(gamma_ns, a1, a0, aem, nf):
    """
    O(as2aem1) non-singlet exact EKO.

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
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j01_exact_qed(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j11_exact_qed(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * ei.jm11_exact(a1, a0, aem, nf)
    )


@nb.njit(cache=True)
def nlo_aem2_exact(gamma_ns, a1, a0, aem, nf):
    """
    O(as2aem2) non-singlet exact EKO.

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
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j01_exact_qed(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j11_exact_qed(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2])
        * ei.jm11_exact(a1, a0, aem, nf)
    )


@nb.njit(cache=True)
def nnlo_aem1_exact(gamma_ns, a1, a0, aem, nf):
    """
    O(as3aem1) non-singlet exact EKO.

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
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j02_exact_qed(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j12_exact_qed(a1, a0, aem, nf)
        + gamma_ns[3, 0] * ei.j22_exact_qed(a1, a0, aem, nf)
        + aem * gamma_ns[0, 1] * ei.jm12_exact(a1, a0, aem, nf)
    )


@nb.njit(cache=True)
def nnlo_aem2_exact(gamma_ns, a1, a0, aem, nf):
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
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j02_exact_qed(a1, a0, aem, nf)
        + gamma_ns[2, 0] * ei.j12_exact_qed(a1, a0, aem, nf)
        + gamma_ns[3, 0] * ei.j22_exact_qed(a1, a0, aem, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2])
        * ei.jm12_exact(a1, a0, aem, nf)
    )

@nb.njit(cache=True)
def solution_running_alpha(func, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations):
    """
    ...
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    al = a_steps[0]
    res = np.prod(
        [ func(gamma_ns, ah, a_steps[step], aem_list[step], nf) for step, ah in enumerate(a_steps[1:])]
    )
    return res


@nb.njit(cache=True)
def dispatcher(
    order, method, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations
):  # pylint: disable=too-many-return-statements
    """
    Determine used kernel and call it.

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
        aem : float
            electromagnetic coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_ns : complex
            non-singlet EKO
    """
    # use always exact in LO
    if order[1] == 0:
        return non_singlet.dispatcher(
            order, method, gamma_ns[1:, 0], a1, a0, nf, ev_op_iterations
        )
        # this if is probably useless since when order[1] == 0
        # the code never enters in this module
    if order[1] == 1:
        if order[0] == 1:
            return solution_running_alpha(lo_aem1_exact, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations)
        if order[0] == 2:
            return solution_running_alpha(nlo_aem1_exact, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations)
        if order[0] == 3:
            return solution_running_alpha(nnlo_aem1_exact, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations)
    if order[1] == 2:
        if order[0] == 1:
            return solution_running_alpha(lo_aem2_exact, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations)
        if order[0] == 2:
            return solution_running_alpha(nlo_aem2_exact, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations)
        if order[0] == 3:
            return solution_running_alpha(nnlo_aem2_exact, gamma_ns, a1, a0, aem_list, nf, ev_op_iterations)
    raise NotImplementedError("Selected order is not implemented")
