# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

from .. import beta
from . import evolution_integrals as ei
from . import non_singlet, utils


@nb.njit(cache=True)
def lo_aem1_exact(gamma_ns, a1, a0, aem, nf):  # lo refers to the order in as1
    """
    |LO| non-singlet exact EKO

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
        e_ns^0 : complex
            |LO| non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j00(a1, a0, nf)
        + aem * gamma_ns[0, 1] * ei.jm10(a1, a0, nf)
    )


@nb.njit(cache=True)
def lo_aem2_exact(gamma_ns, a1, a0, aem, nf):  # lo refers to the order in as1
    """
    |LO| non-singlet exact EKO

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
        e_ns^0 : complex
            |LO| non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j00(a1, a0, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * ei.jm10(a1, a0, nf)
    )


@nb.njit(cache=True)
def nlo_aem1_exact(gamma_ns, a1, a0, aem, nf):
    """
    |NLO| non-singlet exact EKO

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
        e_ns^1 : complex
            |NLO| non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j01_exact(a1, a0, nf)
        + gamma_ns[2, 0] * ei.j11_exact(a1, a0, nf)
        + aem * gamma_ns[0, 1] * ei.jm11_exact(a1, a0, nf)
    )


@nb.njit(cache=True)
def nlo_aem2_exact(gamma_ns, a1, a0, aem, nf):
    """
    |NLO| non-singlet exact EKO

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
        e_ns^1 : complex
            |NLO| non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j01_exact(a1, a0, nf)
        + gamma_ns[2, 0] * ei.j11_exact(a1, a0, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * ei.jm11_exact(a1, a0, nf)
    )


@nb.njit(cache=True)
def nnlo_aem1_exact(gamma_ns, a1, a0, aem, nf):
    """
    |NNLO| non-singlet exact EKO

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
        e_ns^2 : complex
            |NNLO| non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j02_exact(a1, a0, nf)
        + gamma_ns[2, 0] * ei.j12_exact(a1, a0, nf)
        + gamma_ns[3, 0] * ei.j22_exact(a1, a0, nf)
        + aem * gamma_ns[0, 1] * ei.jm12_exact(a1, a0, nf)
    )


@nb.njit(cache=True)
def nnlo_aem2_exact(gamma_ns, a1, a0, aem, nf):
    """
    |NNLO| non-singlet exact EKO

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
        e_ns^2 : complex
            |NNLO| non-singlet exact EKO
    """
    return np.exp(
        (gamma_ns[1, 0] + aem * gamma_ns[1, 1]) * ei.j02_exact(a1, a0, nf)
        + gamma_ns[2, 0] * ei.j12_exact(a1, a0, nf)
        + gamma_ns[3, 0] * ei.j22_exact(a1, a0, nf)
        + (aem * gamma_ns[0, 1] + aem**2 * gamma_ns[0, 2]) * ei.jm12_exact(a1, a0, nf)
    )


@nb.njit(cache=True)
def dispatcher(
    order, method, gamma_ns, a1, a0, aem, nf, ev_op_iterations
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
            order, method, gamma_ns[0], a1, a0, nf, ev_op_iterations
        )
    if order[1] == 1:
        if order[0] == 1:
            return lo_aem1_exact(gamma_ns, a1, a0, aem, nf)
        if order[0] == 2:
            return nlo_aem1_exact(gamma_ns, a1, a0, aem, nf)
        if order[0] == 3:
            return nnlo_aem1_exact(gamma_ns, a1, a0, aem, nf)
    if order[1] == 2:
        if order[0] == 1:
            return lo_aem2_exact(gamma_ns, a1, a0, aem, nf)
        if order[0] == 2:
            return nlo_aem2_exact(gamma_ns, a1, a0, aem, nf)
        if order[0] == 3:
            return nnlo_aem2_exact(gamma_ns, a1, a0, aem, nf)
    raise NotImplementedError("Selected order is not implemented")
