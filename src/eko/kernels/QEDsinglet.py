# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import utils
from .singlet import lo_exact


@nb.njit(cache=True)
def eko_iterate(gamma_singlet, a1, a0, aem, nf, order, ev_op_iterations):
    """Singlet QEDxQCD iterated (exact) EKO

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
    e = np.identity(4, np.complex_)
    al = a_steps[0]
    #    betaQCD = np.array(
    #        [
    #            [
    #                beta.beta_qcd((2, 0), nf),
    #                beta.beta_qcd((3, 0), nf),
    #                beta.beta_qcd((4, 0), nf),
    #                beta.beta_qcd((5, 0), nf),
    #            ],
    #            [beta.beta_qcd((2, 1), nf), 0, 0, 0],
    #            [0, 0, 0, 0],
    #        ]
    #    )
    betaQCD = np.zeros((4, 3), np.complex_)
    for i in range(4):
        betaQCD[i, 0] = beta.beta_qcd((i + 2, 0), nf)
    betaQCD[0, 1] = beta.beta_qcd((2, 1), nf)
    for ah in a_steps[1:]:
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        gamma = np.zeros((4, 4), np.complex_)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += a_half**2 * betaQCD[i, j] * a_half**i * aem**j
                gamma += gamma_singlet[i, j] * a_half**i * aem**j
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix(ln)[0])
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def dispatcher(  # pylint: disable=too-many-return-statements
    order, method, gamma_singlet, a1, a0, aem, nf, ev_op_iterations, ev_op_max_order
):
    """
    Determine used kernel and call it.

    In LO we always use the exact solution.

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
    # use always exact in LO
    #    if order == (1,0):
    #        return lo_exact(gamma_singlet, a1, a0, nf)

    if method in ["iterate-exact", "iterate-expanded"]:
        return eko_iterate(gamma_singlet, a1, a0, aem, nf, order, ev_op_iterations)

    #    if method == "perturbative-exact":
    #        return eko_perturbative(
    #            gamma_singlet, a1, a0, nf, order, ev_op_iterations, ev_op_max_order, True
    #        )
    #    if method == "perturbative-expanded":
    #        return eko_perturbative(
    #            gamma_singlet, a1, a0, nf, order, ev_op_iterations, ev_op_max_order, False
    #        )
    #    if method in ["truncated", "ordered-truncated"]:
    #        return eko_truncated(gamma_singlet, a1, a0, nf, order, ev_op_iterations)
    #    # These methods are scattered for nlo and nnlo
    #    if method == "decompose-exact":
    #        if order[0] == 2:
    #            return nlo_decompose_exact(gamma_singlet, a1, a0, nf)
    #        return nnlo_decompose_exact(gamma_singlet, a1, a0, nf)
    #    if method == "decompose-expanded":
    #        if order[0] == 2:
    #            return nlo_decompose_expanded(gamma_singlet, a1, a0, nf)
    #        return nnlo_decompose_expanded(gamma_singlet, a1, a0, nf)
    raise NotImplementedError("Selected method is not implemented")
