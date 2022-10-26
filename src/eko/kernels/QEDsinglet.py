# -*- coding: utf-8 -*-
"""Collection of QED singlet EKOs."""
import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import utils


@nb.njit(cache=True)
def eko_iterate(gamma_singlet, a1, a0, aem_list, nf, order, ev_op_iterations):
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
    e = np.identity(4, np.complex_)
    al = a_steps[0]
    betaQCD = np.zeros((4, 3))
    for i in range(1, 3 + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for (step, ah) in enumerate(a_steps[1:]):
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        gamma = np.zeros((4, 4), np.complex_)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += betaQCD[i, j] * a_half ** (i + 1) * aem_list[step] ** j
                gamma += gamma_singlet[i, j] * a_half**i * aem_list[step] ** j
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix(ln)[0])
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def dispatcher(  # pylint: disable=too-many-return-statements
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
    """
    Determine used kernel and call it.

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
        return eko_iterate(gamma_singlet, a1, a0, aem_list, nf, order, ev_op_iterations)
    raise NotImplementedError("Selected method is not implemented")
