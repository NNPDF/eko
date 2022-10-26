# -*- coding: utf-8 -*-
"""Collection of QED valence EKOs."""
import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import utils


@nb.njit(cache=True)
def eko_iterate(gamma_valence, a1, a0, aem_list, nf, order, ev_op_iterations):
    """
    Valence iterated (exact) EKO.

    Parameters
    ----------
        gamma_valence : numpy.ndarray
            valence anomalous dimensions matrices
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        aem : float
            electromagnetic coupling value
        nf : int
            number of active flavors
        order : int
            perturbative order
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_v^{order} : numpy.ndarray
            Valence iterated (exact) EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    e = np.identity(2, np.complex_)
    al = a_steps[0]
    betaQCD = np.zeros((4, 3))
    for i in range(1, 3 + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for (step, ah) in enumerate(a_steps[1:]):
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        gamma = np.zeros((2, 2), np.complex_)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += betaQCD[i, j] * a_half ** (i + 1) * aem_list[step] ** j
                gamma += gamma_valence[i, j] * a_half**i * aem_list[step] ** j
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix_2D(ln)[0])
        e = ek @ e
        al = ah
    return e


@nb.njit(cache=True)
def dispatcher(  # pylint: disable=too-many-return-statements
    order,
    method,
    gamma_valence,
    a1,
    a0,
    aem_list,
    nf,
    ev_op_iterations,
    ev_op_max_order,
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
        e_v : numpy.ndarray
            singlet EKO
    """
    if method in ["iterate-exact", "iterate-expanded"]:
        return eko_iterate(gamma_valence, a1, a0, aem_list, nf, order, ev_op_iterations)
    raise NotImplementedError("Selected method is not implemented")
