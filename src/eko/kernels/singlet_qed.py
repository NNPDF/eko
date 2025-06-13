"""Collection of QED singlet EKOs."""

import numba as nb
import numpy as np

from ekore import anomalous_dimensions as ad

from .. import beta
from . import EvoMethods


@nb.njit(cache=True)
def eko_iterate(gamma_singlet, as_list, a_half, nf, order, ev_op_iterations, dim):
    """Singlet QEDxQCD iterated (exact) EKO.

    Parameters
    ----------
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
    a1 : float
        target strong coupling value
    a0 : float
        initial strong coupling value
    aem_list : float
        electromagnetic coupling values
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
    e = np.identity(dim, np.complex128)
    betaQCD = np.zeros((order[0] + 1, order[1] + 1))
    for i in range(1, order[0] + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for step in range(1, ev_op_iterations + 1):
        ah = as_list[step]
        al = as_list[step - 1]
        as_half = a_half[step - 1, 0]
        aem = a_half[step - 1, 1]
        delta_a = ah - al
        gamma = np.zeros((dim, dim), np.complex128)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += betaQCD[i, j] * as_half ** (i + 1) * aem**j
                gamma += gamma_singlet[i, j] * as_half**i * aem**j
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_matrix(ln)[0])
        e = ek @ e
    return e


@nb.njit(cache=True)
def dispatcher(
    order,
    method,
    gamma_singlet,
    as_list,
    a_half,
    nf,
    ev_op_iterations,
    _ev_op_max_order,
):
    """Determine used kernel and call it.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative order
    method : int
        method
    gamma_singlet : numpy.ndarray
        singlet anomalous dimensions matrices
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
    ev_op_max_order : tuple(int,int)
        perturbative expansion order of U

    Returns
    -------
    e_s : numpy.ndarray
        singlet EKO
    """
    if method == EvoMethods.ITERATE_EXACT:
        return eko_iterate(
            gamma_singlet, as_list, a_half, nf, order, ev_op_iterations, 4
        )
    raise NotImplementedError('Only "iterate-exact" is implemented with QED')
