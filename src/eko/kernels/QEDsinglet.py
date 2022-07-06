# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import utils


@nb.njit(cache=True)
def eko_iterate(gamma_singlet, a1, a0, nf, order, ev_op_iterations):
    """
    Singlet NLO or NNLO iterated (exact) EKO

    Parameters
    ----------
        gamma_singlet : numpy.ndarray
            singlet anomalous dimensions matrices
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        order : int
            perturbative order
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_s^{order} : numpy.ndarray
            singlet NLO or NNLO iterated (exact) EKO
    """
    a_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
    beta0 = beta.beta_qcd((2, 0), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    if order[0] >= 3:
        beta2 = beta.beta_qcd((4, 0), nf)
    e = np.identity(4, np.complex_)
    al = a_steps[0]
    for ah in a_steps[1:]:
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        if order[0] == 2:
            ln = (
                (gamma_singlet[0] * a_half + gamma_singlet[1] * a_half**2)
                / (beta0 * a_half**2 + beta1 * a_half**3)
                * delta_a
            )
        elif order[0] == 3:
            ln = (
                (
                    gamma_singlet[0] * a_half
                    + gamma_singlet[1] * a_half**2
                    + gamma_singlet[2] * a_half**3
                )
                / (beta0 * a_half**2 + beta1 * a_half**3 + beta2 * a_half**4)
                * delta_a
            )
        ek = np.ascontiguousarray(ad.exp_singlet(ln)[0])
        e = ek @ e
        al = ah
    return e
