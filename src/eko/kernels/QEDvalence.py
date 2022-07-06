# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

from .. import anomalous_dimensions as ad
from .. import beta
from . import utils


@nb.njit(cache=True)
def eko_iterate(gamma_valence, a1, a0, aem, nf, order, ev_op_iterations):
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
    e = np.identity(2, np.complex_)
    al = a_steps[0]
    betaQCD = np.array(
        [
            [
                beta.beta_qcd((2, 0), nf),
                beta.beta_qcd((3, 0), nf),
                beta.beta_qcd((4, 0), nf),
                beta.beta_qcd((5, 0), nf),
            ],
            [beta.beta_qcd((2, 1), nf), 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    for ah in a_steps[1:]:
        a_half = (ah + al) / 2.0
        delta_a = ah - al
        gamma = np.zeros((2, 2), np.complex_)
        betatot = 0
        for i in range(0, order[0] + 1):
            for j in range(0, order[1] + 1):
                betatot += a_half**2 * betaQCD[i, j] * a_half**i * aem**j
                if (i, j) == (0, 0):
                    continue  # this is probably useless
                gamma += gamma_valence[i, j] * a_half**i * aem**j
        ln = gamma / betatot * delta_a
        ek = np.ascontiguousarray(ad.exp_singlet(ln)[0])
        e = ek @ e
        al = ah
    return e
