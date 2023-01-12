r"""
This module contains the Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
The 3-loop references for the non-singlet :cite:`Moch:2004pa`
and singlet :cite:`Vogt:2004mw` case contain also the lower
order results. The results are also determined in Mellin space in
terms of the anomalous dimensions (note the additional sign!)

.. math::
    \gamma(N) = - \mathcal{M}[\mathbf{P}(x)](N)
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def exp_singlet(gamma_S):
    r"""
    Computes the exponential and the eigensystem of the singlet anomalous dimension matrix

    Parameters
    ----------
        gamma_S : numpy.ndarray
            singlet anomalous dimension matrix

    Returns
    -------
        exp : numpy.ndarray
            exponential of the singlet anomalous dimension matrix :math:`\gamma_{S}(N)`
        lambda_p : complex
            positive eigenvalue of the singlet anomalous dimension matrix
            :math:`\gamma_{S}(N)`
        lambda_m : complex
            negative eigenvalue of the singlet anomalous dimension matrix
            :math:`\gamma_{S}(N)`
        e_p : numpy.ndarray
            projector for the positive eigenvalue of the singlet anomalous
            dimension matrix :math:`\gamma_{S}(N)`
        e_m : numpy.ndarray
            projector for the negative eigenvalue of the singlet anomalous
            dimension matrix :math:`\gamma_{S}(N)`
    """
    # compute eigenvalues
    det = np.sqrt(
        np.power(gamma_S[0, 0] - gamma_S[1, 1], 2) + 4.0 * gamma_S[0, 1] * gamma_S[1, 0]
    )
    lambda_p = 1.0 / 2.0 * (gamma_S[0, 0] + gamma_S[1, 1] + det)
    lambda_m = 1.0 / 2.0 * (gamma_S[0, 0] + gamma_S[1, 1] - det)
    # compute projectors
    identity = np.identity(2, np.complex_)
    c = 1.0 / det
    e_p = +c * (gamma_S - lambda_m * identity)
    e_m = -c * (gamma_S - lambda_p * identity)
    exp = e_m * np.exp(lambda_m) + e_p * np.exp(lambda_p)
    return exp, lambda_p, lambda_m, e_p, e_m
