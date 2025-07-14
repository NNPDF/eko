r"""The Altarelli-Parisi splitting kernels.

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
def exp_matrix_2D(gamma_S):
    r"""Compute the exponential and the eigensystem of the singlet anomalous
    dimension matrix.

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
    identity = np.identity(2, np.complex128)
    c = 1.0 / det
    e_p = +c * (gamma_S - lambda_m * identity)
    e_m = -c * (gamma_S - lambda_p * identity)
    exp = e_m * np.exp(lambda_m) + e_p * np.exp(lambda_p)
    return exp, lambda_p, lambda_m, e_p, e_m


@nb.njit(cache=True)
def exp_matrix(gamma):
    r"""Compute the exponential and the eigensystem of a matrix.

    Parameters
    ----------
    gamma : numpy.ndarray
        input matrix

    Returns
    -------
    exp : numpy.ndarray
        exponential of the matrix gamma :math:`\gamma(N)`
    w : numpy.ndarray
        array of the eigenvalues of the matrix lambda
    e : numpy.ndarray
        projectors on the eigenspaces of the matrix gamma :math:`\gamma(N)`
    """
    dim = gamma.shape[0]
    e = np.zeros((dim, dim, dim), np.complex128)
    # if dim == 2:
    #     exp, lambda_p, lambda_m, e_p, e_m = exp_matrix_2D(gamma)
    #     e[0] = e_p
    #     e[1] = e_m
    #     return exp, np.array([lambda_p, lambda_m]), e
    w, v = np.linalg.eig(gamma)
    v_inv = np.linalg.inv(v)
    exp = np.zeros((dim, dim), np.complex128)
    for i in range(dim):
        e[i] = np.outer(v[:, i], v_inv[i])
        exp += e[i] * np.exp(w[i])
    return exp, w, e
