# -*- coding: utf-8 -*-
r"""
This module contains the scale variation for scheme A
"""
import numba as nb

from .. import beta


@nb.njit(["c16[:,:,:](c16[:,:,:],u1,u1,f8)", "c16[:](c16[:],u1,u1,f8)"], cache=True)
def gamma_fact(gamma, order, nf, L):
    """
    Adjust the anomalous dimensions with the scale variations.

    Parameters
    ----------
        gamma : numpy.ndarray
            anomalous dimensions
        order : int
            perturbation order
        nf : int
            number of active flavors
        L : float
            logarithmic ratio of factorization and renormalization scale

    Returns
    -------
        gamma : numpy.ndarray
            adjusted singlet anomalous dimensions
    """
    # since we are modifying *in-place* be carefull, that the order matters!
    # and indeed, we need to adjust the high elements first
    if order >= 2:
        gamma[2] -= (
            2 * beta.beta(0, nf) * gamma[1] * L
            + (beta.beta(1, nf) * L - beta.beta(0, nf) ** 2 * L ** 2) * gamma[0]
        )
    if order >= 1:
        gamma[1] -= beta.beta(0, nf) * gamma[0] * L
    return gamma
