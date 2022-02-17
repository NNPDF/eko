# -*- coding: utf-8 -*-
r"""
This module contains the scale variation operator in scheme B
"""

import logging

import numba as nb
import numpy as np

from .. import beta

logger = logging.getLogger(__name__)


@nb.njit("c16(c16[:],f8,u1,u1,f8)", cache=True)
def non_singlet_variation(gamma, a_s, order, nf, L):
    """
    Scale Variation non singlet dispatcher

    Parameters
    ----------
        gamma : numpy.ndarray
            anomalous dimensions
        a_s :  float
            target coupling value
        order : int
            perturbation order
        nf : int
            number of active flavors
        L : float
            logarithmic ratio of factorization and renormalization scale

    Returns
    -------
        sv_ker : numpy.ndarray
            scale varion kernel
    """
    sv_ker = 1.0
    if order >= 1:
        sv_ker += a_s * L * gamma[0]
    if order >= 2:
        sv_ker += a_s**2 * (
            gamma[1] * L
            + 1 / 2 * (-beta.beta_0(nf) * gamma[0] + gamma[0] * gamma[0]) * L**2
        )
    return sv_ker


@nb.njit("c16[:,:](c16[:,:,:],f8,u1,u1,f8)", cache=True)
def singlet_variation(gamma, a_s, order, nf, L):
    """
    Scale Variation singlet dispatcher

    Parameters
    ----------
        gamma : numpy.ndarray
            anomalous dimensions
        a_s :  float
            target coupling value
        order : int
            perturbation order
        nf : int
            number of active flavors
        L : float
            logarithmic ratio of factorization and renormalization scale

    Returns
    -------
        sv_ker : numpy.ndarray
            scale varion kernel
    """
    sv_ker = np.eye(2, dtype=np.complex_)
    if order >= 1:
        sv_ker += a_s * L * gamma[0]
    if order >= 2:
        sv_ker += a_s**2 * (
            gamma[1] * L
            + 1
            / 2
            * (
                -beta.beta_0(nf) * gamma[0]
                + np.ascontiguousarray(gamma[0]) @ np.ascontiguousarray(gamma[0])
            )
            * L**2
        )
    return sv_ker
