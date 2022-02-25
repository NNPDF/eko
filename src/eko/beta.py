# -*- coding: utf-8 -*-
r"""
This module contains the QCD beta function coefficients.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import numba as nb

from . import constants
from .anomalous_dimensions.harmonics import zeta3


@nb.njit("f8(u1)", cache=True)
def beta_0(nf):
    """
    Computes the first coefficient of the QCD beta function.

    Implements Eq. (3.1) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_0 : float
            first coefficient of the QCD beta function :math:`\\beta_0^{n_f}`
    """
    beta_0 = 11.0 / 3.0 * constants.CA - 4.0 / 3.0 * constants.TR * nf
    return beta_0


@nb.njit("f8(u1)", cache=True)
def beta_1(nf):
    """
    Computes the second coefficient of the QCD beta function.

    Implements Eq. (3.2) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_1 : float
            second coefficient of the QCD beta function :math:`\\beta_1^{n_f}`
    """
    TF = constants.TR * nf
    b_ca2 = 34.0 / 3.0 * constants.CA * constants.CA
    b_ca = -20.0 / 3.0 * constants.CA * TF
    b_cf = -4.0 * constants.CF * TF
    beta_1 = b_ca2 + b_ca + b_cf
    return beta_1


@nb.njit("f8(u1)", cache=True)
def beta_2(nf):
    """
    Computes the third coefficient of the QCD beta function

    Implements Eq. (3.3) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_2 : float
            third coefficient of the QCD beta function :math:`\\beta_2^{n_f}`
    """
    TF = constants.TR * nf
    beta_2 = (
        2857.0 / 54.0 * constants.CA * constants.CA * constants.CA
        - 1415.0 / 27.0 * constants.CA * constants.CA * TF
        - 205.0 / 9.0 * constants.CF * constants.CA * TF
        + 2.0 * constants.CF * constants.CF * TF
        + 44.0 / 9.0 * constants.CF * TF * TF
        + 158.0 / 27.0 * constants.CA * TF * TF
    )
    return beta_2


@nb.njit("f8(u1)", cache=True)
def beta_3(nf):
    """
    Computes the fourth coefficient of the QCD beta function

    Implements Eq. (3.6) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_3 : float
            fourth coefficient of the QCD beta function :math:`\\beta_3^{n_f}`
    """
    beta_3 = (
        149753.0 / 6.0
        + 3564.0 * zeta3
        + nf * (-1078361.0 / 162.0 - 6508.0 / 27.0 * zeta3)
        + nf**2 * (50065.0 / 162.0 + 6472.0 / 81.0 * zeta3)
        + 1093.0 / 729.0 * nf**3
    )
    return beta_3


@nb.njit("f8(u1,u1)", cache=True)
def beta(k, nf):
    """
    Compute value of a beta coefficients

    Parameters
    ----------
        k : int
            perturbative order
        nf : int
            number of active flavors

    Returns
    -------
        beta : float
            beta_k(nf)
    """
    beta_ = 0
    if k == 0:
        beta_ = beta_0(nf)
    elif k == 1:
        beta_ = beta_1(nf)
    elif k == 2:
        beta_ = beta_2(nf)
    elif k == 3:
        beta_ = beta_3(nf)
    else:
        raise ValueError("Beta coefficients beyond N3LO are not implemented!")
    return beta_


@nb.njit("f8(u1,u1)", cache=True)
def b(k, nf):
    """
    Compute b coefficient.

    Parameters
    ----------
        k : int
            perturbative order
        nf : int
            number of active flavors

    Returns
    -------
        b : float
            b_k(nf)
    """
    return beta(k, nf) / beta(0, nf)
