# -*- coding: utf-8 -*-
r"""
This module contains the QCD beta function coefficients.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import numba as nb

from eko import constants


@nb.njit("f8(u1)", cache=True)
def beta_0(nf: int):
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
def beta_1(nf: int):
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
def beta_2(nf: int):
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
    else:
        raise ValueError("Beta coefficients beyond NNLO are not implemented!")
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
