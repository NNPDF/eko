# -*- coding: utf-8 -*-
r"""
Integrals needed for the exact evolutions.

.. math::
    j^{(n,m)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{(a_s')^{1+n}}{-\beta^{(m)}(a_s')}
"""

import numpy as np

import numba as nb

from .. import beta


@nb.njit("f8(f8,f8,u1)", cache=True)
def j00(a1, a0, nf):
    r"""
    LO-LO exact evolution integral.

    .. math::
        j^{(0,0)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{\beta_0 a_s'}
           = \frac{\ln(a_s/a_s^0)}{\beta_0}

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        j00 : float
            integral
    """
    return np.log(a1 / a0) / beta.beta(0, nf)


@nb.njit("f8(f8,f8,u1)", cache=True)
def j11_exact(a1, a0, nf):
    r"""
    NLO-NLO exact evolution integral.

    .. math::
        j^{(1,1)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,
                                \frac{a_s'^2}{\beta_0 a_s'^2 + \beta_1 a_s'^3}
            = \frac{1}{\beta_1}\ln\left(\frac{1+b_1 a_s}{1+b_1 a_s^0}\right)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        j11 : float
            integral
    """
    beta_1 = beta.beta(1, nf)
    b1 = beta.b(1, nf)
    return (1.0 / beta_1) * np.log((1.0 + a1 * b1) / (1.0 + a0 * b1))


@nb.njit("f8(f8,f8,u1)", cache=True)
def j11_expanded(a1, a0, nf):
    r"""
    NLO-NLO expanded evolution integral.

    .. math::
        j^{(1,1)}_{exp}(a_s,a_s^0) = \frac 1 {\beta_0}(a_s - a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        j11_exp : float
            integral
    """
    return 1.0 / beta.beta(0, nf) * (a1 - a0)


@nb.njit("f8(f8,f8,u1)", cache=True)
def j01_exact(a1, a0, nf):
    r"""
    LO-NLO exact evolution integral.

    .. math::
        j^{(0,1)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,
                            \frac{a_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3}
               = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,1)}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        j11 : float
            integral
    """
    return j00(a1, a0, nf) - beta.b(1, nf) * j11_exact(a1, a0, nf)


@nb.njit("f8(f8,f8,u1)", cache=True)
def j01_expanded(a1, a0, nf):
    r"""
    LO-NLO expanded evolution integral.

    .. math::
        j^{(0,1)}_{exp}(a_s,a_s^0) = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,1)}_{exp}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors

    Returns
    -------
        j01_exp : float
            integral
    """
    return j00(a1, a0, nf) - beta.b(1, nf) * j11_expanded(a1, a0, nf)
