r"""Compute evolution integrals.

Integrals needed for the exact evolutions are given by:

.. math::
    j^{(n,m)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{(a_s')^{1+n}}{-\beta^{(m)}(a_s')}

The expanded integrals are obtained from the exact results by Taylor expanding in the limit
:math:`a_s,a_s^{0} \to 0` until :math:`\mathcal{O}( a_s^{m+1})` for :math:`N^{m}LO` computations.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def jm10(a1, a0, beta0):
    r"""LO-LO QED exact evolution integral.

    .. math::
        j^{(-1,0)}(a_s,a_s^0,aem) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{(\beta_0 + aem \beta_{0,1}) a_s'^2}
           = \frac{1.0 / a0 - 1.0 / as}{\beta_0 + aem \beta_{0,1}}

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        LO beta function

    Returns
    -------
    jm10 : float
        integral
    """
    return (1.0 / a0 - 1.0 / a1) / beta0


@nb.njit(cache=True)
def j00(a1, a0, beta0):
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
        beta0 : float
            LO beta function

    Returns
    -------
        j00 : float
            integral
    """
    return np.log(a1 / a0) / beta0


@nb.njit(cache=True)
def j11_exact(a1, a0, beta0, beta1):
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
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function

    Returns
    -------
        j11 : float
            integral
    """
    b1 = beta1 / beta0
    return (1.0 / beta1) * np.log((1.0 + a1 * b1) / (1.0 + a0 * b1))


@nb.njit(cache=True)
def j11_expanded(a1, a0, beta0):
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
        beta0 : float
            LO beta function

    Returns
    -------
        j11_exp : float
            integral
    """
    return 1.0 / beta0 * (a1 - a0)


@nb.njit(cache=True)
def j01_exact(a1, a0, beta0, beta1):
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
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function

    Returns
    -------
        j01 : float
            integral
    """
    b1 = beta1 / beta0
    return j00(a1, a0, beta0) - b1 * j11_exact(a1, a0, beta0, beta1)


@nb.njit(cache=True)
def j01_expanded(a1, a0, beta0, beta1):
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
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function

    Returns
    -------
        j01_exp : float
            integral
    """
    b1 = beta1 / beta0
    return j00(a1, a0, beta0) - b1 * j11_expanded(a1, a0, beta0)


@nb.njit(cache=True)
def jm11_exact(a1, a0, beta0, beta1):
    r"""LO-NLO exact evolution integral.

    .. math::
        j^{(-1,1)}(a_s,a_s^0,aem) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3}
            = \frac{1.0 / a0 - 1.0 / as}{\beta_0 + aem \beta_{0,1}} + \frac{b_1}{(\beta_0 + aem \beta_{0,1}}  \left(\log(1 + 1 / (as b_1)) - \log(1 + 1 / (a0 b_1)\right)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        LO beta function
    beta1 : float
        NLO beta function

    Returns
    -------
    jm11 : float
        integral
    """
    b1 = beta1 / beta0
    return -(1.0 / a1 - 1.0 / a0) / beta0 + b1 / beta0 * (
        np.log(1.0 + 1.0 / (a1 * b1)) - np.log(1.0 + 1.0 / (a0 * b1))
    )


@nb.njit(cache=True)
def j22_exact(a1, a0, beta0, beta1, beta2):
    r"""
    NNLO-NNLO exact evolution integral.

    .. math::
        j^{(2,2)}(a_s,a_s^0) &=
            \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^3}
                        {\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}
            = \frac{1}{\beta_2}\ln\left(
                    \frac{1 + a_s ( b_1 + b_2 a_s ) }{ 1 + a_s^0 ( b_1 + b_2 a_s^0 )}\right)
                - \frac{b_1 \delta}{ \beta_2 \Delta} \\
            \delta &= \atan \left( \frac{b_1 + 2 a_s b_2 }{ \Delta} \right)
                    - \atan \left( \frac{b_1 + 2 a_s^0 b_2 }{ \Delta} \right) \\
            \Delta &= \sqrt{4 b_2 - b_1^2}

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function
        beta2 : float
            NNLO beta function

    Returns
    -------
        j22 : complex
            integral
    """
    b1 = beta1 / beta0
    b2 = beta2 / beta0
    # allow Delta to be complex for nf = 6, the final result will be real
    Delta = np.sqrt(complex(4 * b2 - b1**2))
    delta = np.arctan((b1 + 2 * a1 * b2) / Delta) - np.arctan(
        (b1 + 2 * a0 * b2) / Delta
    )
    log = np.log((1 + a1 * (b1 + b2 * a1)) / (1 + a0 * (b1 + b2 * a0)))
    return 1 / (2 * beta2) * log - b1 / (beta2) * np.real(delta / Delta)


@nb.njit(cache=True)
def j12_exact(a1, a0, beta0, beta1, beta2):
    r"""
    NLO-NNLO exact evolution integral.

    .. math::
        j^{(1,2)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^2}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
               &= \frac{2 \delta}{\beta_0 \Delta}  \\
        \delta &= \atan \left( \frac{b_1 + 2 a_s b_2 }{ \Delta} \right) - \atan \left( \frac{b_1 + 2 a_s^0 b_2 }{ \Delta} \right) \\
        \Delta &= \sqrt{4 b_2 - b_1^2}

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function
        beta2 : float
            NNLO beta function

    Returns
    -------
        j12 : complex
            integral
    """  # pylint: disable=line-too-long
    b1 = beta1 / beta0
    b2 = beta2 / beta0
    # allow Delta to be complex for nf = 6, the final result will be real
    Delta = np.sqrt(complex(4 * b2 - b1**2))
    delta = np.arctan((b1 + 2 * a1 * b2) / Delta) - np.arctan(
        (b1 + 2 * a0 * b2) / Delta
    )
    return 2.0 / (beta0) * np.real(delta / Delta)


@nb.njit(cache=True)
def j02_exact(a1, a0, beta0, beta1, beta2):
    r"""
    LO-NNLO exact evolution integral.

    .. math::
        j^{(0,2)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,
              \frac{a_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
            &= j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,2)}(a_s,a_s^0) - b_2 j^{(2,2)}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function
        beta2 : float
            NNLO beta function

    Returns
    -------
        j02 : complex
            integral
    """
    b1 = beta1 / beta0
    b2 = beta2 / beta0
    return (
        j00(a1, a0, beta0)
        - b1 * j12_exact(a1, a0, beta0, beta1, beta2)
        - b2 * j22_exact(a1, a0, beta0, beta1, beta2)
    )


@nb.njit(cache=True)
def j22_expanded(a1, a0, beta0):
    r"""
    NNLO-NNLO expanded evolution integral.

    .. math::
        j^{(2,2)}_{exp}(a_s,a_s^0) = \frac{1}{2 \beta_0} \left( a_s^2 -  (a_s^0)^{2} \right)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function

    Returns
    -------
        j22_exp : float
            integral
    """
    return 1 / (2 * beta0) * (a1**2 - a0**2)


@nb.njit(cache=True)
def j12_expanded(a1, a0, beta0, beta1):
    r"""
    NLO-NNLO expanded evolution integral.

    .. math::
        j^{(1,2)}_{exp}(a_s,a_s^0) = \frac{1}{\beta_0}\left[ a_s - a_s^0 -
                      \frac{b_1}{2} \left( a_s^2 - (a_s^0)^{2} \right)\right]

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function

    Returns
    -------
        j12_exp : float
            integral
    """
    b1 = beta1 / beta0
    return 1 / beta0 * (a1 - a0 - b1 / 2 * (a1**2 - a0**2))


@nb.njit(cache=True)
def j02_expanded(a1, a0, beta0, beta1, beta2):
    r"""
    LO-NNLO expanded evolution integral.

    .. math::
        j^{(0,2)}_{exp}(a_s,a_s^0) = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,2)}_{exp}(a_s,a_s^0)
                                      - b_2 j^{(2,2)}_{exp}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function
        beta1 : float
            NLO beta function
        beta2 : float
            NNLO beta function

    Returns
    -------
        j02_exp : float
            integral
    """
    b1 = beta1 / beta0
    b2 = beta2 / beta0
    return (
        j00(a1, a0, beta0)
        - b1 * j12_expanded(a1, a0, beta0, beta1)
        - b2 * j22_expanded(a1, a0, beta0)
    )


@nb.njit(cache=True)
def jm12_exact(a1, a0, beta0, beta1, beta2):
    r"""LO-NNLO exact evolution integral.

    .. math::
        j^{(-1,2)}(a_s,a_s^0,aem) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,
            \frac{1}{(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
            &= j^{(-1,0)}(a_s,a_s^0,aem) - b_1 j^{(0,2)}(a_s,a_s^0) - b_2 j^{(1,2)}(a_s,a_s^0)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
            LO beta function
    beta1 : float
        NLO beta function
    beta2 : float
        NNLO beta function

    Returns
    -------
    jm12 : complex
        integral
    """
    b1 = beta1 / beta0
    b2 = beta2 / beta0
    return (
        jm10(a1, a0, beta0)
        - b1 * j02_exact(a1, a0, beta0, beta1, beta2)
        - b2 * j12_exact(a1, a0, beta0, beta1, beta2)
    )
