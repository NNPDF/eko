r"""Compute evolution integrals.

Integrals needed for the exact evolutions are given by:

.. math::
    j^{(n,m)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{(a_s')^{n}}{-\sum_{i=2}^{m} \beta^{(i)} a_s'^i}

The expanded integrals are obtained from the exact results by Taylor expanding in the limit
:math:`a_s,a_s^{0} \to 0` until :math:`\mathcal{O}( a_s^{m+1})` for :math:`N^{m}LO` computations.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def j12(a1, a0, beta0):
    r""":math:`j^{(1,2)}` exact evolution integral.

    .. math::
        j^{(1,2)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{\beta_0 a_s'}
           = \frac{\ln(a_s/a_s^0)}{\beta_0}

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient

    Returns
    -------
        j12 : float
            integral
    """
    return np.log(a1 / a0) / beta0


@nb.njit(cache=True)
def j23_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(2,3)}` exact evolution integral.

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
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient

    Returns
    -------
        j23_exact : float
            integral
    """
    b1 = b_vec[1]
    return (1.0 / (b1 * beta0)) * np.log((1.0 + a1 * b1) / (1.0 + a0 * b1))


@nb.njit(cache=True)
def j23_expanded(a1, a0, beta0):
    r""":math:`j^{(2,3)}` expanded evolution integral.

    .. math::
        j^{(2,3)}_{exp}(a_s,a_s^0) = \frac 1 {\beta_0}(a_s - a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient

    Returns
    -------
        j23_expanded : float
            integral
    """
    return 1.0 / beta0 * (a1 - a0)


@nb.njit(cache=True)
def j13_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(1,3)}` exact evolution integral.

    .. math::
        j^{(1,3)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,
                            \frac{a_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3}
               = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(2,3)}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient

    Returns
    -------
        j13_exact : float
            integral
    """
    b1 = b_vec[1]
    return j12(a1, a0, beta0) - b1 * j23_exact(a1, a0, beta0, b_vec)


@nb.njit(cache=True)
def j13_expanded(a1, a0, beta0, b_vec):
    r""":math:`j^{(1,3)}` expanded evolution integral.

    .. math::
        j^{(1,3)}_{exp}(a_s,a_s^0) = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(2,3)}_{exp}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient

    Returns
    -------
        j13_expanded : float
            integral
    """
    b1 = b_vec[1]
    return j12(a1, a0, beta0) - b1 * j23_expanded(a1, a0, beta0)


@nb.njit(cache=True)
def j34_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(3,4)}` exact evolution integral.

    .. math::
        j^{(3,4)}(a_s,a_s^0) &=
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
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient
        beta2 : float
            NNLO beta function coefficient

    Returns
    -------
        j34_exact : complex
            integral
    """
    b1 = b_vec[1]
    b2 = b_vec[2]
    beta2 = b2 * beta0
    # allow Delta to be complex for nf = 6, the final result will be real
    Delta = np.sqrt(complex(4 * b2 - b1**2))
    delta = np.arctan((b1 + 2 * a1 * b2) / Delta) - np.arctan(
        (b1 + 2 * a0 * b2) / Delta
    )
    log = np.log((1 + a1 * (b1 + b2 * a1)) / (1 + a0 * (b1 + b2 * a0)))
    return 1 / (2 * beta2) * log - b1 / (beta2) * np.real(delta / Delta)


@nb.njit(cache=True)
def j24_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(2,4)}` exact evolution integral.

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
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient
        beta2 : float
            NNLO beta function coefficient

    Returns
    -------
        j24_exact : complex
            integral
    """  # pylint: disable=line-too-long
    b1 = b_vec[1]
    b2 = b_vec[2]
    # allow Delta to be complex for nf = 6, the final result will be real
    Delta = np.sqrt(complex(4 * b2 - b1**2))
    delta = np.arctan((b1 + 2 * a1 * b2) / Delta) - np.arctan(
        (b1 + 2 * a0 * b2) / Delta
    )
    return 2.0 / (beta0) * np.real(delta / Delta)


@nb.njit(cache=True)
def j14_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(1,4)}` exact evolution integral.

    .. math::
        j^{(0,2)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,
              \frac{a_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
            &= j^{(0,0)}(a_s,a_s^0) - b_1 j^{(2,4)}(a_s,a_s^0) - b_2 j^{(3,4)}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient
        beta2 : float
            NNLO beta function coefficient

    Returns
    -------
        j14_exact : complex
            integral
    """
    b1 = b_vec[1]
    b2 = b_vec[2]
    return (
        j12(a1, a0, beta0)
        - b1 * j24_exact(a1, a0, beta0, b_vec)
        - b2 * j34_exact(a1, a0, beta0, b_vec)
    )


@nb.njit(cache=True)
def j34_expanded(a1, a0, beta0):
    r""":math:`j^{(3,4)}` expanded evolution integral.

    .. math::
        j^{(2,2)}_{exp}(a_s,a_s^0) = \frac{1}{2 \beta_0} \left( a_s^2 -  (a_s^0)^{2} \right)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient

    Returns
    -------
        j34_expanded : float
            integral
    """
    return 1 / (2 * beta0) * (a1**2 - a0**2)


@nb.njit(cache=True)
def j24_expanded(a1, a0, beta0, b_vec):
    r""":math:`j^{(2,4)}` expanded evolution integral.

    .. math::
        j^{(2,4)}_{exp}(a_s,a_s^0) = \frac{1}{\beta_0}\left[ a_s - a_s^0 -
                      \frac{b_1}{2} \left( a_s^2 - (a_s^0)^{2} \right)\right]

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient

    Returns
    -------
        j24_expanded : float
            integral
    """
    b1 = b_vec[1]
    return 1 / beta0 * (a1 - a0 - b1 / 2 * (a1**2 - a0**2))


@nb.njit(cache=True)
def j14_expanded(a1, a0, beta0, b_vec):
    r""":math:`j^{(1,4)}` expanded evolution integral.

    .. math::
        j^{(1,4)}_{exp}(a_s,a_s^0) = j^{(0,0)}(a_s,a_s^0) - b_1 j^{(2,4)}_{exp}(a_s,a_s^0)
                                      - b_2 j^{(3,4)}_{exp}(a_s,a_s^0)

    Parameters
    ----------
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        beta0 : float
            LO beta function coefficient
        beta1 : float
            NLO beta function coefficient
        beta2 : float
            NNLO beta function coefficient

    Returns
    -------
        j14_expanded : float
            integral
    """
    b1 = b_vec[1]
    b2 = b_vec[2]
    return (
        j12(a1, a0, beta0)
        - b1 * j24_expanded(a1, a0, beta0, b_vec)
        - b2 * j34_expanded(a1, a0, beta0)
    )
