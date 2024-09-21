"""Implement the |N3LO| evolution integrals."""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def roots(b_list):
    """Return the roots of a third grade polynomial.

    Return the roots of:

    .. math ::
        1 + b_1 a_s + b_2 a_s^2 + b_3 a_s^3 = 0

    Parameters
    ----------
    b_list : list
        :math:`[b_1, b_2, b_3]`
    Returns
    -------
    list
        list of complex roots
    """
    b1, b2, b3 = b_list
    d1 = -(b2**2) + 3 * b1 * b3
    d2 = -2 * b2**3 + 9 * b1 * b2 * b3 - 27 * b3**2
    d3 = ((d2 + np.sqrt(4 * d1**3 + d2**2)) / 2) ** (1.0 / 3.0)
    r1 = 1 / (3 * b3) * (-b2 - d1 / d3 + d3)
    r2 = (1 / (3 * b3)) * (
        -b2
        + ((1 + 1.0j * np.sqrt(3)) * d1) / (2 * d3)
        - ((1 - 1.0j * np.sqrt(3)) * d3) / 2
    )
    r3 = (1 / (3 * b3)) * (
        -b2
        + ((1 - 1.0j * np.sqrt(3)) * d1) / (2 * d3)
        - ((1 + 1.0j * np.sqrt(3)) * d3) / 2
    )
    return [r1, r2, r3]


@nb.njit(cache=True)
def derivative(r, b_list):
    r"""Return the derivative of a third grade polynomial.

    .. math ::
        \frac{d}{d a_s}(1 + b_1 a_s + b_2 a_s^2 + b_3 a_s^3) = b_1 + 2 b_2 r + 3 b_3 r^2

    Parameters
    ----------
    a_s :
        coupling constant
    b_list : list
        :math:`[b_1, b_2, b_3]`

    Returns
    -------
    float :
        evaluated derivative
    """
    b1, b2, b3 = b_list
    return b1 + 2 * b2 * r + 3 * b3 * r**2


@nb.njit(cache=True)
def j33_exact(a1, a0, beta0, b_list, roots):
    r"""|N3LO|-|N3LO| exact evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::

        j^{(3,3)}(a_s) =  \frac{1}{\beta_0} \sum_{r=r_1}^{r_3} \frac{\ln(a_s-r) r^2}{b_1 + 2 b_2 r + 3 b_3 r^2}

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        :math:`\beta_0`
    b_list : list
        :math:`[b_1, b_2, b_3]`
    roots: :math:`[r_1,r_2,r_3]`
        list of roots of :math:`1 + b_1 a_s + b_2 a_s^2 + b_3 a_s^3`

    Returns
    -------
    float
        evaluated integral
    """
    integral = 0
    for r in roots:
        integral += r**2 / derivative(r, b_list) * np.log((a1 - r) / (a0 - r))
    return (1 / beta0) * integral


@nb.njit(cache=True)
def j23_exact(a1, a0, beta0, b_list, roots):
    r"""|NNLO|-|N3LO| exact evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::

        j^{(2,3)}(a_s) =  \frac{1}{\beta_0} \sum_{r=r_1}^{r_3} \frac{\ln(a_s-r) r}{b_1 + 2 b_2 r + 3 b_3 r^2}

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        :math:`\beta_0`
    b_list : list
        :math:`[b_1, b_2, b_3]`
    roots: :math:`[r_1,r_2,r_3]`
        list of roots of :math:`1 + b_1 a_s + b_2 a_s^2 + b_3 a_s^3`

    Returns
    -------
    float
        evaluated integral
    """
    integral = 0
    for r in roots:
        integral += r / derivative(r, b_list) * np.log((a1 - r) / (a0 - r))
    return (1 / beta0) * integral


@nb.njit(cache=True)
def j13_exact(a1, a0, beta0, b_list, roots):
    r"""|NLO|-|N3LO| exact evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::

        j^{(1,3)}(a_s) = \frac{1}{\beta_0} \sum_{r=r_1}^{r_3} \frac{\ln(a_s-r)}{b_1 + 2 b_2 r + 3 b_3 r^2}

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        :math:`\beta_0`
    b_list : list
        :math:`[b_1, b_2, b_3]`
    roots: :math:`[r_1,r_2,r_3]`
        list of roots of :math:`1 + b_1 a_s + b_2 a_s^2 + b_3 a_s^3`

    Returns
    -------
    float
        evaluated integral
    """
    integral = 0
    for r in roots:
        integral += 1.0 / derivative(r, b_list) * np.log((a1 - r) / (a0 - r))
    return (1 / beta0) * integral


@nb.njit(cache=True)
def j03_exact(j12, j13, j23, j33, b_list):
    r"""|LO|-|N3LO| exact evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::

        j^{(0,3)}(a_s) = \frac{1}{\beta_0} \left( \ln(a) - \sum_{r=r_1}^{r_3} \frac{b_1 \ln(a_s-r) + b_2 \ln(a_s-r) r + b_3 \ln(a_s-r) r^2}{b_1 + 2 b_2 r + 3 b_3 r^2} \right )

    Parameters
    ----------
    j12: float
        |LO|-|LO| evolution integral
    j13: float
        |NLO|-|N3LO| evolution integral
    j23: float
        |NNLO|-|N3LO| evolution integral
    j33: float
        |N3LO|-|N3LO| evolution integral
    b_list : list
        :math:`[b_1, b_2, b_3]`

    Returns
    -------
    float
        evaluated integral
    """
    b1, b2, b3 = b_list
    return j12 - b1 * j13 - b2 * j23 - b3 * j33


@nb.njit(cache=True)
def j33_expanded(a1, a0, beta0):
    r"""|N3LO|-|N3LO| expanded evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::
        j^{(3,3)}_{exp}(a_s) = \frac{1}{3 \beta_0} a_s^3

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        :math:`\beta_0`

    Returns
    -------
    float
        evaluated integral
    """
    return 1 / (3 * beta0) * (a1**3 - a0**3)


@nb.njit(cache=True)
def j23_expanded(a1, a0, beta0, b_list):
    r"""|NNLO|-|N3LO| expanded evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::
        j^{(2,3)}_{exp}(a_s) = \frac{1}{\beta_0} ( \frac{1}{2} a_s^2 - \frac{b_1}{3} as^3)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        :math:`\beta_0`
    b_list : list
        :math:`[b_1, b_2, b_3]`

    Returns
    -------
    float
        evaluated integral
    """
    b1 = b_list[0]
    return 1 / beta0 * (1 / 2 * (a1**2 - a0**2) - b1 / 3 * (a1**3 - a0**3))


@nb.njit(cache=True)
def j13_expanded(a1, a0, beta0, b_list):
    r"""|NLO|-|N3LO| expanded evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::
        j^{(1,3)}_{exp}(a_s) = \frac{1}{\beta_0} ( a_s - \frac{b_1}{2} a_s^2 + \frac{b_1^2-b_2}{3} as^3)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    beta0 : float
        :math:`\beta_0`
    b_list : list
        :math:`[b_1, b_2, b_3]`

    Returns
    -------
    float
        evaluated integral
    """
    b1, b2, _ = b_list
    return (1 / beta0) * (
        (a1 - a0) - b1 / 2 * (a1**2 - a0**2) + (b1**2 - b2) / 3 * (a1**3 - a0**3)
    )


@nb.njit(cache=True)
def j03_expanded(j12, j13, j23, j33, b_list):
    r"""|LO|-|N3LO| expanded evolution definite integral.

    Evaluated at :math:`a_s-a_s^0`.

    .. math::
        j^{(0,3)}_{exp}(a_s) = j^{(0,0)} - b_1 j^{(1,3)}_{exp}(a_s) - b_2 j^{(2,3)}_{exp}(a_s) - b_3 j^{(3,3)}_{exp}(a_s)

    Parameters
    ----------
    j12: float
        |LO|-|LO| evolution integral
    j13: float
        |NLO|-|N3LO| expanded evolution integral
    j23: float
        |NNLO|-|N3LO| expanded evolution integral
    j33: float
        |N3LO|-|N3LO| expanded evolution integral
    b_list : list
        :math:`[b_1, b_2, b_3]`

    Returns
    -------
    float
        evaluated integral

    See Also
    --------
    j03_exact
    """
    return j03_exact(j12, j13, j23, j33, b_list)
