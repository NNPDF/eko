@nb.njit(cache=True)
def j02(a1, a0, beta0):
    r""":math:`j^{(0,2)}` exact evolution integral.

    .. math::
        j^{(0,2)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{\beta_0 a_s'^2}
           = \frac{1.0 / a0 - 1.0 / as}{\beta_0}

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
    j02 : float
        integral
    """
    return (1.0 / a0 - 1.0 / a1) / beta0


@nb.njit(cache=True)
def j03_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(0,3)}` exact evolution integral.

    .. math::
        j^{(0,3)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{\beta_0 a_s'^2 + \beta_1 a_s'^3}
            = \frac{1.0 / a0 - 1.0 / as}{\beta_0 + \frac{b_1}{\beta_0}  \left(\log(1 + 1 / (as b_1)) - \log(1 + 1 / (a0 b_1)\right)

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
    j03_exact : float
        integral
    """
    b1 = b_vec[1]
    return -(1.0 / a1 - 1.0 / a0) / beta0 + b1 / beta0 * (
        np.log(1.0 + 1.0 / (a1 * b1)) - np.log(1.0 + 1.0 / (a0 * b1))
    )


@nb.njit(cache=True)
def j04_exact(a1, a0, beta0, b_vec):
    r""":math:`j^{(0,4)}` exact evolution integral.

    .. math::
        j^{(0,4)}(a_s,a_s^0) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,
            \frac{1}{\beta_0 a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
            &= j^{(-1,0)}(a_s,a_s^0) - b_1 j^{(1,4)}(a_s,a_s^0) - b_2 j^{(2,4)}(a_s,a_s^0)

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
    j04_exact : complex
        integral
    """
    b1 = b_vec[1]
    b2 = b_vec[2]
    return (
        j02(a1, a0, beta0)
        - b1 * j14_exact(a1, a0, beta0, b_vec)
        - b2 * j24_exact(a1, a0, beta0, b_vec)
    )
