r"""Compute evolution integrals needed for QED."""
import numba as nb

from .. import beta
from . import evolution_integrals as ei


@nb.njit(cache=True)
def j12(a1, a0, aem, nf):
    r"""LO-LO QED exact evolution integral.

    .. math::
        j^{(0,0)}(a_s,a_s^0,aem) = \int\limits_{a_s^0}^{a_s} \frac{da_s'}{(\beta_0 + aem \beta_{0,1}) a_s'}
           = \frac{\ln(a_s/a_s^0)}{\beta_0 + aem \beta_{0,1}}

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j12 : float
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    return ei.j12(a1, a0, beta0)


@nb.njit(cache=True)
def j02(a1, a0, aem, nf):
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
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j12 : float
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    return ei.j02(a1, a0, beta0)


@nb.njit(cache=True)
def j23_exact(a1, a0, aem, nf):
    r"""NLO-NLO exact evolution integral.

    .. math::
        j^{(1,1)}(a_s,a_s^0) = \int\limits_{a_s^0}^{a_s}\!da_s'\,
                                \frac{a_s'^2}{(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3}
            = \frac{1}{\beta_1}\ln\left(\frac{1+b_1 a_s}{1+b_1 a_s^0}\right)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j11 : float
        integral
    """
    beta1 = beta.beta_qcd((3, 0), nf)
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    return ei.j23_exact(a1, a0, beta0, beta1)


@nb.njit(cache=True)
def j13_exact(a1, a0, aem, nf):
    r"""LO-NLO QED exact evolution integral.

    .. math::
        j^{(0,1)}(a_s,a_s^0,aem) = \int\limits_{a_s^0}^{a_s}\!da_s'\,
                            \frac{a_s'}{(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3}
               = j^{(0,0)}(a_s,a_s^0,aem) - b_1 j^{(1,1)}(a_s,a_s^0,aem)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j11 : float
        integral
    """
    beta1 = beta.beta_qcd((3, 0), nf)
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    return ei.j13_exact(a1, a0, beta0, beta1)


@nb.njit(cache=True)
def j03_exact(a1, a0, aem, nf):
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
    nf : int
        number of active flavors

    Returns
    -------
    j11 : float
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    return ei.j03_exact(a1, a0, beta0, beta1)


@nb.njit(cache=True)
def j34_exact(a1, a0, aem, nf):
    r"""NNLO-NNLO exact evolution integral.

    .. math::
        j^{(2,2)}(a_s,a_s^0,aem) &=
            \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^3}
                        {(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}
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
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j22 : complex
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    beta2 = beta.beta_qcd((4, 0), nf)
    return ei.j34_exact(a1, a0, beta0, beta1, beta2)


@nb.njit(cache=True)
def j24_exact(a1, a0, aem, nf):
    r"""NLO-NNLO exact evolution integral.

    .. math::
        j^{(1,2)}(a_s,a_s^0,aem) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,\frac{a_s'^2}{(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
               &= \frac{2 \delta}{\beta_0 \Delta}  \\
        \delta &= \atan \left( \frac{b_1 + 2 a_s b_2 }{ \Delta} \right) - \atan \left( \frac{b_1 + 2 a_s^0 b_2 }{ \Delta} \right) \\
        \Delta &= \sqrt{4 b_2 - b_1^2}

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j12 : complex
        integral
    """  # pylint: disable=line-too-long
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    beta2 = beta.beta_qcd((4, 0), nf)
    return ei.j24_exact(a1, a0, beta0, beta1, beta2)


@nb.njit(cache=True)
def j14_exact(a1, a0, aem, nf):
    r"""LO-NNLO exact evolution integral.

    .. math::
        j^{(0,2)}(a_s,a_s^0,aem) &= \int\limits_{a_s^0}^{a_s}\!da_s'\,
              \frac{a_s'}{(\beta_0 + aem \beta_{0,1}) a_s'^2 + \beta_1 a_s'^3 + \beta_2 a_s'^4}\\
            &= j^{(0,0)}(a_s,a_s^0) - b_1 j^{(1,2)}(a_s,a_s^0) - b_2 j^{(2,2)}(a_s,a_s^0)

    Parameters
    ----------
    a1 : float
        target coupling value
    a0 : float
        initial coupling value
    aem : float
        electromagnetic coupling value
    nf : int
        number of active flavors

    Returns
    -------
    j02 : complex
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    beta2 = beta.beta_qcd((4, 0), nf)
    return ei.j14_exact(a1, a0, beta0, beta1, beta2)


@nb.njit(cache=True)
def j04_exact(a1, a0, aem, nf):
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
    nf : int
        number of active flavors

    Returns
    -------
    j02 : complex
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    beta1 = beta.beta_qcd((3, 0), nf)
    beta2 = beta.beta_qcd((4, 0), nf)
    return ei.j04_exact(a1, a0, beta0, beta1, beta2)
