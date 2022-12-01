r"""Compute evolution integrals needed for QED."""
import numba as nb
import numpy as np

from .. import beta
from . import evolution_integrals as ei


@nb.njit(cache=True)
def j00_qed(a1, a0, aem, nf):
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
    j00 : float
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    return np.log(a1 / a0) / beta0


@nb.njit(cache=True)
def jm10(a1, a0, aem, nf):
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
    j00 : float
        integral
    """
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    return (1.0 / a0 - 1.0 / a1) / beta0


@nb.njit(cache=True)
def j11_exact_qed(a1, a0, aem, nf):
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
    b1 = beta1 / beta0
    return (1.0 / beta1) * np.log((1.0 + a1 * b1) / (1.0 + a0 * b1))


@nb.njit(cache=True)
def j01_exact_qed(a1, a0, aem, nf):
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
    b1 = beta1 / beta0
    return j00_qed(a1, a0, aem, nf) - b1 * j11_exact_qed(a1, a0, aem, nf)


@nb.njit(cache=True)
def jm11_exact(a1, a0, aem, nf):
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
    b1 = beta.beta_qcd((3, 0), nf) / beta0
    return -(1.0 / a1 - 1.0 / a0) / beta0 + b1 / beta0 * (
        np.log(1.0 + 1.0 / (a1 * b1)) - np.log(1.0 + 1.0 / (a0 * b1))
    )


@nb.njit(cache=True)
def j22_exact_qed(a1, a0, aem, nf):
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
    b1 = beta.beta_qcd((3, 0), nf) / beta0
    b2 = beta.beta_qcd((4, 0), nf) / beta0
    # allow Delta to be complex for nf = 6, the final result will be real
    Delta = np.sqrt(complex(4 * b2 - b1**2))
    delta = np.arctan((b1 + 2 * a1 * b2) / Delta) - np.arctan(
        (b1 + 2 * a0 * b2) / Delta
    )
    log = np.log((1 + a1 * (b1 + b2 * a1)) / (1 + a0 * (b1 + b2 * a0)))
    return 1 / (2 * beta.beta_qcd((4, 0), nf)) * log - b1 / (
        beta.beta_qcd((4, 0), nf)
    ) * np.real(delta / Delta)


@nb.njit(cache=True)
def j12_exact_qed(a1, a0, aem, nf):
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
    b1 = beta.beta_qcd((3, 0), nf) / beta0
    b2 = beta.beta_qcd((4, 0), nf) / beta0
    # allow Delta to be complex for nf = 6, the final result will be real
    Delta = np.sqrt(complex(4 * b2 - b1**2))
    delta = np.arctan((b1 + 2 * a1 * b2) / Delta) - np.arctan(
        (b1 + 2 * a0 * b2) / Delta
    )
    return 2.0 / (beta0) * np.real(delta / Delta)


@nb.njit(cache=True)
def j02_exact_qed(a1, a0, aem, nf):
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
    b1 = beta.beta_qcd((3, 0), nf) / beta0
    b2 = beta.beta_qcd((4, 0), nf) / beta0
    return (
        j00_qed(a1, a0, aem, nf)
        - b1 * j12_exact_qed(a1, a0, aem, nf)
        - b2 * j22_exact_qed(a1, a0, aem, nf)
    )


@nb.njit(cache=True)
def jm12_exact(a1, a0, aem, nf):
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
    b1 = beta.beta_qcd((3, 0), nf) / beta0
    b2 = beta.beta_qcd((4, 0), nf) / beta0
    return (
        jm10(a1, a0, aem, nf)
        - b1 * j02_exact_qed(a1, a0, aem, nf)
        - b2 * j12_exact_qed(a1, a0, aem, nf)
    )
