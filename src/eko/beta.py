r"""
Compute the QCD beta function coefficients.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import numba as nb

from ekore.harmonics.constants import zeta3

from . import constants


@nb.njit(cache=True)
def beta_qcd_as2(nf):
    r"""Compute the first coefficient of the QCD beta function.

    Implements Eq. (3.1) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_qcd_as2 : float
            first coefficient of the QCD beta function :math:`\\beta_qcd_as2^{n_f}`

    """
    beta_qcd_as2 = 11.0 / 3.0 * constants.CA - 4.0 / 3.0 * constants.TR * nf
    return beta_qcd_as2


@nb.njit(cache=True)
def beta_qed_aem2(nf):
    r"""Compute the first coefficient of the QED beta function.

    Implements Eq. (7) of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_qed_aem2 : float
            first coefficient of the QED beta function :math:`\\beta_qed_aem2^{n_f}`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    nl = 3  # TODO : pass nl as an argument??
    beta_qed_aem2 = (
        -4.0 / 3 * (nl + constants.NC * (nu * constants.eu2 + nd * constants.ed2))
    )
    return beta_qed_aem2


@nb.njit(cache=True)
def beta_qcd_as3(nf):
    r"""Compute the second coefficient of the QCD beta function.

    Implements Eq. (3.2) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_qcd_as3 : float
            second coefficient of the QCD beta function :math:`\\beta_qcd_as3^{n_f}`

    """
    TF = constants.TR * nf
    b_ca2 = 34.0 / 3.0 * constants.CA * constants.CA
    b_ca = -20.0 / 3.0 * constants.CA * TF
    b_cf = -4.0 * constants.CF * TF
    beta_qcd_as3 = b_ca2 + b_ca + b_cf
    return beta_qcd_as3


@nb.njit(cache=True)
def beta_qed_aem3(nf):
    r"""Compute the second coefficient of the QED beta function.

    Implements Eq. (7) of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_qed_aem3 : float
            second coefficient of the QED beta function :math:`\\beta_qed_aem3^{n_f}`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    nl = 3  # TODO : pass nl as an argument??
    beta_qed_aem3 = -4.0 * (
        nl + constants.NC * (nu * constants.eu2**2 + nd * constants.ed2**2)
    )
    return beta_qed_aem3


@nb.njit(cache=True)
def beta_qcd_as2aem1(nf):
    r"""Compute the first QED correction of the QCD beta function.

    Implements Eq. (7) of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_as2aem1 : float
            first QED correction of the QCD beta function :math:`\\beta_as2aem1^{n_f}`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    beta_qcd_as2aem1 = -4.0 * constants.TR * (nu * constants.eu2 + nd * constants.ed2)
    return beta_qcd_as2aem1


@nb.njit(cache=True)
def beta_qed_aem2as1(nf):
    r"""Compute the first QCD correction of the QED beta function.

    Implements Eq. (7) of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_aem2as1 : float
            first QCD correction of the QED beta function :math:`\\beta_aem2as1^{n_f}`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    beta_qed_aem2as1 = (
        -4.0 * constants.CF * constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    )
    return beta_qed_aem2as1


@nb.njit(cache=True)
def beta_qcd_as4(nf):
    r"""Compute the third coefficient of the QCD beta function.

    Implements Eq. (3.3) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_qcd_as4 : float
            third coefficient of the QCD beta function :math:`\\beta_qcd_as4^{n_f}`

    """
    TF = constants.TR * nf
    beta_qcd_as4 = (
        2857.0 / 54.0 * constants.CA * constants.CA * constants.CA
        - 1415.0 / 27.0 * constants.CA * constants.CA * TF
        - 205.0 / 9.0 * constants.CF * constants.CA * TF
        + 2.0 * constants.CF * constants.CF * TF
        + 44.0 / 9.0 * constants.CF * TF * TF
        + 158.0 / 27.0 * constants.CA * TF * TF
    )
    return beta_qcd_as4


@nb.njit(cache=True)
def beta_qcd_as5(nf):
    r"""Compute the fourth coefficient of the QCD beta function.

    Implements Eq. (3.6) of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
        nf : int
            number of active flavors

    Returns
    -------
        beta_qcd_as5 : float
            fourth coefficient of the QCD beta function :math:`\\beta_qcd_as5^{n_f}`

    """
    beta_qcd_as5 = (
        149753.0 / 6.0
        + 3564.0 * zeta3
        + nf * (-1078361.0 / 162.0 - 6508.0 / 27.0 * zeta3)
        + nf**2 * (50065.0 / 162.0 + 6472.0 / 81.0 * zeta3)
        + 1093.0 / 729.0 * nf**3
    )
    return beta_qcd_as5


@nb.njit(cache=True)
def beta_qcd(k, nf):
    r"""Compute value of a beta_qcd coefficients.

    Parameters
    ----------
        k : tuple(int, int)
            perturbative order
        nf : int
            number of active flavors

    Returns
    -------
        beta_qcd : float
            beta_qcd_k(nf)

    """
    beta_ = 0
    if k == (2, 0):
        beta_ = beta_qcd_as2(nf)
    elif k == (3, 0):
        beta_ = beta_qcd_as3(nf)
    elif k == (4, 0):
        beta_ = beta_qcd_as4(nf)
    elif k == (5, 0):
        beta_ = beta_qcd_as5(nf)
    elif k == (2, 1):
        beta_ = beta_qcd_as2aem1(nf)
    else:
        raise ValueError("Beta_QCD coefficients beyond N3LO are not implemented!")
    return beta_


@nb.njit(cache=True)
def beta_qed(k, nf):
    r"""Compute value of a beta_qed coefficients.

    Parameters
    ----------
        k : tuple(int, int)
            perturbative order
        nf : int
            number of active flavors

    Returns
    -------
        beta_qed : float
            beta_qed_k(nf)

    """
    beta_ = 0
    if k == (0, 2):
        beta_ = beta_qed_aem2(nf)
    elif k == (0, 3):
        beta_ = beta_qed_aem3(nf)
    elif k == (1, 2):
        beta_ = beta_qed_aem2as1(nf)
    else:
        raise ValueError("Beta_QED coefficients beyond NLO are not implemented!")
    return beta_


@nb.njit(cache=True)
def b_qcd(k, nf):
    r"""Compute b_qcd coefficient.

    Parameters
    ----------
        k : tuple(int, int)
            perturbative order
        nf : int
            number of active flavors

    Returns
    -------
        b_qcd : float
            b_qcd_k(nf)

    """
    return beta_qcd(k, nf) / beta_qcd((2, 0), nf)


@nb.njit(cache=True)
def b_qed(k, nf):
    r"""Compute b_qed coefficient.

    Parameters
    ----------
        k : tuple(int, int)
            perturbative order
        nf : int
            number of active flavors

    Returns
    -------
        b_qed : float
            b_qed_k(nf)

    """
    return beta_qed(k, nf) / beta_qed((0, 2), nf)
