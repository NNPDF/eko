r"""Compute the beta function coefficients.

The strong coupling :math:`a_s(\mu^2)` and the electro-magnetic coupling
:math:`a_{em}(\mu^2)` obey separate, but coupled |RGE|:

.. math::

    \mu^2 \frac{da_s(\mu^2)}{d\mu^2} = \beta_{QCD}^{(n_f)}(a_s(\mu^2), a_{em}(\mu^2))
        = - \sum_{j=2}\sum_{k=0} \beta_{QCD}^{(j,k),(n_f)}
                  \left(a_s(\mu^2)\right)^j \left(a_{em}(\mu^2)\right)^k\\

    \mu^2 \frac{da_{em}(\mu^2)}{d\mu^2} = \beta_{QED}^{(n_f)}(a_s(\mu^2), a_{em}(\mu^2))
        = - \sum_{j=0}\sum_{k=2} \beta_{QED}^{(j,k),(n_f)}
                  \left(a_s(\mu^2)\right)^j \left(a_{em}(\mu^2)\right)^k

See the :mod:`eko.couplings` module for solutions.
When considering QED corrections the two |RGE| need two be solved
simultaneously.
"""

from typing import Tuple

import numba as nb

from . import constants
from .constants import zeta3


@nb.njit(cache=True)
def beta_qcd_as2(nf: int) -> float:
    r"""Compute the first coefficient of the QCD beta function.

    Implements :eqref:`3.1` of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
    nf :
        number of active flavors

    Returns
    -------
    float
        first coefficient of the QCD beta function :math:`\beta_{QCD}^{(2,0),(n_f)}`
    """
    return 11.0 / 3.0 * constants.CA - 4.0 / 3.0 * constants.TR * nf


@nb.njit(cache=True)
def beta_qed_aem2(nf: int, nl: int) -> float:
    r"""Compute the first coefficient of the QED beta function.

    Implements :eqref:`7` of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
    nf :
        number of active flavors
    nl :
        number of leptons

    Returns
    -------
    float
        first coefficient of the QED beta function :math:`\beta_{QED}^{(0,2),(n_f)}`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return -4.0 / 3 * (nl + constants.NC * (nu * constants.eu2 + nd * constants.ed2))


@nb.njit(cache=True)
def beta_qcd_as3(nf: int) -> float:
    r"""Compute the second coefficient of the QCD beta function.

    Implements :eqref:`3.2` of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
    nf :
        number of active flavors

    Returns
    -------
    float
        second coefficient of the QCD beta function :math:`\beta_{QCD}^{(3,0),(n_f)}`
    """
    TF = constants.TR * nf
    b_ca2 = 34.0 / 3.0 * constants.CA * constants.CA
    b_ca = -20.0 / 3.0 * constants.CA * TF
    b_cf = -4.0 * constants.CF * TF
    return b_ca2 + b_ca + b_cf


@nb.njit(cache=True)
def beta_qed_aem3(nf: int, nl: int) -> float:
    r"""Compute the second coefficient of the QED beta function.

    Implements :eqref:`7` of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
    nf :
        number of active flavors
    nl :
        number of leptons

    Returns
    -------
    float
        second coefficient of the QED beta function :math:`\beta_{QED}^{(0,3),(n_f)}`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return -4.0 * (nl + constants.NC * (nu * constants.eu2**2 + nd * constants.ed2**2))


@nb.njit(cache=True)
def beta_qcd_as2aem1(nf: int) -> float:
    r"""Compute the first QED correction of the QCD beta function.

    Implements :eqref:`7` of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
    nf :
        number of active flavors

    Returns
    -------
    float
        first QED correction of the QCD beta function :math:`\beta_{QCD}^{(2,1),(n_f)}`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return -4.0 * constants.TR * (nu * constants.eu2 + nd * constants.ed2)


@nb.njit(cache=True)
def beta_qed_aem2as1(nf: int) -> float:
    r"""Compute the first QCD correction of the QED beta function.

    Implements :eqref:`7` of :cite:`Surguladze:1996hx`.

    Parameters
    ----------
    nf :
        number of active flavors

    Returns
    -------
    float
        first QCD correction of the QED beta function :math:`\beta_{QED}^{(1,2),(n_f)}`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return (
        -4.0 * constants.CF * constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    )


@nb.njit(cache=True)
def beta_qcd_as4(nf: int) -> float:
    r"""Compute the third coefficient of the QCD beta function.

    Implements :eqref:`3.3` of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
    nf :
        number of active flavors

    Returns
    -------
    float
        third coefficient of the QCD beta function :math:`\beta_{QCD}^{(4,0),(n_f)}`
    """
    TF = constants.TR * nf
    return (
        2857.0 / 54.0 * constants.CA * constants.CA * constants.CA
        - 1415.0 / 27.0 * constants.CA * constants.CA * TF
        - 205.0 / 9.0 * constants.CF * constants.CA * TF
        + 2.0 * constants.CF * constants.CF * TF
        + 44.0 / 9.0 * constants.CF * TF * TF
        + 158.0 / 27.0 * constants.CA * TF * TF
    )


@nb.njit(cache=True)
def beta_qcd_as5(nf: int) -> float:
    r"""Compute the fourth coefficient of the QCD beta function.

    Implements :eqref:`3.6` of :cite:`Herzog:2017ohr`.

    Parameters
    ----------
    nf :
        number of active flavors

    Returns
    -------
    float
        fourth coefficient of the QCD beta function :math:`\beta_{QCD}^{(5,0),(n_f)}`
    """
    return (
        149753.0 / 6.0
        + 3564.0 * zeta3
        + nf * (-1078361.0 / 162.0 - 6508.0 / 27.0 * zeta3)
        + nf**2 * (50065.0 / 162.0 + 6472.0 / 81.0 * zeta3)
        + 1093.0 / 729.0 * nf**3
    )


@nb.njit(cache=True)
def beta_qcd(k: Tuple[int, int], nf: int) -> float:
    r"""Compute QCD beta coefficient.

    Parameters
    ----------
    k :
        perturbative orders
    nf :
        number of active flavors

    Returns
    -------
    float
        :math:`\beta_{QCD}^{k,(nf)}`
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
def beta_qed(k: Tuple[int, int], nf: int, nl: int) -> float:
    r"""Compute QED beta coefficient.

    Parameters
    ----------
    k :
        perturbative order
    nf :
        number of active flavors
    nl :
        number of leptons

    Returns
    -------
    float
        :math:`\beta_{QED}^{k,(nf)}`
    """
    beta_ = 0
    if k == (0, 2):
        beta_ = beta_qed_aem2(nf, nl)
    elif k == (0, 3):
        beta_ = beta_qed_aem3(nf, nl)
    elif k == (1, 2):
        beta_ = beta_qed_aem2as1(nf)
    else:
        raise ValueError("Beta_QED coefficients beyond NLO are not implemented!")
    return beta_


@nb.njit(cache=True)
def b_qcd(k: Tuple[int, int], nf: int) -> float:
    r"""Compute normalized QCD beta coefficient.

    Parameters
    ----------
    k :
        perturbative order
    nf :
        number of active flavors

    Returns
    -------
    float
        :math:`b_{QCD}^{k,(nf)} = \beta_{QCD}^{k,(nf)} / \beta_{QCD}^{(2,0),(nf)}`
    """
    return beta_qcd(k, nf) / beta_qcd((2, 0), nf)


@nb.njit(cache=True)
def b_qed(k: Tuple[int, int], nf: int, nl: int) -> float:
    r"""Compute normalized QED beta coefficient.

    Parameters
    ----------
    k :
        perturbative order
    nf :
        number of active flavors
    nl :
        number of leptons

    Returns
    -------
    float
        :math:`b_{QED}^{k,(nf)} = \beta_{QED}^{k,(nf)} / \beta_{QED}^{(0,2),(nf)}`
    """
    return beta_qed(k, nf, nl) / beta_qed((0, 2), nf, nl)
