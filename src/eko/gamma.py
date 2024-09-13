r"""The |QCD| gamma function coefficients.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import numba as nb

from eko.constants import zeta3, zeta4, zeta5


@nb.njit(cache=True)
def gamma_qcd_as1():
    r"""Compute the first coefficient of the |QCD| gamma function.

    Implements :eqref:`15` of :cite:`Vermaseren:1997fq`.

    Returns
    -------
    gamma_0 : float
        first coefficient of the |QCD| gamma function :math:`\gamma_{m,0}^{n_f}`
    """
    return 4.0


@nb.njit(cache=True)
def gamma_qcd_as2(nf):
    r"""Compute the second coefficient of the |QCD| gamma function.

    Implements :eqref:`15` of :cite:`Vermaseren:1997fq`.

    Parameters
    ----------
    nf : int
        number of active flavors

    Returns
    -------
    gamma_1 : float
        second coefficient of the |QCD| gamma function :math:`\gamma_{m,1}^{n_f}`
    """
    return 202.0 / 3.0 - 20.0 / 9.0 * nf


@nb.njit(cache=True)
def gamma_qcd_as3(nf):
    r"""Compute the third coefficient of the |QCD| gamma function.

    Implements :eqref:`15` of :cite:`Vermaseren:1997fq`.

    Parameters
    ----------
    nf : int
        number of active flavors

    Returns
    -------
    gamma_2 : float
        third coefficient of the |QCD| gamma function :math:`\gamma_{m,2}^{n_f}`
    """
    return 1249.0 - (2216.0 / 27.0 + 160.0 / 3.0 * zeta3) * nf - 140.0 / 81.0 * nf**2


@nb.njit(cache=True)
def gamma_qcd_as4(nf):
    r"""Compute the fourth coefficient of the |QCD| gamma function.

    Implements :eqref:`15` of :cite:`Vermaseren:1997fq`.

    Parameters
    ----------
    nf : int
        number of active flavors

    Returns
    -------
    gamma_3 : float
        fourth coefficient of the |QCD| gamma function :math:`\gamma_{m,3}^{n_f}`
    """
    return (
        4603055.0 / 162.0
        + 135680.0 * zeta3 / 28.0
        - 8800.0 * zeta5
        + (
            -91723.0 / 27.0
            - 34192.0 * zeta3 / 9.0
            + 880.0 * zeta4
            + 18400.0 * zeta5 / 9.0
        )
        * nf
        + (5242.0 / 243.0 + 800.0 * zeta3 / 9.0 - 160.0 * zeta4 / 3.0) * nf**2
        + (332.0 / 243.0 + 64.0 * zeta3 / 27.0) * nf**3
    )


@nb.njit(cache=True)
def gamma(order, nf):
    """Compute the value of a |QCD| gamma coefficient.

    Parameters
    ----------
    order: int
        perturbative order
    nf : int
        number of active flavors

    Returns
    -------
    gamma : float
        |QCD| gamma function coefficient
    """
    _gamma = 0.0

    if order == 1:
        _gamma = gamma_qcd_as1()
    elif order == 2:
        _gamma = gamma_qcd_as2(nf)
    elif order == 3:
        _gamma = gamma_qcd_as3(nf)
    elif order == 4:
        _gamma = gamma_qcd_as4(nf)
    else:
        raise ValueError("QCD gamma coefficients beyond N3LO are not implemented!")
    return _gamma
