# -*- coding: utf-8 -*-
r"""
This module contains the QCD gamma function coefficients.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import scipy.special
from .anomalous_dimensions.harmonics import zeta3, zeta4


def gamma_0():
    return 4.0


def gamma_1(nf):
    return 202.0 / 3.0 - 20.0 / 9.0 * nf


def gamma_2(nf):
    return 1249.0 - (2216.0 / 27.0 + 160.0 / 3.0 * zeta3) * nf - 140.0 / 81.0 * nf ** 2


def gamma_3(nf):
    zeta5 = scipy.special.zeta(5)
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
        + (5242.0 / 243.0 + 800.0 * zeta3 / 9.0 - 160.0 * zeta4 / 3.0) * nf ** 2
        + (332.0 / 243.0 + 64.0 * zeta3 / 27.0) * nf ** 3
    )


def gamma(order, nf):
    """
    Compute the value of a gamma coefficient

    Parameters
    ----------
        order: int
            perturbative order
        nf : int
            number of active flavors

    returns
    -------
        gamma: float
            QCD gamma function coefficient
    """
    _gamma = 0.0

    if order == 0:
        _gamma = gamma_0()
    elif order == 1:
        _gamma = gamma_1(nf)
    elif order == 2:
        _gamma = gamma_2(nf)
    elif order == 3:
        _gamma = gamma_3(nf)
    else:
        raise ValueError("Gamma coefficients beyond N3LO are not implemented!")
    return _gamma
