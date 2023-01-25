# -- coding: utf-8 --
"""This file contains the next-leading-order Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants

from .... import harmonics
from ....harmonics.constants import zeta3

# Non Singlet sector is swapped
from ...unpolarized.space_like.as3 import gamma_nsm as gamma_nsp
from ...unpolarized.space_like.as3 import gamma_nsp as gamma_nsm


@nb.njit(cache=True)
def gamma_ps(n, nf):
    """Compute the |NLO| polarized pure-singlet quark-quark anomalous dimension :cite:`Gluck:1995yr` (eq A.3).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors

    Returns
    -------
    complex
        |NLO| pure-singlet quark-quark anomalous dimension :math:`\\gamma_{ps}^{(1)}(n)`

    """
    gqqps1_nfcf = (2 * (n + 2) * (1 + 2 * n + np.power(n, 3))) / (
        np.power(1 + n, 3) * np.power(n, 3)
    )
    result = 4.0 * constants.TR * nf * constants.CF * gqqps1_nfcf
    return result


@nb.njit(cache=True)
def gamma_qg(n, nf, sx):
    """Compute the |NLO| polarized quark-gluon singlet anomalous dimension :cite:`Gluck:1995yr` (eq A.4).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    complex
        |NLO| quark-gluon singlet anomalous dimension :math:`\\gamma_{qg}^{(1)}(n)`

    """
    S1 = sx[0]
    S2 = sx[1]
    Sp2m = harmonics.S2((n - 1) / 2)
    gqg1_nfca = (
        (np.power(S1, 2) - S2 + Sp2m) * (n - 1) / (n * (n + 1))
        - 4 * S1 / (n * np.power(1 + n, 2))
        - (
            -2
            - 7 * n
            + 3 * np.power(n, 2)
            - 4 * np.power(n, 3)
            + np.power(n, 4)
            + np.power(n, 5)
        )
        / (np.power(n, 3) * np.power(1 + n, 3))
    ) * (2.0)
    gqg1_nfcf = (
        (-np.power(S1, 2) + S2 + 2 * S1 / n) * (n - 1) / (n * (n + 1))
        - (n - 1)
        * (1 + 3.5 * n + 4 * np.power(n, 2) + 5 * np.power(n, 3) + 2.5 * np.power(n, 4))
        / (np.power(n, 3) * np.power(1 + n, 3))
        + 4 * (n - 1) / (np.power(n, 2) * np.power(1 + n, 2)) * 2
    )
    result = (
        4.0 * constants.TR * nf * (constants.CA * gqg1_nfca + constants.CF * gqg1_nfcf)
    )
    return result


@nb.njit(cache=True)
def gamma_gq(n, nf, sx):
    """Compute the |NLO| polarized gluon-quark singlet anomalous dimension :cite:`Gluck:1995yr` (eq A.5).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    complex
        |NLO| gluon-quark singlet anomalous dimension :math:`\\gamma_{gq}^{(1)}(n)`

    """
    S1 = sx[0]
    S2 = sx[1]
    Sp2m = harmonics.S2((n - 1) / 2)
    ggq1_cfcf = (
        ((np.power(S1, 2) + S2) * (n + 2)) / (n * (n + 1))
        - (2 * S1 * (n + 2) * (1 + 3 * n)) / (n * np.power(1 + n, 2))
        - (
            (n + 2)
            * (
                2
                + 15 * n
                + 8 * np.power(n, 2)
                - 12.0 * np.power(n, 3)
                - 9.0 * np.power(n, 4)
            )
        )
        / (np.power(n, 3) * np.power(1 + n, 3))
        + 8 * (n + 2) / (np.power(n, 2) * np.power(1 + n, 2))
    )
    ggq1_cfca = (
        (-np.power(S1, 2) - S2 + Sp2m) * (n + 2) / (n * (n + 1))
        + S1 * (12 + 22 * n + 11 * np.power(n, 2)) / (3 * np.power(n, 2) * (n + 1))
        - (
            36
            + 72 * n
            + 41 * np.power(n, 2)
            + 254 * np.power(n, 3)
            + 271 * np.power(n, 4)
            + 76 * np.power(n, 5)
        )
        / (9 * np.power(n, 3) * np.power(1 + n, 3))
    )
    ggq1_cfnf = 4 * (
        (-S1 * (n + 2)) / (3 * n * (n + 1))
        + ((n + 2) * (2 + 5 * n)) / (9 * n * np.power(1 + n, 2))
    )
    result = constants.CF * (
        (constants.CA * ggq1_cfca)
        + (constants.CF * ggq1_cfcf)
        + (4.0 * constants.TR * nf * ggq1_cfnf)
    )
    return result


@nb.njit(cache=True)
def gamma_gg(n, nf, sx):
    """Compute the |NLO| polarized gluon-gluon singlet anomalous dimension :cite:`Gluck:1995yr` (eq A.6).

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        number of active flavors
    sx : numpy.ndarray
        List of harmonic sums: :math:`S_{1},S_{2}`

    Returns
    -------
    complex
        |NLO| gluon-quark singlet anomalous dimension :math:`\\gamma_{gq}^{(1)}(n)`

    """
    S1 = sx[0]
    Sp2m = harmonics.S2((n - 1) / 2)
    Sp3m = harmonics.S3((n - 1) / 2)
    Sm1 = harmonics.Sm1(n, S1, True)
    Sm21 = harmonics.Sm21(n, S1, Sm1, True)
    SSCHLM = -S1 / n**2 - 5 * zeta3 / 4 - Sm21
    ggg1_caca = (
        -4 * S1 * Sp2m
        - Sp3m
        + 8 * SSCHLM
        + 8 * Sp2m / (n * (n + 1))
        + 2.0
        * S1
        * (
            72
            + 144 * n
            + 67 * np.power(n, 2)
            + 134 * np.power(n, 3)
            + 67 * np.power(n, 4)
        )
        / (9 * np.power(n, 2) * np.power(n + 1, 2))
        - (
            144
            + 258 * n
            + 7 * np.power(n, 2)
            + 698 * np.power(n, 3)
            + 469 * np.power(n, 4)
            + 144 * np.power(n, 5)
            + 48 * np.power(n, 6)
        )
        / (9 * np.power(n, 3) * np.power(1 + n, 3))
    ) * (0.5)
    ggg1_canf = (
        -5 * S1 / 9
        + (-3 + 13 * n + 16 * np.power(n, 2) + 6 * np.power(n, 3) + 3 * np.power(n, 4))
        / (9 * np.power(n, 2) * np.power(1 + n, 2))
    ) * 4
    ggg1_cfnf = (
        4
        + 2 * n
        - 8 * np.power(n, 2)
        + np.power(n, 3)
        + 5 * np.power(n, 4)
        + 3 * np.power(n, 5)
        + np.power(n, 6)
    ) / (np.power(n, 3) * np.power(1 + n, 3))
    # fmt: on
    result = 4 * (
        constants.CA * constants.CA * ggg1_caca
        + constants.TR * nf * (constants.CA * ggg1_canf + constants.CF * ggg1_cfnf)
    )

    return result


@nb.njit(cache=True)
def gamma_singlet(n, nf, sx):
    r"""Compute the |NLO| polarized singlet anomalous dimension matrix.

        .. math::
            \gamma_S^{(1)} = \left(\begin{array}{cc}
            \gamma_{qq}^{(1)} & \gamma_{qg}^{(1)}\\
            \gamma_{gq}^{(1)} & \gamma_{gg}^{(1)}
            \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx: list
        harmonics cache

    Returns
    -------
    numpy.ndarray
        |NLO| singlet anomalous dimension matrix :math:`\gamma_{S}^{(1)}(N)`

    """
    gamma_qq = gamma_nsp(n, nf, sx) + gamma_ps(n, nf)
    gamma_S_0 = np.array(
        [[gamma_qq, gamma_qg(n, nf, sx)], [gamma_gq(n, nf, sx), gamma_gg(n, nf, sx)]],
        np.complex_,
    )
    return gamma_S_0
