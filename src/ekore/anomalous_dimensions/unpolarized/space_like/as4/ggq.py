r"""This module contains the anomalous dimension :math:`\gamma_{gq}^{(3)}`
"""
import numba as nb
import numpy as np

from .....harmonics.log_functions import lm13, lm13m1, lm14, lm15


@nb.njit(cache=True)
def gamma_gq_nf3(n, sx):
    r"""Implements the part proportional to :math:`nf^3` of :math:`\gamma_{gq}^{(3)}`,
    the expression is copied exact from Eq. 3.13 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return 1.3333333333333333 * (
        -11.39728026699467 / (-1.0 + n)
        + 11.39728026699467 / n
        - 2.3703703703703702 / np.power(1.0 + n, 4)
        + 6.320987654320987 / np.power(1.0 + n, 3)
        - 3.1604938271604937 / np.power(1.0 + n, 2)
        - 5.698640133497335 / (1.0 + n)
        - (6.320987654320987 * S1) / (-1.0 + n)
        + (6.320987654320987 * S1) / n
        - (2.3703703703703702 * S1) / np.power(1.0 + n, 3)
        + (6.320987654320987 * S1) / np.power(1.0 + n, 2)
        - (3.1604938271604937 * S1) / (1.0 + n)
        + (6.320987654320987 * (np.power(S1, 2) + S2)) / (-1.0 + n)
        - (6.320987654320987 * (np.power(S1, 2) + S2)) / n
        - (1.1851851851851851 * (np.power(S1, 2) + S2)) / np.power(1.0 + n, 2)
        + (3.1604938271604937 * (np.power(S1, 2) + S2)) / (1.0 + n)
        - (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (-1.0 + n)
        + (0.7901234567901234 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3)) / n
        - (0.3950617283950617 * (np.power(S1, 3) + 3.0 * S1 * S2 + 2.0 * S3))
        / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_gq_nf0(n, sx):
    r"""Implements the part proportional to :math:`nf^0` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    S5 = sx[4][0]
    return (
        -22156.31283903764 / np.power(-1.0 + n, 4)
        + 63019.91215580799 / np.power(-1.0 + n, 3)
        - 52669.92830510009 / np.power(-1.0 + n, 2)
        - 37609.87654320987 / np.power(n, 7)
        - 35065.67901234568 / np.power(n, 6)
        - 175454.58483973087 / np.power(n, 5)
        - 1600.0895275985124 * lm13(n, S1, S2, S3)
        + 21359.28988011691 * lm13m1(n, S1, S2, S3)
        - 375.3983146907502 * lm14(n, S1, S2, S3, S4)
        - 13.443072702331962 * lm15(n, S1, S2, S3, S4, S5)
    )


@nb.njit(cache=True)
def gamma_gq_nf1(n, sx):
    r"""Implements the part proportional to :math:`nf^1` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    S5 = sx[4][0]
    return (
        -4989.9438192798825 / np.power(-1.0 + n, 3)
        + 9496.873384515262 / np.power(-1.0 + n, 2)
        + 5309.62962962963 / np.power(n, 7)
        + 221.23456790123456 / np.power(n, 6)
        + 9092.91243376357 / np.power(n, 5)
        + 215.7864293930184 * lm13(n, S1, S2, S3)
        - 4337.106332800272 * lm13m1(n, S1, S2, S3)
        + 34.49474165523548 * lm14(n, S1, S2, S3, S4)
        + 0.5486968449931413 * lm15(n, S1, S2, S3, S4, S5)
    )


@nb.njit(cache=True)
def gamma_gq_nf2(n, sx):
    r"""Implements the part proportional to :math:`nf^2` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    return (
        -215.9801828033175 / np.power(-1.0 + n, 2)
        - 18.066114610010438 / (-1.0 + n)
        + 778.5349794238683 / np.power(n, 5)
        - 4.756294267863632 * lm13(n, S1, S2, S3)
        - 44.54796646244799 * lm13m1(n, S1, S2, S3)
        - 0.877914951989026 * lm14(n, S1, S2, S3, S4)
    )


@nb.njit(cache=True)
def gamma_gq(n, nf, sx):
    r"""Computes the |N3LO| gluon-quark singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        |N3LO| gluon-quark singlet anomalous dimension
        :math:`\gamma_{gq}^{(3)}(N)`

    See Also
    --------
    gamma_gq_nf0: :math:`\gamma_{gq}^{(3)}|_{nf^0}`
    gamma_gq_nf1: :math:`\gamma_{gq}^{(3)}|_{nf^1}`
    gamma_gq_nf2: :math:`\gamma_{gq}^{(3)}|_{nf^2}`
    gamma_gq_nf3: :math:`\gamma_{gq}^{(3)}|_{nf^3}`

    """
    return (
        gamma_gq_nf0(n, sx)
        + nf * gamma_gq_nf1(n, sx)
        + nf**2 * gamma_gq_nf2(n, sx)
        + nf**3 * gamma_gq_nf3(n, sx)
    )
