# -*- coding: utf-8 -*-
r"""The anomalous dimension :math:`\gamma_{gq}^{(3)}`."""
import numba as nb
import numpy as np

from ...harmonics.log_functions import lm13, lm13m1, lm14, lm15


@nb.njit(cache=True)
def gamma_gq_nf3(n, sx):
    r"""Implement the part proportional to :math:`nf^3` of :math:`\gamma_{gq}^{(3)}`.

    The expression is copied exact from Eq. 3.13 of :cite:`Davies:2016jie`.

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
def gamma_gq_nf0(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^0` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation: "a" ,"b", "best"

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
    common = (
        -22156.31283903764 / np.power(-1.0 + n, 4)
        - 37609.87654320987 / np.power(n, 7)
        - 35065.67901234568 / np.power(n, 6)
        - 175454.58483973087 / np.power(n, 5)
        - 375.3983146907502 * lm14(n, S1, S2, S3, S4)
        - 13.443072702331962 * lm15(n, S1, S2, S3, S4, S5)
    )
    if variation != "b":
        fit_a = (
            63019.91215580807 / np.power(-1.0 + n, 3)
            - 52669.928305100126 / np.power(-1.0 + n, 2)
            - 1600.0895275985033 * lm13(n, S1, S2, S3)
            + 21359.289880116714 * lm13m1(n, S1, S2, S3)
        )
        fit = fit_a
    if variation != "a":
        fit_b = (
            95032.88047770769 / np.power(-1.0 + n, 3)
            - 330966.2248552476 / (-1.0 + n)
            + 948843.5926373288 / np.power(n, 2)
            - 2525.043680756643 * lm13(n, S1, S2, S3)
            - 189126.67461470052 * lm13m1(n, S1, S2, S3)
        )
        fit = fit_b
    if variation == "best":
        fit = (fit_a + fit_b) / 2
    return common + fit


@nb.njit(cache=True)
def gamma_gq_nf1(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^1` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation: "a" ,"b", "best"

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
    common = (
        5309.62962962963 / np.power(n, 7)
        + 221.23456790123456 / np.power(n, 6)
        + 9092.91243376357 / np.power(n, 5)
        + 34.49474165523548 * lm14(n, S1, S2, S3, S4)
        + 0.5486968449931413 * lm15(n, S1, S2, S3, S4, S5)
    )
    if variation != "b":
        fit_a = (
            -4989.943819279911 / np.power(-1.0 + n, 3)
            + 9496.873384515286 / np.power(-1.0 + n, 2)
            + 215.78642939301727 * lm13(n, S1, S2, S3)
            - 4337.106332800244 * lm13m1(n, S1, S2, S3)
        )
        fit = fit_a
    if variation != "a":
        fit_b = (
            885.6738165500071 / np.power(-1.0 + n, 3)
            - 2720.558228705253 / (-1.0 + n)
            + 24465.675802706868 / np.power(n, 2)
            + 175.94743712961272 * lm13(n, S1, S2, S3)
            - 4407.858015480961 * lm13m1(n, S1, S2, S3)
        )
        fit = fit_b
    if variation == "best":
        fit = (fit_a + fit_b) / 2
    return common + fit


@nb.njit(cache=True)
def gamma_gq_nf2(n, sx, variation):
    r"""Implement the part proportional to :math:`nf^2` of :math:`\gamma_{gq}^{(3)}`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation: "a" ,"b", "best"

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{gq}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    common = 778.5349794238683 / np.power(n, 5) - 0.877914951989026 * lm14(
        n, S1, S2, S3, S4
    )
    if variation != "b":
        fit_a = (
            -215.98018280331252 / np.power(-1.0 + n, 2)
            - 18.06611461001508 / (-1.0 + n)
            - 4.756294267863538 * lm13(n, S1, S2, S3)
            - 44.54796646245588 * lm13m1(n, S1, S2, S3)
        )
        fit = fit_a
    if variation != "a":
        fit_b = (
            -672.0138626871961 / (-1.0 + n)
            + 1686.9653718090199 / np.power(n, 2)
            - 6.220583718141115 * lm13(n, S1, S2, S3)
            - 479.1441862431653 * lm13m1(n, S1, S2, S3)
        )
        fit = fit_b
    if variation == "best":
        fit = (fit_a + fit_b) / 2
    return common + fit


@nb.njit(cache=True)
def gamma_gq(n, nf, sx, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : list
        harmonic sums cache
    variation : str
        |N3LO| anomalous dimension variation: "a" ,"b", "best"

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
        gamma_gq_nf0(n, sx, variation)
        + nf * gamma_gq_nf1(n, sx, variation)
        + nf**2 * gamma_gq_nf2(n, sx, variation)
        + nf**3 * gamma_gq_nf3(n, sx)
    )
