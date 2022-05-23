# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,+}^{(3)}`
"""
import numba as nb

from ...constants import CA, CF
from .gNSm import gamma_ns_nf3, gamma_nsm_nf2

# def A_3(n, sx):
#     """
#     Implements the common part proportional to :math:`nf^2`,
#     of :math:`\\gamma_{ns,+}^{(3)},`\\gamma_{ns,-}^{(3)},`\\gamma_{ns,v}^{(3)}`

#     The expression is taken from eq. 3.1 of :cite:`Davies:2016jie`

#     Parameters
#     ----------
#         n : complex
#             Mellin moment
#         sx : list
#             harmonic sums cache

#     Returns
#     -------
#         A_3 : complex
#             |N3LO| non-singlet anomalous dimension common part proportional to :math:`nf^2 C_F^2`
#     """
#     S1, _ = sx[0]
#     S2, _ = sx[1]
#     S3, _, _, _, _, _ = sx[2]
#     S4, S31, _, _, _, _, _ = sx[3]
#     S5, _ = sx[4]
#     S41 = sf.S41(n, S1, S2, S3)
#     S311 = sf.S311(n, S1, S2)
#     S23 = sf.S23(n, S1, S2, S3) if n != 1 else 1.000001659
#     return (
#         1
#         / 81
#         * (
#             1143
#             - 1728 / (1 + n) ** 5
#             - 864 / (n**5 * (1 + n) ** 5)
#             - 2736 / (1 + n) ** 4
#             - 7088 / (1 + n) ** 3
#             - 40920 / (1 + n) ** 2
#             + 20681 / (n + n**2)
#             + (1440 * S1) / (n**4 * (1 + n) ** 4)
#             - 10072 * S2
#             + (6144 * S2) / (1 + n) ** 2
#             - (3328 * S2) / (n + n**2)
#             + 1920 * S2**2
#             + (16 * (59 + 54 * S1 + 72 * S2)) / (n**3 * (1 + n) ** 3)
#             + 2304 * S23
#             + 7968 * S3
#             - (1152 * S3) / (1 + n) ** 2
#             - (384 * S3) / (n + n**2)
#             - 3456 * S2 * S3
#             + 3840 * S31
#             + (576 * S31) / (n + n**2)
#             + 2304 * S311
#             - 3792 * S4
#             - (288 * S4) / (n + n**2)
#             - 4608 * S41
#             + 576 * S5
#             - 2376 * zeta3
#             - (1728 * zeta3) / (1 + n) ** 2
#             - (2880 * zeta3) / (n + n**2)
#             - 1728 * S2 * zeta3
#             + (-4550 - 7312 * S1 + 96 * S2 + 576 * S3 + 864 * zeta3)
#             / (n**2 * (1 + n) ** 2)
#             + 2
#             * S1
#             * (
#                 2119
#                 - 1728 / (1 + n) ** 4
#                 - 576 / (1 + n) ** 3
#                 - 3392 / (1 + n) ** 2
#                 + 3392 / (n + n**2)
#                 - 608 * S2
#                 + 960 * S3
#                 - 576 * S31
#                 + 288 * S4
#                 + 2880 * zeta3
#                 - 1296 * zeta4
#             )
#             + 1944 * zeta4
#             + (1296 * zeta4) / (n + n**2)
#         )
#     )


# def B_3p(n, sx):
#     """
#     Parametrization of eq. 3.2 of :cite:`Davies:2016jie`.
#     This contribution is sub-leading with respect to A_3.
#     The exact expression contains weight 5 harmonics sum not yet
#     implemented in eko.

#     Parameters
#     ----------
#         n : complex
#             Mellin moment
#         sx : list
#             harmonic sums cache

#     Returns
#     -------
#         B_3p : complex
#             |N3LO| singlet-like non-singlet anomalous dimension part
#             proportional to :math:`C_F (C_A - 2 C_F) nf^2`
#     """
#     S1 = sx[0][0]
#     # return -52.6196 + 58.6005 * S1 - 28.9527 * S1 / n**2 + 16.6447 * S1 / n
#     return (
#         -51.8789
#         + 478.99 / (1 + n) ** 4
#         - 430.56 / (n**4 * (1 + n) ** 4)
#         - 387.328 / (1 + n) ** 3
#         + 177.364 / (n**3 * (1 + n) ** 3)
#         + 58.485 * S1
#         - (3.3994 * S1) / n**2
#         + (13.7033 * S1) / n
#     )


def deltaB3(n, sx):
    """
    Parametrization of eq. 3.4 of :cite:`Davies:2016jie`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        B_3m : complex
            |N3LO| valence-like non-singlet anomalous dimension part
            proportional to :math:`C_F (C_A - 2 C_F) nf^2`
    """
    S1, _ = sx[0]
    S2, Sm2 = sx[1]
    deltaB = (
        16
        / (81 * n**5 * (1 + n) ** 5)
        * (
            -54
            + 3 * n**8 * (-13 + 6 * S1)
            - 12 * n * (5 + 6 * S1)
            + 6 * n**7 * (-23 + 12 * S1)
            + n**4 * (73 + 696 * S1 - 252 * S1**2 - 252 * S2 - 252 * Sm2)
            + n**3 * (367 + 174 * S1 - 144 * S1**2 - 144 * S2 - 144 * Sm2)
            - n**2 * (-211 + 132 * S1 + 36 * S1**2 + 36 * S2 + 36 * Sm2)
            - n**6 * (475 - 474 * S1 + 72 * S1**2 + 72 * S2 + 72 * Sm2)
            - 3 * n**5 * (85 - 294 * S1 + 72 * S1**2 + 72 * S2 + 72 * Sm2)
        )
    )
    return deltaB


@nb.njit(cache=True)
def gamma_nsp_nf2(n, sx):
    """
    Implements the singlet-like non-singlet part proportional to :math:`nf^2`.
    This parametrization takes the advantage of eq 3.3 which is known exactly
    and relatively easy.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        g_nsp_nf2 : complex
            |N3LO| singlet-like non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^2}`

    See Also
    --------
        A_3: :math:`A^{(3)}`
        B_3p: :math:`B^{(3)}_{+}`
    """
    return gamma_nsm_nf2(n, sx) - CF * (CA - 2 * CF) * deltaB3(n, sx)


@nb.njit(cache=True)
def gamma_nsp_nf1(n, sx):
    """
    Implements the singlet-like non-singlet part proportional to :math:`nf^1`.
    The expression is the average of the Mellin transform
    of eq. 4.12, 4.13  of :cite:`Moch:2017uml`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        g_nsp_nf1 : complex
            |N3LO| sea non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^1}`
    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return (
        5551.1105
        - 126.419752 / n**6
        + 752.197524 / n**5
        - 2506.4001 / n**4
        + 5760.2261 / n**3
        - 9239.374 / n**2
        + 119259.11141693963 / n
        + 495.79776 / (1 + n) ** 4
        + 1230.0 / (1 + n) ** 3
        - 7077.807 / (1 + n) ** 2
        - 55456.1555 / (1 + n)
        + 42292.153 / (2 + n)
        - 13596.323 / (3 + n)
        + 7896.0 / (4 + n)
        - 5171.934055 * S1
        - (63007.5 * S1) / n**2
        + (59940.977 * S1) / n
        + (30.797760 * S1) / (1 + n) ** 3
        - (62682.807 * S1) / (1 + n)
        + (15.39888 * S1**2) / (1 + n) ** 2
        - (5.13296 * S1**3) / n
        + (5.13296 * S1**3) / (1 + n)
        - (63007.5 * S2) / n
        + (15.39888 * S2) / (1 + n) ** 2
        - (15.39888 * S1 * S2) / n
        + (15.39888 * S1 * S2) / (1 + n)
        - (10.26592 * S3) / n
        + (10.26592 * S3) / (1 + n)
    )


@nb.njit(cache=True)
def gamma_nsp_nf0(n, sx):
    """
    Implements the singlet-like non-singlet part proportional to :math:`nf^0`.
    The expression is the average of the Mellin transform
    of eq. 4.12, 4.13  of :cite:`Moch:2017uml`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        g_nsp_nf0 : complex
            |N3LO| sea non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^0}`
    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return (
        -23396.42
        - 252.83952 / n**7
        + 1580.24688 / n**6
        - 5806.79928 / n**5
        + 15972.888 / n**4
        - 32592.036 / n**3
        + 52366.195 / n**2
        + 471083.45095404686 / n
        - 56.685 / (1 + n) ** 4
        + 2581.794 / (1 + n) ** 3
        + 11825.0 / (1 + n) ** 2
        - 129113.005 / (1 + n)
        + 260862.515 / (2 + n)
        - 87390.43 / (3 + n)
        + 45750.0 / (4 + n)
        + 20702.8395 * S1
        - (334400.0 * S1) / n**2
        + (302200.933 * S1) / n
        - (56.685 * S1) / (1 + n) ** 3
        - (1951.206 * S1) / (1 + n) ** 2
        - (285250.0 * S1) / (1 + n)
        + (975.603 * S1**2) / n
        - (28.3425 * S1**2) / (1 + n) ** 2
        - (975.603 * S1**2) / (1 + n)
        + (9.4475 * S1**3) / n
        - (9.4475 * S1**3) / (1 + n)
        - (333424.397 * S2) / n
        - (28.3425 * S2) / (1 + n) ** 2
        - (975.603 * S2) / (1 + n)
        + (28.3425 * S1 * S2) / n
        - (28.3425 * S1 * S2) / (1 + n)
        + (18.895 * S3) / n
        - (18.895 * S3) / (1 + n)
    )


@nb.njit(cache=True)
def gamma_nsp(n, nf, sx):
    """
    Computes the |N3LO| singlet-like non-singlet anomalous dimension.

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
        gamma_nsp : complex
            |N3LO| singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(3)}(N)`

    See Also
    --------
        gamma_nsp_nf0: :math:`\\gamma_{ns,+}^{(3)}|_{nf^0}`
        gamma_nsp_nf1: :math:`\\gamma_{ns,+}^{(3)}|_{nf^1}`
        gamma_nsp_nf2: :math:`\\gamma_{ns,+}^{(3)}|_{nf^2}`
        gamma_ns_nf3: :math:`\\gamma_{ns}^{(3)}|_{nf^3}`
    """
    return (
        gamma_nsp_nf0(n, sx)
        + nf * gamma_nsp_nf1(n, sx)
        + nf**2 * gamma_nsp_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
