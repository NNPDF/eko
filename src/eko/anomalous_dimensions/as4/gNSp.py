# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,+}^{(3)}`
"""
import numba as nb

from .gNSm import gamma_ns_nf3

# def A_3(n, sx):
#     """
#     Implements the common part proportional to :math:`nf^2`,
#     of :math:`\\gamma_{ns,+}^{(3)},`\\gamma_{ns,-}^{(3)},`\\gamma_{ns,v}^{(3)}`

#     The expression is taken from Eq. 3.1 of :cite:`Davies:2016jie`

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
#     Parametrization of Eq. 3.2 of :cite:`Davies:2016jie`.
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
    Implementation of Eq. 3.4 of :cite:`Davies:2016jie`.

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
        256 / (9 * (1 + n) ** 5)
        - 32 / (3 * n**5 * (1 + n) ** 5)
        - 832 / (9 * (1 + n) ** 4)
        - 320 / (27 * n**4 * (1 + n) ** 4)
        - 1856 / (27 * (1 + n) ** 3)
        + 4336 / (81 * n**3 * (1 + n) ** 3)
        - 1952 / (27 * (1 + n) ** 2)
        - 2800 / (81 * n**2 * (1 + n) ** 2)
        + 1744 / (27 * n * (1 + n))
        + (128 * S1) / (3 * (1 + n) ** 4)
        - (128 * S1) / (9 * n**4 * (1 + n) ** 4)
        + (256 * S1) / (9 * (1 + n) ** 3)
        - (320 * S1) / (27 * n**3 * (1 + n) ** 3)
        + (256 * S1) / (9 * (1 + n) ** 2)
        + (1568 * S1) / (27 * n**2 * (1 + n) ** 2)
        - (224 * S1) / (9 * n * (1 + n))
        - (64 * (S1**2 + S2)) / (9 * n**3 * (1 + n) ** 3)
        - (128 * (S1**2 + S2)) / (9 * n**2 * (1 + n) ** 2)
        - (64 * Sm2) / (9 * n**3 * (1 + n) ** 3)
        - (128 * Sm2) / (9 * n**2 * (1 + n) ** 2)
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
        delta_B3: :math:`\\delta B^{(3)}`
    """
    # return gamma_nsm_nf2(n, sx) - CF * (CA - 2 * CF) * deltaB3(n, sx)
    S1 = sx[0][0]
    return (
        -193.83259645717885
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 278.2205375565073 / n
        + 537.132861181022 / (1.0 + n) ** 3
        - 817.9374228369205 / (1.0 + n) ** 2
        - 80.16230542465289 / (2.0 + n)
        + 195.5772257829161 * S1
        - (491.7139266455562 * S1) / n**2
        + (26.68861454046639 * S1) / n
        + (249.125506580144 * S1) / (1.0 + n) ** 3
        + (276.75480984972495 * S1) / (1.0 + n) ** 2
        - (3.24849037613728 * S1) / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_nsp_nf1(n, sx):
    """
    Implements the parametrized singlet-like non-singlet part proportional to :math:`nf^1`.

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
    return (
        5549.533222114542
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        - 11877.948615823714 / (1.0 + n) ** 3
        + 17141.75538179074 / (1.0 + n) ** 2
        + 2189.7561896037237 / (2.0 + n)
        - 5171.916129085788 * S1
        + (12198.267695106204 * S1) / n**2
        - (2741.830025124657 * S1) / n
        - (6658.5933037552495 * S1) / (1.0 + n) ** 3
        - (6980.106185472365 * S1) / (1.0 + n) ** 2
        + (73.57787513932745 * S1) / (1.0 + n)
    )


@nb.njit(cache=True)
def gamma_nsp_nf0(n, sx):
    """
    Implements the parametrized singlet-like non-singlet part proportional to :math:`nf^0`.

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
    return (
        -23389.366023525115
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        + 75848.9162996206 / (1.0 + n) ** 3
        - 21458.28316538394 / (1.0 + n) ** 2
        - 7874.846331131067 / (2.0 + n)
        + 20702.353028966703 * S1
        - (73014.16193348375 * S1) / n**2
        + (16950.937339235086 * S1) / n
        + (3275.0528283502285 * S1) / (1.0 + n) ** 3
        + (27872.8964453729 * S1) / (1.0 + n) ** 2
        - (501.0138189552833 * S1) / (1.0 + n)
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
