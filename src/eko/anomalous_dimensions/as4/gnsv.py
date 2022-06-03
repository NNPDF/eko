# -*- coding: utf-8 -*-
"""This module contains the anomalous dimension :math:`\\gamma_{ns,v}^{(3)}`

"""
import numba as nb

from .gnsm import gamma_nsm


@nb.njit(cache=True)
def gamma_nss_nf2(n, sx):
    """Implements the sea non-singlet part proportional to :math:`nf^2`
    as in Eq. 3.5 of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nss_nf2 : complex
        |N3LO| sea non-singlet anomalous dimension :math:`\\gamma_{ns,s}^{(3)}|_{nf^2}`

    """
    S1, _ = sx[0]
    S2, Sm2 = sx[1]
    S3, _, _, Sm21, _, Sm3 = sx[2]
    S4, S31, _, Sm22, Sm211, Sm31, Sm4 = sx[3]
    return (
        160
        / 27
        * (
            40 / (1 + n) ** 6
            - 8 / (n**6 * (1 + n) ** 6)
            - 16 / (1 + n) ** 5
            - 24 / (n**5 * (1 + n) ** 5)
            + 108 / (1 + n) ** 4
            + 36 / (n**4 * (1 + n) ** 4)
            + 112 / (1 + n) ** 3
            + 168 / (n**3 * (1 + n) ** 3)
            + 144 / (1 + n) ** 2
            + 174 / (n**2 * (1 + n) ** 2)
            - 64 / (n * (1 + n))
            - 32 / (3 * (2 + n) ** 2)
            - 208 / (3 * (-1 + n) * (2 + n))
            + (
                48 / (1 + n) ** 5
                + 4 / (n**5 * (1 + n) ** 5)
                + 6 / (1 + n) ** 4
                - 24 / (n**4 * (1 + n) ** 4)
                + 76 / (1 + n) ** 3
                - 162 / (n**3 * (1 + n) ** 3)
                + 50 / (1 + n) ** 2
                - 164 / (n**2 * (1 + n) ** 2)
                - 285 / (2 * n * (1 + n))
                + 32 / (3 * (2 + n) ** 2)
                + 304 / (3 * (-1 + n) * (2 + n))
            )
            * S1
            - (
                -(6 / (n**4 * (1 + n) ** 4))
                - 23 / (n**3 * (1 + n) ** 3)
                - 28 / (n**2 * (1 + n) ** 2)
                - 13 / (n * (1 + n))
                + 16 / ((-1 + n) * (2 + n))
            )
            * (-S2 + 2 * (S1**2 + S2))
            + (
                -(8 / (1 + n) ** 3)
                - 6 / (n**3 * (1 + n) ** 3)
                - 6 / (1 + n) ** 2
                - 7 / (n**2 * (1 + n) ** 2)
                - 9 / (n * (1 + n))
                + 16 / ((-1 + n) * (2 + n))
            )
            * S3
            + (
                -(12 / (n**2 * (1 + n) ** 2))
                - 22 / (n * (1 + n))
                + 32 / ((-1 + n) * (2 + n))
            )
            * S4
            + (
                24 / (1 + n) ** 4
                - 8 / (1 + n) ** 3
                + 28 / (n**3 * (1 + n) ** 3)
                + 28 / (1 + n) ** 2
                + 104 / (n**2 * (1 + n) ** 2)
                + 91 / (n * (1 + n))
                - 32 / (3 * (2 + n) ** 2)
                - 304 / (3 * (-1 + n) * (2 + n))
            )
            * Sm2
            + (
                -(8 / (n**3 * (1 + n) ** 3))
                - 16 / (1 + n) ** 2
                - 28 / (n**2 * (1 + n) ** 2)
                - 76 / (n * (1 + n))
                + 32 / (3 * (2 + n) ** 2)
                + 352 / (3 * (-1 + n) * (2 + n))
            )
            * Sm21
            - (-(8 / (n * (1 + n))) + 16 / ((-1 + n) * (2 + n)))
            * (S4 + Sm2**2 + 4 * Sm211 - Sm22)
            + (
                -(16 / (1 + n) ** 3)
                - 4 / (n**3 * (1 + n) ** 3)
                + 4 / (1 + n) ** 2
                + 14 / (n**2 * (1 + n) ** 2)
                + 58 / (n * (1 + n))
                - 32 / (3 * (2 + n) ** 2)
                - 256 / (3 * (-1 + n) * (2 + n))
            )
            * Sm3
            + (
                32 / (1 + n) ** 3
                + 8 / (1 + n) ** 2
                - 56 / (n**2 * (1 + n) ** 2)
                - 112 / (n * (1 + n))
                + 32 / (3 * (2 + n) ** 2)
                + 352 / (3 * (-1 + n) * (2 + n))
            )
            * (S1 * Sm2 - Sm21 + Sm3)
            + (
                -(4 / (n**2 * (1 + n) ** 2))
                - 10 / (n * (1 + n))
                + 16 / ((-1 + n) * (2 + n))
            )
            * (
                -S1 * S3
                + S31
                - S4
                + 2 * Sm31
                - 4 * (S1 * Sm21 - 2 * Sm211 + Sm22 + Sm31)
                + Sm4
                + 2 * (S1 * Sm3 - Sm31 + Sm4)
            )
            - (
                -(8 / (n**2 * (1 + n) ** 2))
                - 12 / (n * (1 + n))
                + 16 / ((-1 + n) * (2 + n))
            )
            * (
                S31
                - S2 * Sm2
                + Sm22
                - Sm4
                + 4
                * (
                    S2 * Sm2
                    - 1 / 2 * (S1**2 + S2) * Sm2
                    + Sm211
                    - Sm22
                    + S1 * (S1 * Sm2 - Sm21 + Sm3)
                    - Sm31
                    + Sm4
                )
            )
        )
    )


@nb.njit(cache=True)
def gamma_nss_nf1(n, sx):
    """Implements the sea non-singlet part proportional to :math:`nf^1`.
    The expression is the average of the Mellin transform
    of Eq. 4.19, 4.20 of :cite:`Moch:2017uml`

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nss_nf1 : complex
        |N3LO| sea non-singlet anomalous dimension :math:`\\gamma_{ns,s}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    return (
        3803.04 / n**6
        - 3560.16 / n**5
        + 1042.89 / n**4
        - 3569.65 / n**3
        + 3337.715 / n**2
        - 1266.77 / n
        - 13.18641 / (1 + n) ** 4
        + 60.4 / (1 + n) ** 3
        - 1743.225 / (1 + n)
        + 127.315 / (1.0 + n) ** 2
        + 4447.245 / (2 + n)
        - 1437.25 / (3 + n)
        - (127.315 * S1) / n
        - (13.18641 * S1) / (1 + n) ** 3
        + (60.4 * S1) / (1 + n) ** 2
        + (127.315 * S1) / (1.0 + n) ** 2
        + (127.315 * n * S1) / (1.0 + n) ** 2
        - (30.2 * S1**2) / n
        - (6.5932 * S1**2) / (1 + n) ** 2
        + (30.2 * S1**2) / (1 + n)
        + (2.197735 * S1**3) / n
        - (2.197735 * S1**3) / (1 + n)
        - (30.2 * S2) / n
        - (6.593205 * S2) / (1 + n) ** 2
        + (30.2 * S2) / (1 + n)
        + (6.593205 * S1 * S2) / n
        - (6.593205 * S1 * S2) / (1 + n)
        + (4.39547 * S3) / n
        - (4.39547 * S3) / (1 + n)
    )


@nb.njit(cache=True)
def gamma_nsv(n, nf, sx):
    """Computes the |N3LO| valence non-singlet anomalous dimension.

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
        gamma_nsv : complex
            |N3LO| valence non-singlet anomalous dimension
            :math:`\\gamma_{ns,v}^{(3)}(N)`

    See Also
    --------
    gamma_nsm: :math:`\\gamma_{ns,-}^{(3)}`
    gamma_nss_nf1: :math:`\\gamma_{ns,s}^{(3)}|_{nf^1}`
    gamma_nss_nf2: :math:`\\gamma_{ns,s}^{(3)}|_{nf^2}`

    """
    return (
        gamma_nsm(n, nf, sx)
        + nf * gamma_nss_nf1(n, sx)
        + nf**2 * gamma_nss_nf2(n, sx)
    )
