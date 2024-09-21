r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{qg}^{(3)}`."""

import numba as nb
import numpy as np

from ......harmonics import cache as c
from ......harmonics.log_functions import lm11, lm12, lm13, lm14, lm14m1, lm15, lm15m1


@nb.njit(cache=True)
def gamma_qg(n, nf, cache, variation):
    r"""Compute the |N3LO| quark-gluon singlet anomalous dimension.

    The routine is taken from :cite:`Falcioni:2023vqq`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| quark-gluon singlet anomalous dimension
        :math:`\gamma_{qg}^{(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    S5 = c.get(c.S5, cache, n)
    nf2 = nf * nf
    nf3 = nf * nf2

    # Known large-x coefficients
    x1L5cff = 1.8518519 * nf - 4.1152263 * 0.1 * nf2
    x1L4cff = 3.5687794 * 10 * nf - 3.5116598 * nf2 - 8.2304527 * 0.01 * nf3
    y1L5cff = 2.8806584 * nf + 8.2304527 * 0.1 * nf2
    y1L4cff = -4.0511391 * 10 * nf + 5.5418381 * nf2 + 1.6460905 * 0.1 * nf3

    # Known small-x coefficients
    bfkl1 = 3.9357613 * 10**3 * nf
    x0L6cff = -1.9588477 * 10 * nf + 2.7654321 * nf2
    x0L5cff = 2.1573663 * 10 * nf + 1.7244444 * 10 * nf2
    x0L4cff = -2.8667643 * 10**3 * nf + 3.0122403 * 10**2 * nf2 + 4.1316872 * nf3

    # The resulting part of the function
    P3QG01 = (
        +bfkl1 * 2 / (-1 + n) ** 3
        + x0L6cff * 720 / n**7
        + x0L5cff * -120 / n**6
        + x0L4cff * 24 / n**5
        + x1L4cff * lm14(n, S1, S2, S3, S4)
        + x1L5cff * lm15(n, S1, S2, S3, S4, S5)
        + y1L4cff * lm14m1(n, S1, S2, S3, S4)
        + y1L5cff * lm15m1(n, S1, S2, S3, S4, S5)
    )

    # The selected approximations for nf = 3, 4, 5
    if nf == 3:
        P3qgApp1 = (
            P3QG01
            + 187500.0 * -(1 / (-1 + n) ** 2)
            + 826060.0 * 1 / ((-1 + n) * n)
            - 150474.0 * 1 / n
            + 226254.0 * (3 + n) / (2 + 3 * n + n**2)
            + 577733.0 * -1 / n**2
            - 180747.0 * 2 / n**3
            + 95411.0 * -6 / n**4
            + 119.8 * lm13(n, S1, S2, S3)
            + 7156.3 * lm12(n, S1, S2)
            + 45790.0 * lm11(n, S1)
            - 95682.0 * (S1 - n * (np.pi**2 / 6 - S2)) / n**2
        )
        P3qgApp2 = (
            P3QG01
            + 135000.0 * -(1 / (-1 + n) ** 2)
            + 484742.0 * 1 / ((-1 + n) * n)
            - 11627.0 * 1 / n
            - 187478.0 * (3 + n) / (2 + 3 * n + n**2)
            + 413512.0 * -1 / n**2
            - 82500.0 * 2 / n**3
            + 29987.0 * -6 / n**4
            - 850.1 * lm13(n, S1, S2, S3)
            - 11425.0 * lm12(n, S1, S2)
            - 75323.0 * lm11(n, S1)
            + 282836.0 * (S1 - n * (np.pi**2 / 6 - S2)) / n**2
        )
    elif nf == 4:
        P3qgApp1 = (
            P3QG01
            + 250000.0 * -(1 / (-1 + n) ** 2)
            + 1089180.0 * 1 / ((-1 + n) * n)
            - 241088.0 * 1 / n
            + 342902.0 * (3 + n) / (2 + 3 * n + n**2)
            + 720081.0 * -1 / n**2
            - 247071.0 * 2 / n**3
            + 126405.0 * -6 / n**4
            + 272.4 * lm13(n, S1, S2, S3)
            + 10911.0 * lm12(n, S1, S2)
            + 60563.0 * lm11(n, S1)
            - 161448.0 * (S1 - n * (np.pi**2 / 6 - S2)) / n**2
        )
        P3qgApp2 = (
            P3QG01
            + 180000.0 * -(1 / (-1 + n) ** 2)
            + 634090.0 * 1 / ((-1 + n) * n)
            - 55958.0 * 1 / n
            - 208744.0 * (3 + n) / (2 + 3 * n + n**2)
            + 501120.0 * -1 / n**2
            - 116073.0 * 2 / n**3
            + 39173.0 * -6 / n**4
            - 1020.8 * lm13(n, S1, S2, S3)
            - 13864.0 * lm12(n, S1, S2)
            - 100922.0 * lm11(n, S1)
            + 343243.0 * (S1 - n * (np.pi**2 / 6 - S2)) / n**2
        )
    elif nf == 5:
        P3qgApp1 = (
            P3QG01
            + 312500.0 * -(1 / (-1 + n) ** 2)
            + 1345700.0 * 1 / ((-1 + n) * n)
            - 350466.0 * 1 / n
            + 480028.0 * (3 + n) / (2 + 3 * n + n**2)
            + 837903.0 * -1 / n**2
            - 315928.0 * 2 / n**3
            + 157086.0 * -6 / n**4
            + 472.7 * lm13(n, S1, S2, S3)
            + 15415.0 * lm12(n, S1, S2)
            + 75644.0 * lm11(n, S1)
            - 244869.0 * (S1 - n * (np.pi**2 / 6 - S2)) / n**2
        )
        P3qgApp2 = (
            P3QG01
            + 225000.0 * -(1 / (-1 + n) ** 2)
            + 776837.0 * 1 / ((-1 + n) * n)
            - 119054.0 * 1 / n
            - 209530.0 * (3 + n) / (2 + 3 * n + n**2)
            + 564202.0 * -1 / n**2
            - 152181.0 * 2 / n**3
            + 48046.0 * -6 / n**4
            - 1143.8 * lm13(n, S1, S2, S3)
            - 15553.0 * lm12(n, S1, S2)
            - 126212.0 * lm11(n, S1)
            + 385995.0 * (S1 - n * (np.pi**2 / 6 - S2)) / n**2
        )
    else:
        raise NotImplementedError("nf=6 is not available at N3LO")

    # We return one of the two error-band representatives
    # or the present best estimate, their average
    if variation == 1:
        P3QGA = P3qgApp1
    elif variation == 2:
        P3QGA = P3qgApp2
    else:
        P3QGA = 0.5 * (P3qgApp1 + P3qgApp2)
    return -P3QGA
