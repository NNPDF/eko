r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{ps}^{(3)}`."""

import numba as nb

from ......harmonics import cache as c
from ......harmonics.log_functions import (
    lm11m1,
    lm12m1,
    lm12m2,
    lm13m1,
    lm13m2,
    lm14m1,
    lm14m2,
)


@nb.njit(cache=True)
def gamma_ps(n, nf, cache, variation):
    r"""Compute the |N3LO| pure singlet quark-quark anomalous dimension.

    The routine is taken from :cite:`Falcioni:2023luc`.

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
        |N3LO| pure singlet quark-quark anomalous dimension
        :math:`\gamma_{ps}^{(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    nf2 = nf * nf
    nf3 = nf * nf2
    xm1lm1 = -(1 / (-1 + n) ** 2) + 1 / n**2

    # Known large-x coefficients
    x1L4cff = -5.6460905 * 10 * nf + 3.6213992 * nf2
    x1L3cff = -2.4755054 * 10**2 * nf + 4.0559671 * 10 * nf2 - 1.5802469 * nf3
    y1L4cff = -1.3168724 * 10 * nf
    y1L3cff = -1.9911111 * 10**2 * nf + 1.3695473 * 10 * nf2

    # Known small-x coefficients
    bfkl1 = 1.7492273 * 10**3 * nf
    x0L6cff = -7.5061728 * nf + 7.9012346 * 0.1 * nf2
    x0L5cff = 2.8549794 * 10 * nf + 3.7925926 * nf2
    x0L4cff = -8.5480010 * 10**2 * nf + 7.7366255 * 10 * nf2 - 1.9753086 * 0.1 * nf3

    # The resulting part of the function
    P3ps01 = (
        +bfkl1 * 2 / (-1 + n) ** 3
        + x0L6cff * 720 / n**7
        + x0L5cff * -120 / n**6
        + x0L4cff * 24 / n**5
        + x1L3cff * lm13m1(n, S1, S2, S3)
        + x1L4cff * lm14m1(n, S1, S2, S3, S4)
        + y1L3cff * lm13m2(n, S1, S2, S3)
        + y1L4cff * lm14m2(n, S1, S2, S3, S4)
    )

    # The selected approximations for nf = 3, 4, 5
    if nf == 3:
        P3psApp1 = (
            P3ps01
            + 67731.0 * xm1lm1
            + 274100.0 * 1 / ((-1 + n) * n)
            - 104493.0 * (1 / n - n / (2 + 3 * n + n**2))
            + 34403.0 * 1 / (6 + 5 * n + n**2)
            + 353656.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 10620.0 * 2 / n**3
            + 40006.0 * -6 / n**4
            - 7412.1 * lm11m1(n, S1)
            - 2365.1 * lm12m1(n, S1, S2)
            + 1533.0 * lm12m2(n, S1, S2)
        )
        P3psApp2 = (
            P3ps01
            + 54593.0 * xm1lm1
            + 179748.0 * 1 / ((-1 + n) * n)
            - 195263.0 * 1 / (n + n**2)
            + 12789.0 * 2 / (3 + 4 * n + n**2)
            + 4700.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            - 103604.0 * 2 / n**3
            - 2758.3 * -6 / n**4
            - 2801.2 * lm11m1(n, S1)
            - 1986.9 * lm12m1(n, S1, S2)
            - 6005.9 * lm12m2(n, S1, S2)
        )
    elif nf == 4:
        P3psApp1 = (
            P3ps01
            + 90154.0 * xm1lm1
            + 359084.0 * 1 / ((-1 + n) * n)
            - 136319.0 * (1 / n - n / (2 + 3 * n + n**2))
            + 45379.0 * 1 / (6 + 5 * n + n**2)
            + 461167.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 13869.0 * 2 / n**3
            + 52525.0 * -6 / n**4
            - 7498.2 * lm11m1(n, S1)
            - 2491.5 * lm12m1(n, S1, S2)
            + 1727.2 * lm12m2(n, S1, S2)
        )
        P3psApp2 = (
            P3ps01
            + 72987.0 * xm1lm1
            + 235802.0 * 1 / ((-1 + n) * n)
            - 254921.0 * 1 / (n + n**2)
            + 17138.0 * 2 / (3 + 4 * n + n**2)
            + 5212.9 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            - 135378.0 * 2 / n**3
            - 3350.9 * -6 / n**4
            - 1472.7 * lm11m1(n, S1)
            - 1997.2 * lm12m1(n, S1, S2)
            - 8123.3 * lm12m2(n, S1, S2)
        )
    elif nf == 5:
        P3psApp1 = (
            P3ps01
            + 112481.0 * xm1lm1
            + 440555.0 * 1 / ((-1 + n) * n)
            - 166581.0 * (1 / n - n / (2 + 3 * n + n**2))
            + 56087.0 * 1 / (6 + 5 * n + n**2)
            + 562992.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 16882.0 * 2 / n**3
            + 64577.0 * -6 / n**4
            - 6570.1 * lm11m1(n, S1)
            - 2365.7 * lm12m1(n, S1, S2)
            + 1761.7 * lm12m2(n, S1, S2)
        )
        P3psApp2 = (
            P3ps01
            + 91468.0 * xm1lm1
            + 289658.0 * 1 / ((-1 + n) * n)
            - 311749.0 * 1 / (n + n**2)
            + 21521.0 * 2 / (3 + 4 * n + n**2)
            + 4908.9 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            - 165795.0 * 2 / n**3
            - 3814.9 * -6 / n**4
            + 804.5 * lm11m1(n, S1)
            - 1760.8 * lm12m1(n, S1, S2)
            - 10295.0 * lm12m2(n, S1, S2)
        )
    else:
        raise NotImplementedError("nf=6 is not available at N3LO")

    # We return (for now) one of the two error-band boundaries
    # or the present best estimate, their average
    if variation == 1:
        P3psA = P3psApp1
    elif variation == 2:
        P3psA = P3psApp2
    else:
        P3psA = 0.5 * (P3psApp1 + P3psApp2)
    return -P3psA
