r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{gg}^{(3)}`."""

import numba as nb
import numpy as np

from eko.constants import zeta3

from ......harmonics import cache as c
from ......harmonics.log_functions import (
    lm11,
    lm11m1,
    lm11m2,
    lm12m1,
    lm12m2,
    lm13m1,
    lm13m2,
    lm14m1,
)


@nb.njit(cache=True)
def gamma_gg(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-gluon singlet anomalous dimension.

    The routine is taken from :cite:`Falcioni:2024qpd`.
    A previous version based only on the lowest 10 moments
    was given in :cite:`Moch:2023tdj`.

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
        |N3LO| gluon-gluon singlet anomalous dimension
        :math:`\gamma_{gg}^{(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)

    nf2 = nf * nf
    nf3 = nf * nf2

    # The known large-x coefficients [except delta(1-x)]
    A4gluon = 40880.330 - 11714.246 * nf + 440.04876 * nf2 + 7.3627750 * nf3
    B4gluon = 68587.64 - 18143.983 * nf + 423.81135 * nf2 + 9.0672154 * 0.1 * nf3

    # The coefficient of delta(1-x), also called the virtual anomalous
    # dimension. nf^0 and nf^1 are still approximate, but the error at
    # nf^1 is far too small to be relevant in this context.
    if variation == 1:
        B4gluon = B4gluon - 0.2
    elif variation == 2:
        B4gluon = B4gluon + 0.2

    Ccoeff = 8.5814120 * 10**4 - 1.3880515 * 10**4 * nf + 1.3511111 * 10**2 * nf2
    Dcoeff = 5.4482808 * 10**4 - 4.3411337 * 10**3 * nf - 2.1333333 * 10 * nf2

    x1L4cff = 5.6460905 * 10 * nf - 3.6213992 * nf2
    x1L3cff = 2.4755054 * 10**2 * nf - 4.0559671 * 10 * nf2 + 1.5802469 * nf3

    # The known coefficients of 1/x*ln^a x terms, a = 3,2
    bfkl0 = -8.3086173 * 10**3
    bfkl1 = -1.0691199 * 10**5 - 9.9638304 * 10**2 * nf

    x0L6cff = 1.44 * 10**2 - 2.7786008 * 10 * nf + 7.9012346 * 0.1 * nf2
    x0L5cff = -1.44 * 10**2 - 1.6208066 * 10**2 * nf + 1.4380247 * 10 * nf2
    x0L4cff = (
        2.6165784 * 10**4
        - 3.3447551 * 10**3 * nf
        + 9.1522635 * 10 * nf2
        - 1.9753086 * 0.1 * nf3
    )

    #  The resulting part of the function
    P3gg01 = (
        +bfkl0 * (-(6 / (-1 + n) ** 4))
        + bfkl1 * 2 / (-1 + n) ** 3
        + x0L6cff * 720 / n**7
        + x0L5cff * -120 / n**6
        + x0L4cff * 24 / n**5
        + A4gluon * (-S1)
        + B4gluon
        + Ccoeff * lm11(n, S1)
        + Dcoeff * 1 / n
        + x1L4cff * lm14m1(n, S1, S2, S3, S4)
        + x1L3cff * lm13m1(n, S1, S2, S3)
    )

    # The selected approximations for nf = 3, 4, 5
    if nf == 3:
        P3ggApp1 = (
            P3gg01
            - 421311.0 * (-(1 / (-1 + n) ** 2) + 1 / n**2)
            - 325557.0 * 1 / ((-1 + n) * n)
            + 1679790.0 * (1 / (n + n**2))
            - 1456863.0 * (1 / (2 + 3 * n + n**2))
            + 3246307.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 2026324.0 * 2 / n**3
            + 549188.0 * (-(6 / n**4))
            + 8337.0 * Lm11m1
            + 26718.0 * Lm12m1
            - 27049.0 * lm13m2(n, S1, S2, S3)
        )
        P3ggApp2 = (
            P3gg01
            - 700113.0 * (-(1 / (-1 + n) ** 2) + 1 / n**2)
            - 2300581.0 * 1 / ((-1 + n) * n)
            + 896407.0 * (1 / n - n / (2 + 3 * n + n**2))
            - 162733.0 * (1 / (6 + 5 * n + n**2))
            - 2661862.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 196759.0 * 2 / n**3
            - 260607.0 * (-(6 / n**4))
            + 84068.0 * Lm11m1
            + 346318.0 * Lm12m1
            + 315725.0
            * (
                -3 * S1**2
                + n * S1 * (np.pi**2 - 6 * S2)
                - 3 * (S2 + 2 * n * (S3 - zeta3))
            )
            / (3 * n**2)
        )
    elif nf == 4:
        P3ggApp1 = (
            P3gg01
            - 437084.0 * (-(1 / (-1 + n) ** 2) + 1 / n**2)
            - 361570.0 * 1 / ((-1 + n) * n)
            + 1696070.0 * (1 / (n + n**2))
            - 1457385.0 * (1 / (2 + 3 * n + n**2))
            + 3195104.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 2009021.0 * 2 / n**3
            + 544380.0 * (-(6 / n**4))
            + 9938.0 * Lm11m1
            + 24376.0 * Lm12m1
            - 22143.0 * lm13m2(n, S1, S2, S3)
        )
        P3ggApp2 = (
            P3gg01
            - 706649.0 * (-(1 / (-1 + n) ** 2) + 1 / n**2)
            - 2274637.0 * 1 / ((-1 + n) * n)
            + 836544.0 * (1 / n - n / (2 + 3 * n + n**2))
            - 199929.0 * (1 / (6 + 5 * n + n**2))
            - 2683760.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 168802.0 * 2 / n**3
            - 250799.0 * (-(6 / n**4))
            + 36967.0 * Lm11m1
            + 24530.0 * Lm12m1
            - 71470.0 * lm12m2(n, S1, S2)
        )
    elif nf == 5:
        P3ggApp1 = (
            P3gg01
            - 439426.0 * (-(1 / (-1 + n) ** 2) + 1 / n**2)
            - 293679.0 * 1 / ((-1 + n) * n)
            + 1916281.0 * (1 / (n + n**2))
            - 1615883.0 * (1 / (2 + 3 * n + n**2))
            + 3648786.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 2166231.0 * 2 / n**3
            + 594588.0 * (-(6 / n**4))
            + 50406.0 * Lm11m1
            + 24692.0 * Lm12m1
            + 174067.0 * lm11m2(n, S1)
        )
        P3ggApp2 = (
            P3gg01
            - 705978.0 * (-(1 / (-1 + n) ** 2) + 1 / n**2)
            - 2192234.0 * 1 / ((-1 + n) * n)
            + 1730508.0 * (1 / (2 + 3 * n + n**2))
            + 353143.0 * ((12 + 9 * n + n**2) / (6 * n + 11 * n**2 + 6 * n**3 + n**4))
            - 2602682.0 * (-(1 / n**2) + 1 / (1 + n) ** 2)
            + 178960.0 * 2 / n**3
            - 218133.0 * (-(6 / n**4))
            + 2285.0 * Lm11m1
            + 19295.0 * Lm12m1
            - 13719.0 * lm12m2(n, S1, S2)
        )
    else:
        raise NotImplementedError("nf=6 is not available at N3LO")

    # We return (for now) one of the two error-band representatives
    # or the present best estimate, their average
    if variation == 1:
        P3GGA = P3ggApp1
    elif variation == 2:
        P3GGA = P3ggApp2
    else:
        P3GGA = 0.5 * (P3ggApp1 + P3ggApp2)

    return -P3GGA
