r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{gq}^{(3)}`."""

import numba as nb

from eko.constants import zeta2

from ......harmonics import cache as c
from ......harmonics.log_functions import (
    lm11,
    lm12,
    lm13,
    lm14,
    lm14m1,
    lm15,
    lm15m1,
)


@nb.njit(cache=True)
def gamma_gq(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

    The routine is taken from :cite:`Falcioni:2025hfz`.
    A previous version was given in :cite:`Falcioni:2024xyt`.
    While a version based only on the lowest 10 moments
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
        |N3LO| gluon-quark singlet anomalous dimension
        :math:`\gamma_{gq}^{(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    S5 = c.get(c.S5, cache, n)
    nf2 = nf * nf

    # Known large-x coefficients
    x1L5cff = 1.3443073 * 10 - 5.4869684 * 0.1 * nf
    x1L4cff = 3.7539831 * 10**2 - 3.4494742 * 10 * nf + 8.7791495 * 0.1 * nf2
    y1L5cff = 2.2222222 * 10 - 5.4869684 * 0.1 * nf
    y1L4cff = 6.6242163 * 10**2 - 4.7992684 * 10 * nf + 8.7791495 * 0.1 * nf2

    # Small-x, Casimir scaled from P_gg (approx. for bfkl1)
    bfkl0 = -8.3086173 * 10**3 / 2.25
    bfkl1 = (-1.0691199 * 10**5 - nf * 9.9638304 * 10**2) / 2.25

    # Small-x double-logs with x^0
    x0L6cff = 5.2235940 * 10 - 7.3744856 * nf
    x0L5cff = -2.9221399 * 10**2 + 1.8436214 * nf
    x0L4cff = 7.3106077 * 10**3 - 3.7887135 * 10**2 * nf - 3.2438957 * 10 * nf2

    # The resulting part of the function
    P3GQ01 = (
        +bfkl0 * (-(6 / (-1 + n) ** 4))
        + bfkl1 * 2 / (-1 + n) ** 3
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
        P3gqApp1 = (
            P3GQ01
            + 3.5 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 27891.0 * 1 / ((-1 + n) * n)
            - 309124.0 * 1 / n
            + 1056866.0 * (3 + n) / (2 + 3 * n + n**2)
            - 124735.0 * -1 / n**2
            - 16246.0 * 2 / n**3
            + 131175.0 * -6 / n**4
            + 4970.1 * lm13(n, S1, S2, S3)
            + 60041.0 * lm12(n, S1, S2)
            + 343181.0 * lm11(n, S1)
            - 958330.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
        P3gqApp2 = (
            P3GQ01
            + 7.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 1139334.0 * 1 / ((-1 + n) * n)
            + 143008.0 * 1 / n
            - 290390.0 * (3 + n) / (2 + 3 * n + n**2)
            - 659492.0 * -1 / n**2
            + 303685.0 * 2 / n**3
            - 81867.0 * -6 / n**4
            + 1811.8 * lm13(n, S1, S2, S3)
            - 465.9 * lm12(n, S1, S2)
            - 51206.0 * lm11(n, S1)
            + 274249.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    elif nf == 4:
        P3gqApp1 = (
            P3GQ01
            + 3.5 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 8302.8 * 1 / ((-1 + n) * n)
            - 347706.0 * 1 / n
            + 1105306.0 * (3 + n) / (2 + 3 * n + n**2)
            - 127650.0 * -1 / n**2
            - 29728.0 * 2 / n**3
            + 137537.0 * -6 / n**4
            + 4658.1 * lm13(n, S1, S2, S3)
            + 59205.0 * lm12(n, S1, S2)
            + 345513.0 * lm11(n, S1)
            - 995120.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
        P3gqApp2 = (
            P3GQ01
            + 7.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 1129822.0 * 1 / ((-1 + n) * n)
            + 108527.0 * 1 / n
            - 254166.0 * (3 + n) / (2 + 3 * n + n**2)
            - 667254.0 * -1 / n**2
            + 293099.0 * 2 / n**3
            - 77437.0 * -6 / n**4
            + 1471.3 * lm13(n, S1, S2, S3)
            - 1850.3 * lm12(n, S1, S2)
            - 52451.0 * lm11(n, S1)
            + 248634.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    elif nf == 5:
        P3gqApp1 = (
            P3GQ01
            + 3.5 * bfkl1 * (-(1 / (-1 + n) ** 2))
            + 14035.0 * 1 / ((-1 + n) * n)
            - 384003.0 * 1 / n
            + 1152711.0 * (3 + n) / (2 + 3 * n + n**2)
            - 126346.0 * -1 / n**2
            - 42967.0 * 2 / n**3
            + 144270.0 * -6 / n**4
            + 4385.5 * lm13(n, S1, S2, S3)
            + 58688.0 * lm12(n, S1, S2)
            + 348988.0 * lm11(n, S1)
            - 1031165.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
        P3gqApp2 = (
            P3GQ01
            + 7.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 1117561.0 * 1 / ((-1 + n) * n)
            + 76329.0 * 1 / n
            - 218973.0 * (3 + n) / (2 + 3 * n + n**2)
            - 670799.0 * -1 / n**2
            + 282763.0 * 2 / n**3
            - 72633.0 * -6 / n**4
            + 1170.0 * lm13(n, S1, S2, S3)
            - 2915.5 * lm12(n, S1, S2)
            - 52548.0 * lm11(n, S1)
            + 223771.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    elif nf == 6:
        P3gqApp1 = (
            P3GQ01
            + 3.5 * bfkl1 * (-(1 / (-1 + n) ** 2))
            + 39203.0 * 1 / ((-1 + n) * n)
            - 417914.0 * 1 / n
            + 1199042.0 * (3 + n) / (2 + 3 * n + n**2)
            - 120750.0 * -1 / n**2
            - 55941.0 * 2 / n**3
            + 151383.0 * -6 / n**4
            + 4149.2 * lm13(n, S1, S2, S3)
            + 58466.0 * lm12(n, S1, S2)
            + 353589.0 * lm11(n, S1)
            - 1066510.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
        P3gqApp2 = (
            P3GQ01
            + 7.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 1102470.0 * 1 / ((-1 + n) * n)
            + 46517.0 * 1 / n
            - 184858.0 * (3 + n) / (2 + 3 * n + n**2)
            - 670056.0 * -1 / n**2
            + 272689.0 * 2 / n**3
            - 67453.0 * -6 / n**4
            + 905.0 * lm13(n, S1, S2, S3)
            - 3686.2 * lm12(n, S1, S2)
            - 51523.0 * lm11(n, S1)
            + 199594.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    else:
        raise NotImplementedError("Select nf=3,..,6 for N3LO evolution")

    # We return (for now) one of the two error-band representatives
    # or the present best estimate, their average
    if variation == 1:
        P3GQA = P3gqApp1
    elif variation == 2:
        P3GQA = P3gqApp2
    else:
        P3GQA = 0.5 * (P3gqApp1 + P3gqApp2)
    return -P3GQA
