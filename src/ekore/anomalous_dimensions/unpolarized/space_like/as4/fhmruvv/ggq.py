r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{gq}^{(3)}`."""

import numba as nb

from eko.constants import zeta2

from ......harmonics import cache as c
from ......harmonics.log_functions import (
    lm11,
    lm12,
    lm12m1,
    lm13,
    lm14,
    lm14m1,
    lm15,
    lm15m1,
)


@nb.njit(cache=True)
def gamma_gq(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

    The routine is taken from :cite:`Falcioni:2024xyt`.
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
            + 6.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 744384.0 * 1 / ((-1 + n) * n)
            + 2453640.0 * 1 / n
            - 1540404.0 * (2 / (1 + n) + 1 / (2 + n))
            + 1933026.0 * -1 / n**2
            + 1142069.0 * 2 / n**3
            + 162196.0 * -6 / n**4
            - 2172.1 * lm13(n, S1, S2, S3)
            - 93264.1 * lm12(n, S1, S2)
            - 786973.0 * lm11(n, S1)
            + 875383.0 * lm12m1(n, S1, S2)
        )
        P3gqApp2 = (
            P3GQ01
            + 3.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            + 142414.0 * 1 / ((-1 + n) * n)
            - 326525.0 * 1 / n
            + 2159787.0 * ((3 + n) / (2 + 3 * n + n**2))
            - 289064.0 * -1 / n**2
            - 176358.0 * 2 / n**3
            + 156541.0 * -6 / n**4
            + 9016.5 * lm13(n, S1, S2, S3)
            + 136063.0 * lm12(n, S1, S2)
            + 829482.0 * lm11(n, S1)
            - 2359050.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    elif nf == 4:
        P3gqApp1 = (
            P3GQ01
            + 6.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 743535.0 * 1 / ((-1 + n) * n)
            + 2125286.0 * 1 / n
            - 1332472.0 * (2 / (1 + n) + 1 / (2 + n))
            + 1631173.0 * -1 / n**2
            + 1015255.0 * 2 / n**3
            + 142612.0 * -6 / n**4
            - 1910.4 * lm13(n, S1, S2, S3)
            - 80851.0 * lm12(n, S1, S2)
            - 680219.0 * lm11(n, S1)
            + 752733.0 * lm12m1(n, S1, S2)
        )
        P3gqApp2 = (
            P3GQ01
            + 3.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            + 160568.0 * 1 / ((-1 + n) * n)
            - 361207.0 * 1 / n
            + 2048948.0 * ((3 + n) / (2 + 3 * n + n**2))
            - 245963.0 * -1 / n**2
            - 171312.0 * 2 / n**3
            + 163099.0 * -6 / n**4
            + 8132.2 * lm13(n, S1, S2, S3)
            + 124425.0 * lm12(n, S1, S2)
            + 762435.0 * lm11(n, S1)
            - 2193335.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    elif nf == 5:
        P3gqApp1 = (
            P3GQ01
            + 6.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 785864.0 * 1 / ((-1 + n) * n)
            + 285034.0 * 1 / n
            - 131648.0 * (2 / (1 + n) + 1 / (2 + n))
            - 162840.0 * -1 / n**2
            + 321220.0 * 2 / n**3
            + 12688.0 * -6 / n**4
            + 1423.4 * lm13(n, S1, S2, S3)
            + 1278.9 * lm12(n, S1, S2)
            - 30919.9 * lm11(n, S1)
            + 47588.0 * lm12m1(n, S1, S2)
        )
        P3gqApp2 = (
            P3GQ01
            + 3.0 * bfkl1 * (-(1 / (-1 + n) ** 2))
            + 177094.0 * 1 / ((-1 + n) * n)
            - 470694.0 * 1 / n
            + 1348823.0 * ((3 + n) / (2 + 3 * n + n**2))
            - 52985.0 * -1 / n**2
            - 87354.0 * 2 / n**3
            + 176885.0 * -6 / n**4
            + 4748.8 * lm13(n, S1, S2, S3)
            + 65811.9 * lm12(n, S1, S2)
            + 396390.0 * lm11(n, S1)
            - 1190212.0 * (S1 - n * (zeta2 - S2)) / n**2
        )
    else:
        raise NotImplementedError("nf=6 is not available at N3LO")

    # We return (for now) one of the two error-band representatives
    # or the present best estimate, their average
    if variation == 1:
        P3GQA = P3gqApp1
    elif variation == 2:
        P3GQA = P3gqApp2
    else:
        P3GQA = 0.5 * (P3gqApp1 + P3gqApp2)
    return -P3GQA
