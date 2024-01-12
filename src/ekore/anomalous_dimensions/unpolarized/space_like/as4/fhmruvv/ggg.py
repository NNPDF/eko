r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{gg}^{(3)}`."""
import numba as nb

from ......harmonics import cache as c
from ......harmonics.log_functions import lm11, lm12m1, lm13m1


@nb.njit(cache=True)
def gamma_gg(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-gluon singlet anomalous dimension.

    The routine is taken from :cite:`Moch:2023tdj`.

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

    nf2 = nf * nf
    nf3 = nf * nf2

    # The known large-x coefficients [except delta(1-x)]
    A4gluon = 40880.330 - 11714.246 * nf + 440.04876 * nf2 + 7.3627750 * nf3
    B4gluon = 68587.64 - 18143.983 * nf + 423.81135 * nf2 + 9.0672154 * 0.1 * nf3

    Ccoeff = 8.5814120 * 10**4 - 1.3880515 * 10**4 * nf + 1.3511111 * 10**2 * nf2
    Dcoeff = 5.4482808 * 10**4 - 4.3411337 * 10**3 * nf - 2.1333333 * 10 * nf2

    # The known coefficients of 1/x*ln^a x terms, a = 3,2
    bfkl0 = -8.308617314 * 10**3
    bfkl1 = -1.069119905 * 10**5 - 9.963830436 * 10**2 * nf

    #  The resulting part of the function
    P3gg01 = (
        +bfkl0 * (-(6 / (-1 + n) ** 4))
        + bfkl1 * 2 / (-1 + n) ** 3
        + A4gluon * (-S1)
        + B4gluon
        + Ccoeff * lm11(n, S1)
        + Dcoeff * 1 / n
    )

    # The selected approximations for nf = 3, 4, 5
    if nf == 3:
        P3ggApp1 = (
            P3gg01
            + 3.4 * bfkl1 * -(1 / (-1 + n) ** 2)
            - 345063.0 * 1 / ((-1 + n) * n)
            + 86650.0 * (1 / n - 1 / (1 + n) + 1 / (2 + n) - 1 / (3 + n))
            + 158160.0 * (-(1 / n**2))
            - 15741.0 * lm12m1(n, S1, S2)
            - 9417.0 * lm13m1(n, S1, S2, S3)
        )
        P3ggApp2 = (
            P3gg01
            + 5.4 * bfkl1 * -(1 / (-1 + n) ** 2)
            - 1265632.0 * 1 / ((-1 + n) * n)
            - 656644.0 * (1 / n - 1 / (1 + n) + 1 / (2 + n) - 1 / (3 + n))
            - 1352233.0 * (-(1 / n**2))
            + 203298.0 * lm12m1(n, S1, S2)
            + 39112.0 * lm13m1(n, S1, S2, S3)
        )
    elif nf == 4:
        P3ggApp1 = (
            P3gg01
            + 3.4 * bfkl1 * -(1 / (-1 + n) ** 2)
            - 342625.0 * 1 / ((-1 + n) * n)
            + 100372.0 * (1 / n - 1 / (1 + n) + 1 / (2 + n) - 1 / (3 + n))
            + 189167.0 * (-(1 / n**2))
            - 29762.0 * lm12m1(n, S1, S2)
            - 12102.0 * lm13m1(n, S1, S2, S3)
        )
        P3ggApp2 = (
            P3gg01
            + 5.4 * bfkl1 * -(1 / (-1 + n) ** 2)
            - 1271540.0 * 1 / ((-1 + n) * n)
            - 649661.0 * (1 / n - 1 / (1 + n) + 1 / (2 + n) - 1 / (3 + n))
            - 1334919.0 * (-(1 / n**2))
            + 191263.0 * lm12m1(n, S1, S2)
            + 36867.0 * lm13m1(n, S1, S2, S3)
        )
    elif nf == 5:
        P3ggApp1 = (
            P3gg01
            + 3.4 * bfkl1 * -(1 / (-1 + n) ** 2)
            - 337540.0 * 1 / ((-1 + n) * n)
            + 119366.0 * (1 / n - 1 / (1 + n) + 1 / (2 + n) - 1 / (3 + n))
            + 223769.0 * (-(1 / n**2))
            - 45129.0 * lm12m1(n, S1, S2)
            - 15046.0 * lm13m1(n, S1, S2, S3)
        )
        P3ggApp2 = (
            P3gg01
            + 5.4 * bfkl1 * -(1 / (-1 + n) ** 2)
            - 1274800.0 * 1 / ((-1 + n) * n)
            - 637406.0 * (1 / n - 1 / (1 + n) + 1 / (2 + n) - 1 / (3 + n))
            - 1314010.0 * (-(1 / n**2))
            + 177882.0 * lm12m1(n, S1, S2)
            + 34362.0 * lm13m1(n, S1, S2, S3)
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
