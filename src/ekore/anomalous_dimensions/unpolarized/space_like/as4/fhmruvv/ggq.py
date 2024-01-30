r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{gq}^{(3)}`."""

import numba as nb

from ......harmonics import cache as c
from ......harmonics.log_functions import lm12, lm13, lm14, lm15


@nb.njit(cache=True)
def gamma_gq(n, nf, cache, variation):
    r"""Compute the |N3LO| gluon-quark singlet anomalous dimension.

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

    # Small-x, Casimir scaled from P_gg (approx. for bfkl1)
    bfkl0 = -8.3086173 * 10**3 / 2.25
    bfkl1 = (-1.0691199 * 10**5 - nf * 9.9638304 * 10**2) / 2.25

    # The resulting part of the function
    P3GQ01 = (
        +bfkl0 * (-(6 / (-1 + n) ** 4))
        + bfkl1 * 2 / (-1 + n) ** 3
        + x1L4cff * lm14(n, S1, S2, S3, S4)
        + x1L5cff * lm15(n, S1, S2, S3, S4, S5)
    )

    # The selected approximations for nf = 3, 4, 5
    if nf == 3:
        P3gqApp1 = (
            P3GQ01
            + 3.4 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 161562.0 * 1 / ((-1 + n) * n)
            + 36469.0 * 1 / n
            + 72317.0 * (-(1 / n**2))
            - 3977.3 * lm12(n, S1, S2)
            + 484.4 * lm13(n, S1, S2, S3)
        )
        P3gqApp2 = (
            P3GQ01
            + 5.4 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 546482.0 * 1 / ((-1 + n) * n)
            - 39464.0 * 1 / n
            - 401000.0 * (-(1 / n**2))
            + 13270.0 * lm12(n, S1, S2)
            + 3289.0 * lm13(n, S1, S2, S3)
        )
    elif nf == 4:
        P3gqApp1 = (
            P3GQ01
            + 3.4 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 158805.0 * 1 / ((-1 + n) * n)
            + 35098.0 * 1 / n
            + 87258.0 * (-(1 / n**2))
            - 4834.1 * lm12(n, S1, S2)
            + 176.6 * lm13(n, S1, S2, S3)
        )
        P3gqApp2 = (
            P3GQ01
            + 5.4 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 547215.0 * 1 / ((-1 + n) * n)
            - 41523.0 * 1 / n
            - 390350.0 * (-(1 / n**2))
            + 12571.0 * lm12(n, S1, S2)
            + 3007.0 * lm13(n, S1, S2, S3)
        )
    elif nf == 5:
        P3gqApp1 = (
            P3GQ01
            + 3.4 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 154336.0 * 1 / ((-1 + n) * n)
            + 33889.0 * 1 / n
            + 103440.0 * (-(1 / n**2))
            - 5745.8 * lm12(n, S1, S2)
            - 128.6 * lm13(n, S1, S2, S3)
        )
        P3gqApp2 = (
            P3GQ01
            + 5.4 * bfkl1 * (-(1 / (-1 + n) ** 2))
            - 546236.0 * 1 / ((-1 + n) * n)
            - 43421.0 * 1 / n
            - 378460.0 * (-(1 / n**2))
            + 11816.0 * lm12(n, S1, S2)
            + 2727.3 * lm13(n, S1, S2, S3)
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
