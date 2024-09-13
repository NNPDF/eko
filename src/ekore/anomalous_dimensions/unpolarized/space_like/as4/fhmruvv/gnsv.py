r"""The unpolarized, space-like anomalous dimension
:math:`\gamma_{ns,v}^{(3)}`."""

import numba as nb

from ......harmonics import cache as c
from ......harmonics.log_functions import lm11m1, lm12m1, lm13m1
from .gnsm import gamma_nsm


@nb.njit(cache=True)
def gamma_nss(n, nf, cache, variation):
    r"""Compute the |N3LO| sea non-single anomalous dimension.

    The routine is taken from :cite:`Moch:2017uml`.

    The :math:`nf^2` part is a high-accuracy (0.1% or better) parametrization
    of the exact expression obtained in :cite:`Davies:2016jie`, see xpns3m.f

    The :math:`nf^1` part is an approximation based on the first 9 odd moments.
    The two sets spanning the error estimate are called via  IMOD = 1
    and  IMOD = 2.  Any other value of IMOD invokes their average.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    variation : int
        |N3LO| anomalous dimension variation

    Returns
    -------
    complex
        |N3LO| sea non-singlet anomalous dimension :math:`\gamma_{ns,s}^{(3)}`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)

    # nf^1: two approximations
    P3NSA11 = (
        2880 / n**7
        - 11672.4 / n**6
        + 12802.560000000001 / n**5
        - 7626.66 / n**4
        + 6593.2 / n**3
        - 3687.6 / n**2
        + 4989.2 / (1 + n)
        - 6596.93 / (2 + n)
        + 1607.73 / (3 + n)
        + 60.4 * lm12m1(n, S1, S2)
        + 4.685 * lm13m1(n, S1, S2, S3)
    )
    P3NSA12 = (
        -2880 / n**7
        + 4066.32 / n**6
        - 5682.24 / n**5
        + 5540.88 / n**4
        + 546.1 / n**3
        - 2987.83 / n**2
        + 2533.54 / n
        - 1502.75 / (1 + n)
        - 2297.56 / (2 + n)
        + 1266.77 / (3 + n)
        - 254.63 * lm11m1(n, S1)
        - 0.28953 * lm13m1(n, S1, S2, S3)
    )

    # nf^2 (parametrized)
    P3NSSA2 = (
        47.4074 / n**6
        - 142.222 / n**5
        + 32.1201 / n**4
        - 132.824 / n**3
        + 647.397 / n**2
        + 19.7 * lm11m1(n, S1)
        - 3.43547 * lm12m1(n, S1, S2)
        - 1262.0951538579698 / n
        - 187.17000000000002 / (1 + n) ** 4
        + 453.885 / (1 + n) ** 3
        + 147.01749999999998 / (1 + n) ** 2
        + 1614.1000000000001 / (1 + n)
        - 380.12500000000006 / (2 + n)
        - 42.575 / (3 + n)
        + (42.977500000000006 * S2) / n
        + (0.0900000000000047 * (477.52777777775293 + n) * S1) / (n**2 * (1 + n))
    )

    if variation == 1:
        P3NSSA = nf * P3NSA11 + nf**2 * P3NSSA2
    elif variation == 2:
        P3NSSA = nf * P3NSA12 + nf**2 * P3NSSA2
    else:
        P3NSSA = 0.5 * nf * (P3NSA11 + P3NSA12) + nf**2 * P3NSSA2
    return -P3NSSA


@nb.njit(cache=True)
def gamma_nsv(n, nf, cache, variation):
    r"""Compute the |N3LO| valence non-singlet anomalous dimension.

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
        |N3LO| valence non-singlet anomalous dimension
        :math:`\gamma_{ns,v}^{(3)}(N)`
    """
    return gamma_nsm(n, nf, cache, variation) + gamma_nss(n, nf, cache, variation)
