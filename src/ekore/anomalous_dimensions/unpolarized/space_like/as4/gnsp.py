r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{ns,+}^{(3)}`."""
import numba as nb

from .....harmonics import cache as c
from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1
from .gnsm import gamma_ns_nf3


@nb.njit(cache=True)
def gamma_nsp_nf2(n, cache):
    r"""Return the parametrized singlet-like non-singlet part proportional to :math:`nf^2`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\gamma_{ns,+}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    return (
        -193.85996720614918
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 180.43192466504917 / n
        + 40.25996504960858 / (1.0 + n) ** 3
        - 122.47540320835951 / (1.0 + n) ** 2
        + 2.2109706180574356 / (2.0 + n)
        + 195.5772257829161 * S1
        - (457.3382023459039 * S1) / n**2
        + (26.68861454046639 * S1) / n
        - 16.251628171151797 * Lm11m1
        + 89.75298129002869 * Lm12m1
        + 10.429745596675895 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp_nf1(n, cache):
    r"""Return the parametrized singlet-like non-singlet part proportional to :math:`nf^1`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\gamma_{ns,+}^{(3)}|_{nf^1}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    return (
        5550.218627756056
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 3248.3974879855336 / n
        + 1045.759889779711 / (1.0 + n) ** 3
        - 1517.7637756534405 / (1.0 + n) ** 2
        - 71.40345295295322 / (2.0 + n)
        - 5171.916129085788 * S1
        + (11292.198190230158 * S1) / n**2
        - (2741.830025124657 * S1) / n
        - 380.3748812945835 * Lm11m1
        - 2478.792283615627 * Lm12m1
        - 308.36695798751623 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp_nf0(n, cache):
    r"""Return the parametrized singlet-like non-singlet part proportional to :math:`nf^0`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\gamma_{ns,+}^{(3)}|_{nf^0}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    return (
        -23391.048834675064
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 11126.581215589995 / n
        + 45365.96043751846 / (1.0 + n) ** 3
        - 11976.515209798297 / (1.0 + n) ** 2
        - 858.6772726856868 / (2.0 + n)
        + 20702.353028966703 * S1
        - (67084.61612292472 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - 45610.240259669365 * Lm11m1
        - 5030.833428351564 * Lm12m1
        - 1337.0046605808507 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp(n, nf, cache):
    r"""Compute the |N3LO| singlet-like non-singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| singlet-like non-singlet anomalous dimension
        :math:`\gamma_{ns,+}^{(3)}(N)`

    """
    return (
        gamma_nsp_nf0(n, cache)
        + nf * gamma_nsp_nf1(n, cache)
        + nf**2 * gamma_nsp_nf2(n, cache)
        + nf**3 * gamma_ns_nf3(n, cache)
    )
