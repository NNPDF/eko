r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{ns,-}^{(3)}`."""
import numba as nb

from eko.constants import CF, zeta3

from .....harmonics import cache as c
from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1


@nb.njit(cache=True)
def gamma_ns_nf3(n, cache):
    r"""Return the common part proportional to :math:`nf^3`.

    Holds for :math:`\gamma_{ns,+}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,v}^{(3)}`.
    The expression is copied exact from :eqref:`3.6` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ns}^{(3)}|_{nf^3}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)
    eta = 1 / n * 1 / (n + 1)
    g_ns_nf3 = CF * (
        -32 / 27 * zeta3 * eta
        - 16 / 9 * zeta3
        - 16 / 27 * eta**4
        - 16 / 81 * eta**3
        + 80 / 27 * eta**2
        - 320 / 81 * eta
        + 32 / 27 * 1 / (n + 1) ** 4
        + 128 / 27 * 1 / (n + 1) ** 2
        + 64 / 27 * S1 * zeta3
        - 32 / 81 * S1
        - 32 / 81 * S2
        - 160 / 81 * S3
        + 32 / 27 * S4
        + 131 / 81
    )
    return g_ns_nf3


@nb.njit(cache=True)
def gamma_nsm_nf2(n, cache):
    r"""Return the parametrized valence-like non-singlet part proportional to :math:`nf^2`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\gamma_{ns,-}^{(3)}|_{nf^2}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    return (
        -193.8861961960614
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 180.43192466504917 / n
        - 156.88366085122723 / (1.0 + n) ** 3
        + 16.276755890287458 / (1.0 + n) ** 2
        + 9.929878787981112 / (3.0 + n)
        + 195.5772257829161 * S1
        - (430.0434990330973 * S1) / n**2
        + (26.68861454046639 * S1) / n
        + 182.8021908863633 * Lm11m1
        + 161.78964675346762 * Lm12m1
        + 26.1810397685143 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm_nf1(n, cache):
    r"""Return the parametrized valence-like non-singlet part proportional to :math:`nf^1`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\gamma_{ns,-}^{(3)}|_{nf^1}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    return (
        5550.079294018526
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 3248.3974879855336 / n
        - 2898.2943249560426 / (1.0 + n) ** 3
        - 689.0970388084964 / (1.0 + n) ** 2
        - 19.315781886816087 / (3.0 + n)
        - 5171.916129085788 * S1
        + (12317.648319304566 * S1) / n**2
        - (2741.830025124657 * S1) / n
        + 2591.510450390595 * Lm11m1
        - 1731.0652413316732 * Lm12m1
        - 170.02391773352406 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm_nf0(n, cache):
    r"""Return the parametrized valence-like non-singlet part proportional to :math:`nf^0`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\gamma_{ns,-}^{(3)}|_{nf^0}`

    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    return (
        -23371.597456219817
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 11126.581215589995 / n
        + 198059.65286720797 / (1.0 + n) ** 3
        - 96709.11158512249 / (1.0 + n) ** 2
        - 6538.89846648667 / (3.0 + n)
        + 20702.353028966703 * S1
        - (97061.21566841986 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - 197681.93739849582 * Lm11m1
        - 56555.718627160735 * Lm12m1
        - 12779.041608676462 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm(n, nf, cache):
    r"""Compute the |N3LO| valence-like non-singlet anomalous dimension.

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
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\gamma_{ns,-}^{(3)}(N)`

    """
    return (
        gamma_nsm_nf0(n, cache)
        + nf * gamma_nsm_nf1(n, cache)
        + nf**2 * gamma_nsm_nf2(n, cache)
        + nf**3 * gamma_ns_nf3(n, cache)
    )
