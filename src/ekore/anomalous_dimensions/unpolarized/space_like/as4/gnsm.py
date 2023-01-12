"""This module contains the anomalous dimension :math:`\\gamma_{ns,-}^{(3)}`

"""
import numba as nb

from eko.constants import CF
from .....harmonics.constants import zeta3
from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1


@nb.njit(cache=True)
def gamma_ns_nf3(n, sx):
    """Implements the common part proportional to :math:`nf^3`,
    of :math:`\\gamma_{ns,+}^{(3)},\\gamma_{ns,-}^{(3)},\\gamma_{ns,v}^{(3)}`.
    The expression is copied exact from Eq. 3.6. of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf3 : complex
        |N3LO| non-singlet anomalous dimension :math:`\\gamma_{ns}^{(3)}|_{nf^3}`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
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
def gamma_nsm_nf2(n, sx):
    """Implements the parametrized valence-like non-singlet part proportional to :math:`nf^2`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf2 : complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -193.85692903712987
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 278.2205375565073 / n
        + 433.6813853974481 / (1.0 + n) ** 3
        + 77.59871714838847 / (1.0 + n) ** 2
        - 96.10213330829208 / (3.0 + n)
        + 195.5772257829161 * S1
        - (680.3854257809448 * S1) / n**2
        + (26.68861454046639 * S1) / n
        - 76.56139620361596 * Lm11m1
        + 126.58793206934183 * Lm12m1
        + 13.001858308748329 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm_nf1(n, sx):
    """Implements the parametrized valence-like non-singlet part proportional to :math:`nf^1`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf1 : complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        5549.7951398549685
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        - 9279.55898142154 / (1.0 + n) ** 3
        - 2551.4893186377476 / (1.0 + n) ** 2
        + 2632.8150611003307 / (3.0 + n)
        - 5171.916129085788 * S1
        + (16105.088692447685 * S1) / n**2
        - (2741.830025124657 * S1) / n
        + 2990.9289986960293 * Lm11m1
        - 2357.8218160665683 * Lm12m1
        - 151.90951695259517 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm_nf0(n, sx):
    """Implements the parametrized valence-like non-singlet part proportional to :math:`nf^0`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
        g_ns_nf0 : complex
            |N3LO| valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-}^{(3)}|_{nf^0}`
    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -23383.08164724965
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        - 14396.446623479116 / (1.0 + n) ** 3
        - 83049.79603015777 / (1.0 + n) ** 2
        - 13242.729433996457 / (3.0 + n)
        + 20702.353028966703 * S1
        - (39225.8841951996 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - 32830.98964964463 * Lm11m1
        - 13907.903486384817 * Lm12m1
        - 4337.4357337052215 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    """Computes the |N3LO| valence-like non-singlet anomalous dimension.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    sx : list
        harmonic sums cache

    Returns
    -------
    gamma_nsp : complex
        |N3LO| valence-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(3)}(N)`

    See Also
    --------
    gamma_nsm_nf0: :math:`\\gamma_{ns,-}^{(3)}|_{nf^0}`
    gamma_nsm_nf1: :math:`\\gamma_{ns,-}^{(3)}|_{nf^1}`
    gamma_nsm_nf2: :math:`\\gamma_{ns,-}^{(3)}|_{nf^2}`
    gamma_ns_nf3: :math:`\\gamma_{ns}^{(3)}|_{nf^3}`

    """
    return (
        gamma_nsm_nf0(n, sx)
        + nf * gamma_nsm_nf1(n, sx)
        + nf**2 * gamma_nsm_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
