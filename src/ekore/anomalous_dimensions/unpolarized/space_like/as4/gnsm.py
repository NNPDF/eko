r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{ns,-}^{(3)}`."""
import numba as nb

from eko.constants import CF, zeta3

from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1


@nb.njit(cache=True)
def gamma_ns_nf3(n, sx):
    r"""Return the common part proportional to :math:`nf^3`.

    Holds for :math:`\gamma_{ns,+}^{(3)},\gamma_{ns,-}^{(3)},\gamma_{ns,v}^{(3)}`.
    The expression is copied exact from :eqref:`3.6` of :cite:`Davies:2016jie`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_ns_nf3 : complex
        |N3LO| non-singlet anomalous dimension :math:`\gamma_{ns}^{(3)}|_{nf^3}`

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
    r"""Return the parametrized valence-like non-singlet part proportional to :math:`nf^2`.

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
        :math:`\gamma_{ns,-}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -193.89011158240442
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 278.2205375565073 / n
        - 192.0247446101425 / (1.0 + n) ** 3
        + 93.91039406948033 / (1.0 + n) ** 2
        - 85.81679567653221 / (3.0 + n)
        + 195.5772257829161 * S1
        - (488.477491593376 * S1) / n**2
        + (26.68861454046639 * S1) / n
        + 361.0392247000297 * Lm11m1
        + 232.1144024429168 * Lm12m1
        + 35.38568209541474 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm_nf1(n, sx):
    r"""Return the parametrized valence-like non-singlet part proportional to :math:`nf^1`.

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
        :math:`\gamma_{ns,-}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        5550.182834015097
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        - 1969.0104529610248 / (1.0 + n) ** 3
        - 2742.0697059315535 / (1.0 + n) ** 2
        + 2512.6444931763654 / (3.0 + n)
        - 5171.916129085788 * S1
        + (13862.898314841788 * S1) / n**2
        - (2741.830025124657 * S1) / n
        - 2121.855469704418 * Lm11m1
        - 3590.759053757736 * Lm12m1
        - 413.4348940200741 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm_nf0(n, sx):
    r"""Return the parametrized valence-like non-singlet part proportional to :math:`nf^0`.

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
        :math:`\gamma_{ns,-}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -23372.01191013195
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        + 194339.87834020052 / (1.0 + n) ** 3
        - 88491.39062175922 / (1.0 + n) ** 2
        - 16673.930496518376 / (3.0 + n)
        + 20702.353028966703 * S1
        - (103246.60425090564 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - 178815.0878250944 * Lm11m1
        - 49111.66189344577 * Lm12m1
        - 11804.70644702107 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    r"""Compute the |N3LO| valence-like non-singlet anomalous dimension.

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
        :math:`\gamma_{ns,-}^{(3)}(N)`

    """
    return (
        gamma_nsm_nf0(n, sx)
        + nf * gamma_nsm_nf1(n, sx)
        + nf**2 * gamma_nsm_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
