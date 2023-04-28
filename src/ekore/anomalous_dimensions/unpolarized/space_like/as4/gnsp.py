r"""The unpolarized, space-like anomalous dimension :math:`\gamma_{ns,+}^{(3)}`."""
import numba as nb

from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1
from .gnsm import gamma_ns_nf3


@nb.njit(cache=True)
def gamma_nsp_nf2(n, sx):
    r"""Return the parametrized singlet-like non-singlet part proportional to :math:`nf^2`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nsp_nf2 : complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\gamma_{ns,+}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -193.862483821757
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 278.2205375565073 / n
        + 59.46630017646719 / (1.0 + n) ** 3
        - 152.70402416764668 / (1.0 + n) ** 2
        - 94.57207315818547 / (2.0 + n)
        + 195.5772257829161 * S1
        - (517.9354004395117 * S1) / n**2
        + (26.68861454046639 * S1) / n
        + 1.5006487633206929 * Lm11m1
        + 113.48340560825889 * Lm12m1
        + 13.865450025251006 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp_nf1(n, sx):
    r"""Return the parametrized singlet-like non-singlet part proportional to :math:`nf^1`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nsp_nf1 : complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\gamma_{ns,+}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        5550.285178175209
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        + 537.8609133198307 / (1.0 + n) ** 3
        - 718.3874592628895 / (1.0 + n) ** 2
        + 2487.96294221855 / (2.0 + n)
        - 5171.916129085788 * S1
        + (12894.65275887218 * S1) / n**2
        - (2741.830025124657 * S1) / n
        - 849.8232086542307 * Lm11m1
        - 3106.3285877376907 * Lm12m1
        - 399.22204467960154 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp_nf0(n, sx):
    r"""Return the parametrized singlet-like non-singlet part proportional to :math:`nf^0`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nsp_nf0 : complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\gamma_{ns,+}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -23391.315223909038
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        + 47399.00434062458 / (1.0 + n) ** 3
        - 15176.296853013831 / (1.0 + n) ** 2
        - 11103.411980157494 / (2.0 + n)
        + 20702.353028966703 * S1
        - (73498.98594171858 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - 43731.12143482942 * Lm11m1
        - 2518.9090401926924 * Lm12m1
        - 973.3270027901576 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp(n, nf, sx):
    r"""Compute the |N3LO| singlet-like non-singlet anomalous dimension.

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
        |N3LO| singlet-like non-singlet anomalous dimension
        :math:`\gamma_{ns,+}^{(3)}(N)`

    """
    return (
        gamma_nsp_nf0(n, sx)
        + nf * gamma_nsp_nf1(n, sx)
        + nf**2 * gamma_nsp_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
