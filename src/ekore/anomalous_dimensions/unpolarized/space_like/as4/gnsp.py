"""This module contains the anomalous dimension :math:`\\gamma_{ns,+}^{(3)}`

"""
import numba as nb

from .....harmonics.log_functions import lm11m1, lm12m1, lm13m1
from .gnsm import gamma_ns_nf3


@nb.njit(cache=True)
def gamma_nsp_nf2(n, sx):
    """Implements the parametrized singlet-like non-singlet part proportional to :math:`nf^2`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nsp_nf2 : complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^2}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -193.85479604848626
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 278.2205375565073 / n
        + 351.0191873322347 / (1.0 + n) ** 3
        - 115.83082177850747 / (1.0 + n) ** 2
        - 97.28108136993096 / (2.0 + n)
        + 195.5772257829161 * S1
        - (617.3467445575227 * S1) / n**2
        + (26.68861454046639 * S1) / n
        - 179.1958949029147 * Lm11m1
        + 78.89342505600462 * Lm12m1
        + 7.093169138453943 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp_nf1(n, sx):
    """Implements the parametrized singlet-like non-singlet part proportional to :math:`nf^1`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nsp_nf1 : complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        5550.063827367692
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        - 7856.698193404992 / (1.0 + n) ** 3
        - 1780.0619991736955 / (1.0 + n) ** 2
        + 2565.962271083018 / (2.0 + n)
        - 5171.916129085788 * S1
        + (15756.961591759926 * S1) / n**2
        - (2741.830025124657 * S1) / n
        + 4352.896032770554 * Lm11m1
        - 2110.3938915736526 * Lm12m1
        - 204.23062235898342 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp_nf0(n, sx):
    """Implements the parametrized singlet-like non-singlet part proportional to :math:`nf^0`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    g_nsp_nf0 : complex
        |N3LO| singlet-like non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, sx[1][0])
    Lm13m1 = lm13m1(n, S1, sx[1][0], sx[2][0])
    return (
        -23391.854890259732
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        + 26932.571453853965 / (1.0 + n) ** 3
        - 17764.722568307116 / (1.0 + n) ** 2
        - 10913.245002601085 / (2.0 + n)
        + 20702.353028966703 * S1
        - (66520.50722889033 * S1) / n**2
        + (16950.937339235086 * S1) / n
        - 31046.58355660925 * Lm11m1
        - 90.76127492188596 * Lm12m1
        - 497.92637248421335 * Lm13m1
    )


@nb.njit(cache=True)
def gamma_nsp(n, nf, sx):
    """Computes the |N3LO| singlet-like non-singlet anomalous dimension.

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
        :math:`\\gamma_{ns,+}^{(3)}(N)`

    See Also
    --------
    gamma_nsp_nf0: :math:`\\gamma_{ns,+}^{(3)}|_{nf^0}`
    gamma_nsp_nf1: :math:`\\gamma_{ns,+}^{(3)}|_{nf^1}`
    gamma_nsp_nf2: :math:`\\gamma_{ns,+}^{(3)}|_{nf^2}`
    gamma_ns_nf3: :math:`\\gamma_{ns}^{(3)}|_{nf^3}`

    """
    return (
        gamma_nsp_nf0(n, sx)
        + nf * gamma_nsp_nf1(n, sx)
        + nf**2 * gamma_nsp_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
