# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,+}^{(3)}`
"""
import numba as nb

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
    return (
        -193.83259645717885
        - 18.962962962962962 / n**5
        + 99.1604938271605 / n**4
        - 226.44075306899038 / n**3
        + 395.60497732877303 / n**2
        + 278.2205375565073 / n
        + 537.132861181022 / (1.0 + n) ** 3
        - 817.9374228369205 / (1.0 + n) ** 2
        - 80.16230542465289 / (2.0 + n)
        + 195.5772257829161 * S1
        - (491.7139266455562 * S1) / n**2
        + (26.68861454046639 * S1) / n
        + (249.125506580144 * S1) / (1.0 + n) ** 3
        + (276.75480984972495 * S1) / (1.0 + n) ** 2
        - (3.24849037613728 * S1) / (1.0 + n)
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
        |N3LO| sea non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^1}`

    """
    S1 = sx[0][0]
    return (
        5549.533222114542
        - 126.41975308641975 / n**6
        + 752.1975308641976 / n**5
        - 2253.1105700880144 / n**4
        + 5247.1769880520205 / n**3
        - 8769.153217295072 / n**2
        - 5834.355552528428 / n
        - 11877.948615823714 / (1.0 + n) ** 3
        + 17141.75538179074 / (1.0 + n) ** 2
        + 2189.7561896037237 / (2.0 + n)
        - 5171.916129085788 * S1
        + (12198.267695106204 * S1) / n**2
        - (2741.830025124657 * S1) / n
        - (6658.5933037552495 * S1) / (1.0 + n) ** 3
        - (6980.106185472365 * S1) / (1.0 + n) ** 2
        + (73.57787513932745 * S1) / (1.0 + n)
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
        |N3LO| sea non-singlet anomalous dimension :math:`\\gamma_{ns,+}^{(3)}|_{nf^0}`

    """
    S1 = sx[0][0]
    return (
        -23389.366023525115
        - 252.8395061728395 / n**7
        + 1580.2469135802469 / n**6
        - 5806.800104704373 / n**5
        + 14899.91711929902 / n**4
        - 28546.38768506619 / n**3
        + 50759.65541232588 / n**2
        + 21477.757730073346 / n
        + 75848.9162996206 / (1.0 + n) ** 3
        - 21458.28316538394 / (1.0 + n) ** 2
        - 7874.846331131067 / (2.0 + n)
        + 20702.353028966703 * S1
        - (73014.16193348375 * S1) / n**2
        + (16950.937339235086 * S1) / n
        + (3275.0528283502285 * S1) / (1.0 + n) ** 3
        + (27872.8964453729 * S1) / (1.0 + n) ** 2
        - (501.0138189552833 * S1) / (1.0 + n)
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
