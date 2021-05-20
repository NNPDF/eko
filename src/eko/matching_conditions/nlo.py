# -*- coding: utf-8 -*-
r"""
This module contains the |NLO| operator-matrix elements (OMEs)
for the matching conditions in the |VFNS|.
Heavy quark contribution for intrinsic evolution are included.
The expession are taken from :cite:`Ball_2016` and Mellin transformed with Mathematica.
"""
import numba as nb

from ..constants import CF


@nb.njit("c16(c16,f8)", cache=True)
def A_gh_1(n, L):
    """
    math:`A_{gH}^{(1)}` operator-matrix element defined as the
    mellin transform of :math:`K_{gh}` given in Eq. (20b) of :cite:`Ball_2016`.

    Parameters
    ----------
        n : complex
            Mellin moment
        L : float
            :math:`ln(\frac{q^2}{m_h^2})`

    Returns
    -------
        A_hg_1 : complex
            |NLO|  :math:`A_{gH}^{(1)}` operator-matrix element
    """

    den = 1.0 / (n * (n ** 2 - 1)) ** 2
    agh = -4 + n * (2 + n * (15 + n * (3 + n - n ** 2)))
    agh_m = n * (n ** 2 - 1) * (2 + n + n ** 2) * L
    return 2 * CF * den * (agh + agh_m)


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_hh_1(n, sx, L):
    """
    math:`A_{HH}^{(1)}` operator-matrix element defined as the
    mellin transform of :math:`K_{hh}` given in Eq. (20a) of :cite:`Ball_2016`.

    Parameters
    ----------
        N : complex
            Mellin moment
        sx : numpy.ndarray
            List of harmonic sums
        L : float
            :math:`ln(\frac{q^2}{m_h^2})`

    Returns
    -------
        A_hh_1 : complex
            |NLO|  :math:`A_{HH}^{(1)}` operator-matrix element
    """
    S1 = sx[0]
    den = 1.0 / (n * (n + 1))
    ahh = 1 - 2 * n - 2 / (1 + n) + (2 + 4 * n) * S1
    ahh_m = (1 + 2 * n) * L
    return -2 * CF * den * (ahh + ahh_m)
