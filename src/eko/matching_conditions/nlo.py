# -*- coding: utf-8 -*-
r"""
This module contains the |NLO| operator-matrix elements (OMEs)
for the matching conditions in the |VFNS|.
Heavy quark contribution for intrinsic evolution are taken from :cite:`Ball_2016`
and Mellin transformed with Mathematica.
The other matching conditions for the |VFNS| at :math:`\mu_F^2 != m_H^2`
are provided in :cite:`Buza_1998` and Mellin transformed with Mathematica.
"""
import numba as nb
import numpy as np

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


@nb.njit("c16(c16,c16[:],f8)", cache=True)
def A_ns_1(n, sx, L):
    """
    math:`A_{qq,H}^{NS,(1)}` operator-matrix element.
    Mellin transform of QCDNUM function ``A1QQNS``

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
        A_ns_1 : complex
            |NLO|  :math:`A_{qq,H}^{NS,(1)}` operator-matrix element
    """
    # TODO: this expression is not correct
    # S1 = sx[0]
    # den = 8.0 / (3.0 * n * (n + 1) ** 2)
    # ans = 1 + n + 2 * n ** 2 - 2 * (1 + n) * (1 + 2 * n) * S1
    # ans_m = -(1 + n) * (1 + 2 * n) * L
    # return (ans + ans_m) * den
    return 0.0


@nb.njit("c16(c16,f8)", cache=True)
def A_hg_1(n, L):
    """
    math:`A_{Hg}^{S,(1)}` operator-matrix element defined as the
    mellin transform of Eq. (B.2) from :cite:`Buza_1998`.

    Parameters
    ----------
        N : complex
            Mellin moment
        L : float
            :math:`ln(\frac{q^2}{m_h^2})`

    Returns
    -------
        A_hg_1 : complex
            |NLO|  :math:`A_{Hg}^{S,(1)}` operator-matrix element
    """
    den = 1.0 / (n * (n + 1) * (2 + n))
    num = 2 * (2 + n + n ** 2)
    return num * den * L


@nb.njit("c16(c16,f8)", cache=True)
def A_gg_1(n, L):
    """
    math:`A_{gg,H}^{S,(1)}` operator-matrix element defined as the
    mellin transform of Eq. (B.6) from :cite:`Buza_1998`.

    Parameters
    ----------
        N : complex
            Mellin moment
        L : float
            :math:`ln(\frac{q^2}{m_h^2})`

    Returns
    -------
        A_gg_1 : complex
            |NLO|  :math:`A_{gg,H}^{S,(1)}` operator-matrix element
    """
    return -2.0 / (3.0 * n) * L


@nb.njit("c16[:,:](c16,c16[:],f8)", cache=True)
def A_singlet_1(n, sx, L):
    r"""
      Computes the |NLO| heavy-quark singlet operator matrix elements

      .. math::
          A^{S,(1)} = \left(\begin{array}{cc}
            A_{qq,H}^{NS,(1)} + A_{hq}^{PS,(1)} & A_{hg}^{S,(1)}\\
            0 & A_{gg,H}^{S,(1)}
          \end{array}\right)

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
        A_S_1 : numpy.ndarray
            |NLO| heavy-quark singlet operator matrix elements :math:`A^{S,(1)}(N)`

      See Also
      --------
        A_ns_1 : :math:`A_{qq,H}^{NS,(1)}`
        A_hq_1 : :math:`A_{hq}^{PS,(1)}`
        A_hg_1 : :math:`A_{hg}^{S,(1)}`
        A_gg_1 : :math:`A_{gg,H}^{S,(1)}`
    """
    A_hq = A_ns_1(n, sx, L)  # + A_hq_1(n,sx,L)
    A_hg = A_hg_1(n, L)
    A_gg = A_gg_1(n, L)
    A_S_1 = np.array([[A_hq, A_hg], [0.0, A_gg]], np.complex_)
    return A_S_1
