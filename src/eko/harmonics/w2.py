# -*- coding: utf-8 -*-
"""
Weight 2 harmonics sum.
"""
import numba as nb

from .constants import zeta2
from .polygamma import cern_polygamma


@nb.njit(cache=True)
def S2(N):
    r"""
    Computes the harmonic sum :math:`S_2(N)`.

    .. math::
      S_2(N) = \sum\limits_{j=1}^N \frac 1 {j^2} = -\psi_1(N+1)+\zeta(2)

    with :math:`\psi_1(N)` the trigamma function and :math:`\zeta` the
    Riemann zeta function.

    Parameters
    ----------
        N : complex
            Mellin moment

    Returns
    -------
        S_2 : complex
            Harmonic sum :math:`S_2(N)`

    See Also
    --------
        eko.harmonics.polygamma.cern_polygamma : :math:`\psi_k(N)`
    """
    return -cern_polygamma(N + 1.0, 1) + zeta2


@nb.njit(cache=True)
def Sm2(N):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2}(N)`.

    .. math::
      S_{-2}(N) = \sum\limits_{j=1}^N \frac (-1)^j j^2

    Parameters
    ----------
        N : complex
            Mellin moment

    Returns
    -------
        Sm2 : complex
            Harmonic sum :math:`S_{-2}(N)`

    See Also
    --------
        eko.anomalous_dimension.w2.S2 : :math:`S_2(N)`
    """
    return (-1) ** N / 4 * (S2(N / 2) - S2((N - 1) / 2)) - zeta2 / 2
