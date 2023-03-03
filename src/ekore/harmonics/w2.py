"""Weight 2 harmonic sums."""

import numba as nb

from .constants import zeta2
from .polygamma import cern_polygamma


@nb.njit(cache=True)
def S2(N):
    r"""Computes the harmonic sum :math:`S_2(N)`.

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
        ekore.harmonics.polygamma.cern_polygamma : :math:`\psi_k(N)`
    """
    return -cern_polygamma(N + 1.0, 1) + zeta2


@nb.njit(cache=True)
def Sm2(N, hS2, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-2}(N)`.

    .. math::
      S_{-2}(N) = \sum\limits_{j=1}^N \frac {(-1)^j}{j^2}

    Parameters
    ----------
        N : complex
            Mellin moment
        hS2:  complex
            Harmonic sum :math:`S_{2}(N)`
        is_singlet: bool, None
            symmetry factor: True for singlet like quantities
            (:math:`\eta=(-1)^N = 1`), False for non-singlet like quantities
            (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
        Sm2 : complex
            Harmonic sum :math:`S_{-2}(N)`

    See Also
    --------
        eko.anomalous_dimension.w2.S2 : :math:`S_2(N)`
    """
    if is_singlet is None:
        return (
            1
            / 2
            * ((1 - (-1) ** N) / 2 * S2((N - 1) / 2) + ((-1) ** N + 1) / 2 * S2(N / 2))
            - hS2
        )
    if is_singlet:
        return 1 / 2 * S2(N / 2) - hS2
    return 1 / 2 * S2((N - 1) / 2) - hS2
