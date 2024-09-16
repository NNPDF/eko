"""Weight 1 harmonic sums."""

import numba as nb
import numpy as np

from .polygamma import cern_polygamma


@nb.njit(cache=True)
def S1(N):
    r"""Compute the harmonic sum :math:`S_1(N)`.

    .. math::
      S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi_0(N+1)+\gamma_E

    with :math:`\psi_0(N)` the digamma function and :math:`\gamma_E` the
    Euler-Mascheroni constant.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    S_1 : complex
        (simple) Harmonic sum :math:`S_1(N)`

    See Also
    --------
        ekore.harmonics.polygamma.cern_polygamma : :math:`\psi_k(N)`
    """
    return cern_polygamma(N + 1.0, 0) + np.euler_gamma


@nb.njit(cache=True)
def Sm1(N, hS1, hS1mh, hS1h, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-1}(N)`.

    .. math::
      S_{-1}(N) = \sum\limits_{j=1}^N \frac {(-1)^j} j

    Parameters
    ----------
    N : complex
        Mellin moment
    hS1:  complex
        Harmonic sum :math:`S_{1}(N)`
    hS1mh: complex
        Harmonic sum :math:`S_{1}((N-1)/2)`
    hS1h: complex
        Harmonic sum :math:`S_{1}(N/2)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
        False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm1 : complex
        Harmonic sum :math:`S_{-1}(N)`

    See Also
    --------
        eko.anomalous_dimension.w1.S1 : :math:`S_1(N)`
    """
    if is_singlet is None:
        return (1 - (-1) ** N) / 2 * hS1mh + ((-1) ** N + 1) / 2 * hS1h - hS1
    if is_singlet:
        return hS1h - hS1
    return hS1mh - hS1
