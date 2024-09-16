"""Weight 5 harmonic sums."""

import numba as nb

from eko.constants import zeta5

from .polygamma import cern_polygamma


@nb.njit(cache=True)
def S5(N):
    r"""Compute the harmonic sum :math:`S_5(N)`.

    .. math::
      S_5(N) = \sum\limits_{j=1}^N \frac 1 {j^5} = \frac 1 24 \psi_4(N+1)+\zeta(5)

    with :math:`\psi_4(N)` the 4th-polygamma function and :math:`\zeta` the
    Riemann zeta function.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    S_5 : complex
        Harmonic sum :math:`S_5(N)`

    See Also
    --------
    ekore.harmonics.polygamma.cern_polygamma : :math:`\psi_k(N)`
    """
    return zeta5 + 1.0 / 24.0 * cern_polygamma(N + 1.0, 4)


@nb.njit(cache=True)
def Sm5(N, hS5, hS5mh, hS5h, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-5}(N)`.

    .. math::
      S_{-5}(N) = \sum\limits_{j=1}^N \frac {(-1)^j} {j^5}

    Parameters
    ----------
    N : complex
        Mellin moment
    hS5:  complex
        Harmonic sum :math:`S_{5}(N)`
    hS5mh: complex
        Harmonic sum :math:`S_{5}((N-1)/2)`
    hS5h: complex
        Harmonic sum :math:`S_{5}(N/2)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm5 : complex
        Harmonic sum :math:`S_{-5}(N)`

    See Also
    --------
    eko.harmonic.w5.S5 : :math:`S_5(N)`
    """
    if is_singlet is None:
        return (
            1 / 2**4 * ((1 - (-1) ** N) / 2 * hS5mh + ((-1) ** N + 1) / 2 * hS5h) - hS5
        )
    if is_singlet:
        return 1 / 2**4 * hS5h - hS5
    return 1 / 2**4 * hS5mh - hS5
