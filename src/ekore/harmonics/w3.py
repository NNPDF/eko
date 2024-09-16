"""Weight 3 harmonic sums."""

import numba as nb

from eko.constants import log2, zeta2, zeta3

from . import g_functions as gf
from .polygamma import cern_polygamma, symmetry_factor


@nb.njit(cache=True)
def S3(N):
    r"""Compute the harmonic sum :math:`S_3(N)`.

    .. math::
      S_3(N) = \sum\limits_{j=1}^N \frac 1 {j^3} = \frac 1 2 \psi_2(N+1)+\zeta(3)

    with :math:`\psi_2(N)` the 2nd-polygamma function and :math:`\zeta` the
    Riemann zeta function.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    S_3 : complex
        Harmonic sum :math:`S_3(N)`

    See Also
    --------
    ekore.harmonics.polygamma.cern_polygamma : :math:`\psi_k(N)`
    """
    return 0.5 * cern_polygamma(N + 1.0, 2) + zeta3


@nb.njit(cache=True)
def Sm3(N, hS3, hS3mh, hS3h, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-3}(N)`.

    .. math::
      S_{-3}(N) = \sum\limits_{j=1}^N \frac {(-1)^j} {j^3}

    Parameters
    ----------
    N : complex
        Mellin moment
    hS3:  complex
        Harmonic sum :math:`S_{3}(N)`
    hS3mh: complex
        Harmonic sum :math:`S_{3}((N-1)/2)`
    hS3h: complex
        Harmonic sum :math:`S_{3}(N/2)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
        False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm3 : complex
        Harmonic sum :math:`S_{-3}(N)`

    See Also
    --------
    ekore.harmonics.w3.S3 : :math:`S_3(N)`
    """
    if is_singlet is None:
        return (
            1 / 2**2 * ((1 - (-1) ** N) / 2 * hS3mh + ((-1) ** N + 1) / 2 * hS3h) - hS3
        )
    if is_singlet:
        return 1 / 2**2 * hS3h - hS3
    return 1 / 2**2 * hS3mh - hS3


@nb.njit(cache=True)
def S21(N, S1, S2):
    r"""Analytic continuation of harmonic sum :math:`S_{2,1}(N)`.

    As implemented in :eqref:`B.5.77` of :cite:`MuselliPhD` and :eqref:`37` of
    :cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`

    Returns
    -------
    S21 : complex
        Harmonic sum :math:`S_{2,1}(N)`
    """
    return -gf.mellin_g18(N, S1, S2) + 2 * zeta3


@nb.njit(cache=True)
def Sm21(N, S1, Sm1, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,1}(N)`.

    As implemented in :eqref:`B.5.75` of :cite:`MuselliPhD` and :eqref:`22` of
    :cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1:  complex
        Harmonic sum :math:`S_{1}(N)`
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N = 1`),
        False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm21 : complex
        Harmonic sum :math:`S_{-2,1}(N)`
    """
    # Note mellin g3 was integrated following x^(N-1) convention.
    eta = symmetry_factor(N, is_singlet)
    return (
        -eta * gf.mellin_g3(N + 1, S1 + 1 / (N + 1))
        + zeta2 * Sm1
        - 5 / 8 * zeta3
        + zeta2 * log2
    )


@nb.njit(cache=True)
def S2m1(N, S2, Sm1, Sm2, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{2,-1}(N)`.

    As implemented in :eqref:`B.5.76` of :cite:`MuselliPhD` and :eqref:`23` of
    :cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S2: complex
        Harmonic sum :math:`S_{2}(N)`
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`
    Sm2: complex
        Harmonic sum :math:`S_{-2}(N)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    S2m1 : complex
        Harmonic sum :math:`S_{2,-1}(N)`
    """
    eta = symmetry_factor(N, is_singlet)
    return (
        -eta * gf.mellin_g4(N)
        - log2 * (S2 - Sm2)
        - 1 / 2 * zeta2 * Sm1
        + 1 / 4 * zeta3
        - 1 / 2 * zeta2 * log2
    )


@nb.njit(cache=True)
def Sm2m1(N, S1, S2, Sm2):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,-1}(N)`.

    As implemented in :eqref:`B.5.74` of :cite:`MuselliPhD` and :eqref:`38` of
    :cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`
    Sm2: complex
        Harmonic sum :math:`S_{-2}(N)`

    Returns
    -------
    Sm2m1 : complex
        Harmonic sum :math:`S_{-2,-1}(N)`
    """
    return -gf.mellin_g19(N, S1) + log2 * (S2 - Sm2) - 5 / 8 * zeta3
