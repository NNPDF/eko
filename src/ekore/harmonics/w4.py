"""Weight 4 harmonic sums."""

import numba as nb

from eko.constants import li4half, log2, zeta2, zeta3, zeta4

from . import g_functions as gf
from .polygamma import cern_polygamma, symmetry_factor


@nb.njit(cache=True)
def S4(N):
    r"""Compute the harmonic sum :math:`S_4(N)`.

    .. math::
      S_4(N) = \sum\limits_{j=1}^N \frac 1 {j^4} = - \frac 1 6 \psi_3(N+1)+\zeta(4)

    with :math:`\psi_3(N)` the 3rd-polygamma function and :math:`\zeta` the
    Riemann zeta function.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    S_4 : complex
        Harmonic sum :math:`S_4(N)`

    See Also
    --------
    ekore.harmonics.polygamma.cern_polygamma : :math:`\psi_k(N)`
    """
    return zeta4 - 1.0 / 6.0 * cern_polygamma(N + 1.0, 3)


@nb.njit(cache=True)
def Sm4(N, hS4, hS4mh, hS4h, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-4}(N)`.

    .. math::
      S_{-4}(N) = \sum\limits_{j=1}^N \frac {(-1)^j} {j^4}

    Parameters
    ----------
    N : complex
        Mellin moment
    hS4:  complex
        Harmonic sum :math:`S_{4}(N)`
    hS4mh: complex
        Harmonic sum :math:`S_{4}((N-1)/2)`
    hS4h: complex
        Harmonic sum :math:`S_{4}(N/2)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm4 : complex
        Harmonic sum :math:`S_{-4}(N)`

    See Also
    --------
    eko.anomalous_dimension.w4.S4 : :math:`S_4(N)`
    """
    if is_singlet is None:
        return (
            1 / 2**3 * ((1 - (-1) ** N) / 2 * hS4mh + ((-1) ** N + 1) / 2 * hS4h) - hS4
        )
    if is_singlet:
        return 1 / 2**3 * hS4h - hS4
    return 1 / 2**3 * hS4mh - hS4


@nb.njit(cache=True)
def Sm31(N, S1, Sm1, Sm2, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-3,1}(N)`.

    As implemented in :eqref:`B.5.93` of :cite:`MuselliPhD` and :eqref:`25` of
    cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`
    Sm2: complex
        Harmonic sum :math:`S_{-2}(N)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm31 : complex
        Harmonic sum :math:`S_{-3,1}(N)`

    See Also
    --------
    ekore.harmonics.g_functions.mellin_g6 : :math:`g_6(N)`
    """
    eta = symmetry_factor(N, is_singlet)
    return (
        eta * gf.mellin_g6(N, S1)
        + zeta2 * Sm2
        - zeta3 * Sm1
        - 3 / 5 * zeta2**2
        + 2 * li4half
        + 3 / 4 * zeta3 * log2
        - 1 / 2 * zeta2 * log2**2
        + 1 / 12 * log2**4
    )


@nb.njit(cache=True)
def Sm22(N, S1, S2, Sm2, Sm31, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,2}(N)`.

    As implemented in :eqref:`B.5.94` of :cite:`MuselliPhD` and :eqref:`24` of
    cite:`Bl_mlein_2000`.

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
    Sm31: complex
        Harmonic sum :math:`S_{-3,1}(N)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm22 : complex
        Harmonic sum :math:`S_{-2,2}(N)`

    See Also
    --------
    ekore.harmonics.g_functions.mellin_g5 : :math:`g_5(N)`
    """
    eta = symmetry_factor(N, is_singlet)
    return (
        eta * gf.mellin_g5(N, S1, S2) - 2 * Sm31 + 2 * zeta2 * Sm2 + 3 / 40 * zeta2**2
    )


@nb.njit(cache=True)
def Sm211(N, S1, S2, Sm1, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,1,1}(N)`.

    As implemented in :eqref:`B.5.104` of :cite:`MuselliPhD` and :eqref:`27` of
    cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm221 : complex
        Harmonic sum :math:`S_{-2,1,1}(N)`

    See Also
    --------
    ekore.harmonics.g_functions.mellin_g8 : :math:`g_8(N)`
    """
    eta = symmetry_factor(N, is_singlet)
    return (
        -eta * gf.mellin_g8(N, S1, S2)
        + zeta3 * Sm1
        - li4half
        + 1 / 8 * zeta2**2
        + 1 / 8 * zeta3 * log2
        + 1 / 4 * zeta2 * log2**2
        - 1 / 24 * log2**4
    )


@nb.njit(cache=True)
def S211(N, S1, S2, S3):
    r"""Analytic continuation of harmonic sum :math:`S_{2,1,1}(N)`.

    As implemented in :eqref:`B.5.115` of :cite:`MuselliPhD` and :eqref:`40` of
    cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`
    S3: complex
        Harmonic sum :math:`S_{3}(N)`

    Returns
    -------
    S211 : complex
        Harmonic sum :math:`S_{2,1,1}(N)`

    See Also
    --------
    ekore.harmonics.g_functions.mellin_g21 : :math:`g_21(N)`
    """
    return -gf.mellin_g21(N, S1, S2, S3) + 6 / 5 * zeta2**2


@nb.njit(cache=True)
def S31(N, S1, S2, S3, S4):
    r"""Analytic continuation of harmonic sum :math:`S_{3,1}(N)`.

    As implemented in :eqref:`B.5.99` of :cite:`MuselliPhD` and :eqref:`41` of
    cite:`Bl_mlein_2000`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`
    S3: complex
        Harmonic sum :math:`S_{3}(N)`
    S4: complex
        Harmonic sum :math:`S_{4}(N)`

    Returns
    -------
    S31 : complex
        Harmonic sum :math:`S_{3,1}(N)`

    See Also
    --------
    ekore.harmonics.g_functions.mellin_g22 : :math:`g_22(N)`
    """
    return (
        1 / 2 * gf.mellin_g22(N, S1, S2, S3)
        - 1 / 4 * S4
        - 1 / 4 * S2**2
        + zeta2 * S2
        - 3 / 20 * zeta2**2
    )
