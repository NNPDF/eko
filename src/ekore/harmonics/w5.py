"""Weight 5 harmonic sums."""

import numba as nb

from . import f_functions as f
from .constants import log2, zeta2, zeta3, zeta4, zeta5
from .polygamma import cern_polygamma, symmetry_factor


@nb.njit(cache=True)
def S5(N):
    r"""Computes the harmonic sum :math:`S_5(N)`.

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
def Sm5(N, hS5, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-5}(N)`.

    .. math::
      S_{-5}(N) = \sum\limits_{j=1}^N \frac {(-1)^j} {j^5}

    Parameters
    ----------
    N : complex
        Mellin moment
    hS5:  complex
        Harmonic sum :math:`S_{5}(N)`
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
            1
            / 2**4
            * ((1 - (-1) ** N) / 2 * S5((N - 1) / 2) + ((-1) ** N + 1) / 2 * S5(N / 2))
            - hS5
        )
    if is_singlet:
        return 1 / 2**4 * S5(N / 2) - hS5
    return 1 / 2**4 * S5((N - 1) / 2) - hS5


@nb.njit(cache=True)
def S41(N, S1, S2, S3):
    r"""Analytic continuation of harmonic sum :math:`S_{4,1}(N)`.

    As implemented in eq 9.1 of :cite:`Blumlein:2009ta`.

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
    S41 : complex
        Harmonic sum :math:`S_{4,1}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F9 :
        :math:`\mathcal{M}[(\text{Li}_4(x)/(x-1))_{+}](N)`

    """
    return -f.F9(N, S1) + S1 * zeta4 - S2 * zeta3 + S3 * zeta2


@nb.njit(cache=True)
def S311(N, S1, S2):
    r"""Analytic continuation of harmonic sum :math:`S_{3,1,1}(N)`.

    As implemented in eq 9.21 of :cite:`Blumlein:2009ta`.

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
    S311 : complex
        Harmonic sum :math:`S_{3,1,1}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F11 :
        :math:`\mathcal{M}[(\text{S}_{2,2}(x)/(x-1))_{+}](N)`

    """
    return f.F11(N, S1, S2) + zeta3 * S2 - zeta4 / 4 * S1


@nb.njit(cache=True)
def S221(N, S1, S2, S21):
    r"""Analytic continuation of harmonic sum :math:`S_{2,2,1}(N)`.

    As implemented in eq 9.23 of :cite:`Blumlein:2009ta`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    S2: complex
        Harmonic sum :math:`S_{2}(N)`
    S21: complex
        Harmonic sum :math:`S_{2,1}(N)`

    Returns
    -------
    S221 : complex
        Harmonic sum :math:`S_{2,2,1}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F11 :
        :math:`\mathcal{M}[(\text{S}_{2,2}(x)/(x-1))_{+}](N)`
    eko.harmonic.f_functions.F13 :
        :math:`\mathcal{M}[(\text{Li}_{2}^2(x)/(x-1))_{+}](N)`

    """
    return (
        -2 * f.F11(N, S1, S2)
        + 1 / 2 * f.F13(N, S1, S2)
        + zeta2 * S21
        - 3 / 10 * zeta2**2 * S1
    )


@nb.njit(cache=True)
def Sm221(N, S1, Sm1, S21, Sm21):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,2,1}(N)`.

    As implemented in eq 9.25 of :cite:`Blumlein:2009ta`.

    Parameters
    ----------
    N : complex
        Mellin moment
    S1: complex
        Harmonic sum :math:`S_{1}(N)`
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`
    S21: complex
        Harmonic sum :math:`S_{2,1}(N)`
    Sm21: complex
        Harmonic sum :math:`S_{-2,1}(N)`

    Returns
    -------
    Sm221 : complex
        Harmonic sum :math:`S_{-2,2,1}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F14F12 :
        :math:`\mathcal{M}[(2 \text{S}_{2,2}(x) - 1/2 \text{Li}_{2}^2(x))/(x+1)](N)`

    """
    return (
        (-1) ** (N + 1) * (f.F14F12(N, S1, S21))
        + zeta2 * Sm21
        - 3 / 10 * zeta2**2 * Sm1
        - 0.119102
        + 0.0251709
    )


@nb.njit(cache=True)
def S21m2(N, S1, S2, Sm1, Sm2, Sm3, S21, Sm21, S2m1):
    r"""Analytic continuation of harmonic sum :math:`S_{2,1,-2}(N)`.

    As implemented in eq 9.26 of :cite:`Blumlein:2009ta`.

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
    Sm2: complex
        Harmonic sum :math:`S_{-2}(N)`
    Sm3: complex
        Harmonic sum :math:`S_{-3}(N)`
    S21: complex
        Harmonic sum :math:`S_{2,1}(N)`
    Sm21: complex
        Harmonic sum :math:`S_{-2,1}(N)`
    S2m1: complex
        Harmonic sum :math:`S_{2,-1}(N)`

    Returns
    -------
    S21m2 : complex
        Harmonic sum :math:`S_{2,1,-2}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F14F12 :
        :math:`\mathcal{M}[
            (\text{ln}(x) \text{S}_{1,2}(−x) − \text{Li}_2^2(−x)/2)/(x+1)
            ](N)`

    """
    return (
        (-1) ** (N) * f.F16(N, S1, Sm1, Sm2, Sm3, Sm21)
        - 1 / 2 * zeta2 * (S21 - S2m1)
        - (1 / 8 * zeta3 - 1 / 2 * zeta2 * log2) * (S2 - Sm2)
        + 1 / 8 * zeta2**2 * Sm1
        + 0.0854806
    )


@nb.njit(cache=True)
def S2111(N, S1, S2, S3):
    r"""Analytic continuation of harmonic sum :math:`S_{2,1,1,1}(N)`.

    As implemented in eq 9.33 of :cite:`Blumlein:2009ta`.

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
    S2111 : complex
        Harmonic sum :math:`S_{2,1,1,1}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F17 :
        :math:`\mathcal{M}[(\text{S}_{1,3}(x)/(x-1))_{+}](N)`

    """
    return -f.F17(N, S1, S2, S3) + zeta4 * S1


@nb.njit(cache=True)
def Sm2111(N, S1, S2, S3, Sm1):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,1,1,1}(N)`.

    As implemented in eq 9.34 of :cite:`Blumlein:2009ta`.

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
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`

    Returns
    -------
    Sm2111 : complex
        Harmonic sum :math:`S_{-2,1,1,1}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F18 :
        :math:`\mathcal{M}[\text{S}_{1,3}(x)/(x+1)](N)`

    """
    return (
        (-1) ** (N + 1) * f.F18(N, S1, S2, S3)
        + zeta4 * Sm1
        - 0.706186
        + 0.693147 * zeta4
    )


@nb.njit(cache=True)
def S23(N, S1, S2, S3):
    r"""Analytic continuation of harmonic sum :math:`S_{2,3}(N)`.

    As implemented in eq 9.3 of :cite:`Blumlein:2009ta`.

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
    S23 : complex
        Harmonic sum :math:`S_{2,3}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F19 :
        :math:`\mathcal{M}[
            ((
                \text{ln}(x)[\text{S}_{1,2}(1-x) - \zeta_3]
                +3 [\text{S}_{1,3}(1-x) - \zeta_4]
            /(x-1))_{+}](N)`

    """
    return f.F19(N, S1, S2, S3) + 3 * zeta4 * S1


@nb.njit(cache=True)
def Sm23(N, Sm1, Sm2, Sm3, is_singlet=None):
    r"""Analytic continuation of harmonic sum :math:`S_{-2,3}(N)`.

    As implemented in eq 9.4 of :cite:`Blumlein:2009ta`.

    Parameters
    ----------
    N : complex
        Mellin moment
    Sm1: complex
        Harmonic sum :math:`S_{-1}(N)`
    Sm2: complex
        Harmonic sum :math:`S_{-2}(N)`
    Sm3: complex
        Harmonic sum :math:`S_{-3}(N)`
    is_singlet: bool, None
        symmetry factor: True for singlet like quantities (:math:`\eta=(-1)^N =
        1`), False for non-singlet like quantities (:math:`\eta=(-1)^N=-1`)

    Returns
    -------
    Sm23 : complex
        Harmonic sum :math:`S_{-2,3}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F19 :
        :math:`\mathcal{M}[
            (
                \text{ln}(x)[\text{S}_{1,2}(1-x) - \zeta_3]
                +3 [\text{S}_{1,3}(1-x) - \zeta_4]
            /(x+1)](N)`

    """
    eta = symmetry_factor(N, is_singlet)
    return (
        eta * f.F20(N, Sm1, Sm2, Sm3)
        + 3 * zeta4 * Sm1
        + 21 / 32 * zeta5
        + 3 * zeta4 * log2
        - 3 / 4 * zeta2 * zeta3
    )


@nb.njit(cache=True)
def S2m3(N, S2, Sm1, Sm2, Sm3):
    r"""Analytic continuation of harmonic sum :math:`S_{2,-3}(N)`.

    As implemented in eq 9.5 of :cite:`Blumlein:2009ta`.

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
    Sm3: complex
        Harmonic sum :math:`S_{-3}(N)`

    Returns
    -------
    S2m3 : complex
        Harmonic sum :math:`S_{2,-3}(N)`

    See Also
    --------
    eko.harmonic.f_functions.F19 :
        :math:`\mathcal{M}[
            ((
                1/2 \text{ln}^2(x) \text{Li}_2(-x)
                -2  \text{ln}(x) \text{Li}_3(-x)
                +3 \text{Li}_4(-x)
            )/(x-1)](N)`

    """
    return (
        (-1) ** (N + 1) * f.F21(N, Sm1, Sm2, Sm3)
        + 3 / 4 * zeta3 * (Sm2 - S2)
        - 21 / 8 * zeta4 * Sm1
        - 1.32056
    )
