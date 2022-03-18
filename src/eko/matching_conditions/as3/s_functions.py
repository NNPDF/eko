# -*- coding: utf-8 -*-
"""
This module contains some additional harmonics sum.
Defintion are coming from :cite:`MuselliPhD,Bl_mlein_2000,Blumlein:2009ta`
"""
import numba as nb
import numpy as np

from eko.anomalous_dimensions import harmonics

from . import f_functions as f
from . import g_functions as gf

zeta2 = harmonics.zeta2
zeta3 = harmonics.zeta3
zeta4 = harmonics.zeta4
zeta5 = harmonics.zeta5

li4half = 0.517479
log2 = np.log(2)


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm1(N):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-1}(N)`.

    .. math::
      S_{-1}(N) = \sum\limits_{j=1}^N \frac (-1)^j j

    Parameters
    ----------
        N : complex
            Mellin moment

    Returns
    -------
        Sm1 : complex
            Harmonic sum :math:`S_{-1}(N)`

    See Also
    --------
        eko.anomalous_dimension.harmonics.harmonic_S1 : :math:`S_1(N)`
    """
    return (-1) ** N / 2 * (
        harmonics.harmonic_S1(N / 2) - harmonics.harmonic_S1((N - 1) / 2)
    ) - log2


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm2(N):
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
        eko.anomalous_dimension.harmonics.harmonic_S2 : :math:`S_2(N)`
    """
    return (-1) ** N / 4 * (
        harmonics.harmonic_S2(N / 2) - harmonics.harmonic_S2((N - 1) / 2)
    ) - zeta2 / 2


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm3(N):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-3}(N)`.

    .. math::
      S_{-3}(N) = \sum\limits_{j=1}^N \frac (-1)^j j^3

    Parameters
    ----------
        N : complex
            Mellin moment

    Returns
    -------
        Sm3 : complex
            Harmonic sum :math:`S_{-3}(N)`

    See Also
    --------
        eko.anomalous_dimension.harmonics.harmonic_S3 : :math:`S_3(N)`
    """
    return (-1) ** N / 8 * (
        harmonics.harmonic_S3(N / 2) - harmonics.harmonic_S3((N - 1) / 2)
    ) - 3 / 4 * zeta3


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm4(N):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-4}(N)`.

    .. math::
      S_{-4}(N) = \sum\limits_{j=1}^N \frac (-1)^j j^4

    Parameters
    ----------
        N : complex
            Mellin moment

    Returns
    -------
        Sm4 : complex
            Harmonic sum :math:`S_{-4}(N)`

    See Also
    --------
        eko.anomalous_dimension.harmonics.harmonic_S4 : :math:`S_4(N)`
    """
    return (-1) ** N / 16 * (
        harmonics.harmonic_S4(N / 2) - harmonics.harmonic_S4((N - 1) / 2)
    ) - 7 / 8 * zeta4


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm5(N):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-5}(N)`.

    .. math::
      S_{-5}(N) = \sum\limits_{j=1}^N \frac (-1)^j j^5

    Parameters
    ----------
        N : complex
            Mellin moment

    Returns
    -------
        Sm5 : complex
            Harmonic sum :math:`S_{-5}(N)`

    See Also
    --------
        eko.anomalous_dimension.harmonics.harmonic_S5 : :math:`S_5(N)`
    """
    return (-1) ** N / 32 * (
        harmonics.harmonic_S5(N / 2) - harmonics.harmonic_S5((N - 1) / 2)
    ) - 15 / 16 * zeta5


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_S21(N, S1, S2):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,1}(N)`
    as implemented in eq B.5.77 of :cite:`MuselliPhD` and eq 37 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`

    Returns
    -------
        S21 : complex
            Harmonic sum :math:`S_{2,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g18 : :math:`g_18(N)`
    """
    return -gf.mellin_g18(N, S1, S2) + 2 * zeta3


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm21(N, Sm1):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,1}(N)`
    as implemented in eq B.5.75 of :cite:`MuselliPhD` and eq 22 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`

    Returns
    -------
        Sm21 : complex
            Harmonic sum :math:`S_{-2,1}(N)`

    See Also
    --------
        eko.anomalous_dimension.harmonics.melling_g3 : :math:`g_3(N)`
    """
    # Note mellin g3 was integrated following x^(N-1) convention.
    return (
        -((-1) ** N) * harmonics.mellin_g3(N + 1)
        + zeta2 * Sm1
        - 5 / 8 * zeta3
        + zeta2 * log2
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S2m1(N, S2, Sm1, Sm2):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,-1}(N)`
    as implemented in eq B.5.76 of :cite:`MuselliPhD` and eq 23 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`
        Sm2: complex
            Hamrmonic sum :math:`S_{-2}(N)`

    Returns
    -------
        S2m1 : complex
            Harmonic sum :math:`S_{2,-1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g4 : :math:`g_4(N)`
    """
    return (
        -((-1) ** N) * gf.mellin_g4(N)
        - np.log(2) * (S2 - Sm2)
        - 1 / 2 * zeta2 * Sm1
        + 1 / 4 * zeta3
        - 1 / 2 * zeta2 * log2
    )


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_Sm31(N, Sm1, Sm2):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-3,1}(N)`
    as implemented in eq B.5.93 of :cite:`MuselliPhD` and eq 25 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`
        Sm2: complex
            Hamrmonic sum :math:`S_{-2}(N)`

    Returns
    -------
        Sm31 : complex
            Harmonic sum :math:`S_{-3,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g6 : :math:`g_6(N)`
    """
    return (
        (-1) ** N * gf.mellin_g6(N)
        + zeta2 * Sm2
        - zeta3 * Sm1
        - 3 / 5 * zeta2**2
        + 2 * li4half
        + 3 / 4 * zeta3 * log2
        - 1 / 2 * zeta2 * log2**2
        + 1 / 12 * log2**4
    )


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm22(N, Sm31):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,2}(N)`
    as implemented in eq B.5.94 of :cite:`MuselliPhD` and eq 24 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        Sm31: complex
            Hamrmonic sum :math:`S_{-3,1}(N)`
    Returns
    -------
        Sm22 : complex
            Harmonic sum :math:`S_{-2,2}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g5 : :math:`g_5(N)`
    """
    return (
        (-1) ** N * gf.mellin_g5(N)
        - 2 * Sm31
        + 2 * zeta2 * harmonic_Sm2(N)
        + 3 / 40 * zeta2**2
    )


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm211(N, Sm1):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,1,1}(N)`
    as implemented in eq B.5.104 of :cite:`MuselliPhD` and eq 27 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`

    Returns
    -------
        Sm31 : complex
            Harmonic sum :math:`S_{-2,1,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g8 : :math:`g_8(N)`
    """
    return (
        -((-1) ** N) * gf.mellin_g8(N)
        + zeta3 * Sm1
        - li4half
        + 1 / 8 * zeta2**2
        + 1 / 8 * zeta3 * log2
        + 1 / 4 * zeta2 * log2**2
        - 1 / 24 * log2**4
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_Sm2m1(N, S1, S2, Sm2):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,-1}(N)`
    as implemented in eq B.5.74 of :cite:`MuselliPhD` and eq 38 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        Sm2: complex
            Hamrmonic sum :math:`S_{-2}(N)`

    Returns
    -------
        Sm2m1 : complex
            Harmonic sum :math:`S_{-2,-1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g19 : :math:`g_19(N)`
    """
    return -gf.mellin_g19(N, S1) + log2 * (S2 - Sm2) - 5 / 8 * zeta3


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S211(N, S1, S2, S3):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,1,1}(N)`
    as implemented in eq B.5.115 of :cite:`MuselliPhD` and eq 40 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S3: complex
            Hamrmonic sum :math:`S_{3}(N)`

    Returns
    -------
        S211 : complex
            Harmonic sum :math:`S_{2,1,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g21 : :math:`g_21(N)`
    """
    return -gf.mellin_g21(N, S1, S2, S3) + 6 / 5 * zeta2**2


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_S31(N, S2, S4):
    r"""
    Analytic continuation of harmonic sum :math:`S_{3,1}(N)`
    as implemented in eq B.5.99 of :cite:`MuselliPhD` and eq 41 of cite:`Bl_mlein_2000`.

    Parameters
    ----------
        N : complex
            Mellin moment
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S4: complex
            Hamrmonic sum :math:`S_{4}(N)`

    Returns
    -------
        S31 : complex
            Harmonic sum :math:`S_{3,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.g_functions.mellin_g22 : :math:`g_22(N)`
    """
    return (
        1 / 2 * gf.mellin_g22(N)
        - 1 / 4 * S4
        - 1 / 4 * S2**2
        + zeta2 * S2
        - 3 / 20 * zeta2**2
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S41(N, S1, S2, S3):
    r"""
    Analytic continuation of harmonic sum :math:`S_{4,1}(N)`
    as implemented in eq 9.1 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S3: complex
            Hamrmonic sum :math:`S_{3}(N)`

    Returns
    -------
        S41 : complex
            Harmonic sum :math:`S_{4,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F9 :
            :math:`\mathcal{M}[(\text{Li}_4(x)/(x-1))_{+}](N)`
    """
    return -f.F9(N, S1) + S1 * zeta4 - S2 * zeta3 + S3 * zeta2


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_S311(N, S1, S2):
    r"""
    Analytic continuation of harmonic sum :math:`S_{3,1,1}(N)`
    as implemented in eq 9.21 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`

    Returns
    -------
        S311 : complex
            Harmonic sum :math:`S_{3,1,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F11 :
            :math:`\mathcal{M}[(\text{S}_{2,2}(x)/(x-1))_{+}](N)`
    """
    return f.F11(N, S1, S2) + zeta3 * S2 - zeta4 / 4 * S1


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S221(N, S1, S2, S21):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,2,1}(N)`
    as implemented in eq 9.23 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S21: complex
            Hamrmonic sum :math:`S_{2,1}(N)`

    Returns
    -------
        S221 : complex
            Harmonic sum :math:`S_{2,2,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F11 :
            :math:`\mathcal{M}[(\text{S}_{2,2}(x)/(x-1))_{+}](N)`
        eko.matching_conditions.n3lo.f_functions.F13 :
            :math:`\mathcal{M}[(\text{Li}_{2}^2(x)/(x-1))_{+}](N)`
    """
    return (
        -2 * f.F11(N, S1, S2)
        + 1 / 2 * f.F13(N, S1, S2)
        + zeta2 * S21
        - 3 / 10 * zeta2**2 * S1
    )


@nb.njit("c16(c16,c16,c16,c16,c16)", cache=True)
def harmonic_Sm221(N, S1, Sm1, S21, Sm21):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,2,1}(N)`
    as implemented in eq 9.25 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`
        S21: complex
            Hamrmonic sum :math:`S_{2,1}(N)`
        Sm21: complex
            Hamrmonic sum :math:`S_{-2,1}(N)`

    Returns
    -------
        Sm221 : complex
            Harmonic sum :math:`S_{-2,2,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F14F12 :
            :math:`\mathcal{M}[(2 \text{S}_{2,2}(x) - 1/2 \text{Li}_{2}^2(x))/(x+1)](N)`
    """
    return (
        (-1) ** (N + 1) * (f.F14F12(N, S1, S21))
        + zeta2 * Sm21
        - 3 / 10 * zeta2**2 * Sm1
        - 0.119102
        + 0.0251709
    )


@nb.njit("c16(c16,c16,c16,c16,c16,c16,c16,c16,c16)", cache=True)
def harmonic_S21m2(N, S1, S2, Sm1, Sm2, Sm3, S21, Sm21, S2m1):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,1,-2}(N)`
    as implemented in eq 9.26 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`
        Sm2: complex
            Hamrmonic sum :math:`S_{-2}(N)`
        Sm3: complex
            Hamrmonic sum :math:`S_{-3}(N)`
        S21: complex
            Hamrmonic sum :math:`S_{2,1}(N)`
        Sm21: complex
            Hamrmonic sum :math:`S_{-2,1}(N)`
        S2m1: complex
            Hamrmonic sum :math:`S_{2,-1}(N)`

    Returns
    -------
        S21m2 : complex
            Harmonic sum :math:`S_{2,1,-2}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F14F12 :
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


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S2111(N, S1, S2, S3):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,1,1,1}(N)`
    as implemented in eq 9.33 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S3: complex
            Hamrmonic sum :math:`S_{3}(N)`

    Returns
    -------
        S2111 : complex
            Harmonic sum :math:`S_{2,1,1,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F17 :
            :math:`\mathcal{M}[(\text{S}_{1,3}(x)/(x-1))_{+}](N)`
    """
    return -f.F17(N, S1, S2, S3) + zeta4 * S1


@nb.njit("c16(c16,c16,c16,c16,c16)", cache=True)
def harmonic_Sm2111(N, S1, S2, S3, Sm1):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,1,1,1}(N)`
    as implemented in eq 9.34 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S3: complex
            Hamrmonic sum :math:`S_{3}(N)`
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`

    Returns
    -------
        Sm2111 : complex
            Harmonic sum :math:`S_{-2,1,1,1}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F18 :
            :math:`\mathcal{M}[\text{S}_{1,3}(x)/(x+1)](N)`
    """
    return (
        (-1) ** (N + 1) * f.F18(N, S1, S2, S3)
        + zeta4 * Sm1
        - 0.706186
        + 0.693147 * zeta4
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S23(N, S1, S2, S3):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,3}(N)`
    as implemented in eq 9.3 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S1: complex
            Hamrmonic sum :math:`S_{1}(N)`
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        S3: complex
            Hamrmonic sum :math:`S_{3}(N)`

    Returns
    -------
        S23 : complex
            Harmonic sum :math:`S_{2,3}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F19 :
            :math:`\mathcal{M}[
                ((
                    \text{ln}(x)[\text{S}_{1,2}(1-x) - \zeta_3]
                    +3 [\text{S}_{1,3}(1-x) - \zeta_4]
                /(x-1))_{+}](N)`
    """
    return f.F19(N, S1, S2, S3) + 3 * zeta4 * S1


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_Sm23(N, Sm1, Sm2, Sm3):
    r"""
    Analytic continuation of harmonic sum :math:`S_{-2,3}(N)`
    as implemented in eq 9.4 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`
        Sm2: complex
            Hamrmonic sum :math:`S_{-2}(N)`
        Sm3: complex
            Hamrmonic sum :math:`S_{-3}(N)`

    Returns
    -------
        Sm23 : complex
            Harmonic sum :math:`S_{-2,3}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F19 :
            :math:`\mathcal{M}[
                (
                    \text{ln}(x)[\text{S}_{1,2}(1-x) - \zeta_3]
                    +3 [\text{S}_{1,3}(1-x) - \zeta_4]
                /(x+1)](N)`
    """
    return (
        (-1) ** N * f.F20(N, Sm1, Sm2, Sm3)
        + 3 * zeta4 * Sm1
        + 21 / 32 * zeta5
        + 3 * zeta4 * log2
        - 3 / 4 * zeta2 * zeta3
    )


@nb.njit("c16(c16,c16,c16,c16,c16)", cache=True)
def harmonic_S2m3(N, S2, Sm1, Sm2, Sm3):
    r"""
    Analytic continuation of harmonic sum :math:`S_{2,-3}(N)`
    as implemented in eq 9.5 of cite:` Bl_mlein_2009`

    Parameters
    ----------
        N : complex
            Mellin moment
        S2: complex
            Hamrmonic sum :math:`S_{2}(N)`
        Sm1: complex
            Hamrmonic sum :math:`S_{-1}(N)`
        Sm2: complex
            Hamrmonic sum :math:`S_{-2}(N)`
        Sm3: complex
            Hamrmonic sum :math:`S_{-3}(N)`

    Returns
    -------
        S2m3 : complex
            Harmonic sum :math:`S_{2,-3}(N)`

    See Also
    --------
        eko.matching_conditions.n3lo.f_functions.F19 :
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
