# -*- coding: utf-8 -*-
"""
This module contains some additional harmonics sum.
Defintion are coming from :cite:`MuselliPhD,Bl_mlein_2009`
"""
import numpy as np
import numba as nb
import mpmath as mp 

from eko.anomalous_dimensions import harmonics
from . import g_functions as gf
from . import f_functions as f

zeta2 = harmonics.zeta2
zeta3 = harmonics.zeta3
zeta4 = harmonics.zeta4
zeta5 = harmonics.zeta5

li4half = 0.517479
log2 = np.log(2)

# @nb.njit("c16(c16,c16)", cache=True)
def binomial(x, y):
    """Binomial coefficient for complex numbers"""
    # return sp.gamma(x + 1) / (sp.gamma(y + 1) * sp.gamma(x - y + 1))
    # more accurate for large numbers.
    return complex(1/((x +1) * mp.beta(x-y+1,y+1)))


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm1(N):
    return (-1) ** N / 2 * (
        harmonics.harmonic_S1(N / 2) - harmonics.harmonic_S1((N - 1) / 2)
    ) - log2


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm2(N):
    return (-1) ** N / 4 * (
        harmonics.harmonic_S2(N / 2) - harmonics.harmonic_S2((N - 1) / 2)
    ) - zeta2 / 2


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm3(N):
    return (-1) ** N / 8 * (
        harmonics.harmonic_S3(N / 2) - harmonics.harmonic_S3((N - 1) / 2)
    ) - 3 / 4 * zeta3


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm4(N):
    return (-1) ** N / 16 * (
        harmonics.harmonic_S4(N / 2) - harmonics.harmonic_S4((N - 1) / 2)
    ) - 7 / 8 * zeta4


@nb.njit("c16(c16)", cache=True)
def harmonic_Sm5(N):
    return (-1) ** N / 32 * (
        harmonics.harmonic_S5(N / 2) - harmonics.harmonic_S5((N - 1) / 2)
    ) - 15 / 16 * zeta5


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_S21(N, S1, S2):
    return -gf.mellin_g18(N, S1, S2) + 2 * zeta3


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm21(N, Sm1):
    # Note Mellin G3 was integrated following x^(N-1) convention.
    return (
        -((-1) ** N) * harmonics.mellin_g3(N + 1)
        + zeta2 * Sm1
        - 5 / 8 * zeta3
        + zeta2 * log2
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S2m1(N, S2, Sm1, Sm2):
    return (
        -((-1) ** N) * gf.mellin_g4(N)
        - np.log(2) * (S2 - Sm2)
        - 1 / 2 * zeta2 * Sm1
        + 1 / 4 * zeta3
        - 1 / 2 * zeta2 * log2
    )


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_Sm31(N, Sm1, Sm2):
    return (
        (-1) ** N * gf.mellin_g6(N)
        + zeta2 * Sm2
        - zeta3 * Sm1
        - 3 / 5 * zeta2 ** 2
        + 2 * li4half
        + 3 / 4 * zeta3 * log2
        - 1 / 2 * zeta2 * log2 ** 2
        + 1 / 12 * log2 ** 4
    )


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm22(N, Sm31):
    return (
        (-1) ** N * gf.mellin_g5(N)
        - 2 * Sm31
        + 2 * zeta2 * harmonic_Sm2(N)
        + 3 / 40 * zeta2 ** 2
    )


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm211(N, Sm1):
    return (
        -((-1) ** N) * gf.mellin_g8(N)
        + zeta3 * Sm1
        - li4half
        + 1 / 8 * zeta2 ** 2
        + 1 / 8 * zeta3 * log2
        + 1 / 4 * zeta2 * log2 ** 2
        - 1 / 24 * log2 ** 4
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_Sm2m1(N, S1, S2, Sm2):
    return -gf.mellin_g19(N, S1) + log2 * (S2 - Sm2) - 5 / 8 * zeta3


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S211(N, S1, S2, S3):
    return -gf.mellin_g21(N, S1, S2, S3) + 6 / 5 * zeta2 ** 2


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_S31(N, S2, S4):
    return (
        1 / 2 * gf.mellin_g22(N)
        - 1 / 4 * S4
        - 1 / 4 * S2 ** 2
        + zeta2 * S2
        - 3 / 20 * zeta2 ** 2
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S41(N, S1, S2, S3):
    return -f.F9(N, S1) + S1 * zeta4 - S2 * zeta3 + S3 * zeta2


@nb.njit("c16(c16,c16,c16)", cache=True)
def harmonic_S311(N, S1, S2):
    return f.F11(N, S1, S2) + zeta3 * S2 - zeta4 / 4 * S1


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S221(N, S1, S2, S21):
    return (
        -2 * f.F11(N, S1, S2)
        + 1 / 2 * f.F13(N, S1, S2)
        + zeta2 * S21
        - 3 / 10 * zeta2 ** 2 * S1
    )


@nb.njit("c16(c16,c16,c16,c16,c16)", cache=True)
def harmonic_Sm221(N, S1, Sm1, S21, Sm21):
    return (
        (-1) ** (N + 1) * (f.F14F12(N, S1, S21))
        + zeta2 * Sm21
        - 3 / 10 * zeta2 ** 2 * Sm1
        - 0.119102
        + 0.0251709
    )


@nb.njit("c16(c16,c16,c16,c16,c16,c16,c16,c16,c16)", cache=True)
def harmonic_S21m2(N, S1, S2, Sm1, Sm2, Sm3, S21, Sm21, S2m1):
    return (
        (-1) ** (N) * f.F16(N, S1, Sm1, Sm2, Sm3, Sm21)
        - 1 / 2 * zeta2 * (S21 - S2m1)
        - (1 / 8 * zeta3 - 1 / 2 * zeta2 * log2) * (S2 - Sm2)
        + 1 / 8 * zeta2 ** 2 * Sm1
        + 0.0854806
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S2111(N, S1, S2, S3):
    return -f.F17(N, S1, S2, S3) + zeta4 * S1


@nb.njit("c16(c16,c16,c16,c16,c16)", cache=True)
def harmonic_Sm2111(N, S1, S2, S3, Sm1):
    return (
        (-1) ** (N + 1) * f.F18(N, S1, S2, S3)
        + zeta4 * Sm1
        - 0.706186
        + 0.693147 * zeta4
    )


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_S23(N, S1, S2, S3):
    return f.F19(N, S1, S2, S3) + 3 * zeta4 * S1


@nb.njit("c16(c16,c16,c16,c16)", cache=True)
def harmonic_Sm23(N, Sm1, Sm2, Sm3):
    return (
        (-1) ** N * f.F20(N, Sm1, Sm2, Sm3)
        + 3 * zeta4 * Sm1
        + 21 / 32 * zeta5
        + 3 * zeta4 * log2
        - 3 / 4 * zeta2 * zeta3
    )


@nb.njit("c16(c16,c16,c16,c16,c16)", cache=True)
def harmonic_S2m3(N, S2, Sm1, Sm2, Sm3):
    return (
        (-1) ** (N + 1) * f.F21(N, Sm1, Sm2, Sm3)
        + 3 / 4 * zeta3 * (Sm2 - S2)
        - 21 / 8 * zeta4 * Sm1
        - 1.32056
    )
