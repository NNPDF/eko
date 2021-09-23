# -*- coding: utf-8 -*-
"""
This module contains some additional harmonics sum.
Defintion are coming from :cite:`MuselliPhD,Bl_mlein_2009`
"""
import numpy as np
import numba as nb

from eko.anomalous_dimensions import harmonics
from . import g_functions as gf

zeta2 = harmonics.zeta2
zeta3 = harmonics.zeta3

li4half = 0.517479
log2 = np.log(2)


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm1(N, Sp1):
    return (-1) ** N / 2 * (Sp1 - harmonics.harmonic_S1((N - 1) / 2)) - log2


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm2(N, Sp2):
    return (-1) ** N / 4 * (Sp2 - harmonics.harmonic_S2((N - 1) / 2)) - zeta2 / 2


@nb.njit("c16(c16,c16)", cache=True)
def harmonic_Sm3(N, Sp3):
    return (-1) ** N / 8 * (Sp3 - harmonics.harmonic_S3((N - 1) / 2)) - 3 / 4 * zeta3


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


# def harmonic_Sm31(N,Sm1,Sm2):
#     return (
#         (-1) ** N * gf.mellin_g6(N)
#         + zeta2 * Sm2
#         - zeta3 * Sm1
#         - 3 / 5 * zeta2 ** 2
#         + 2 * li4half
#         + 3 / 4 * zeta3 * log2
#         - 1 / 2 * zeta2 * log2 ** 2
#         + 1 / 12 * log2 ** 4
#     )
