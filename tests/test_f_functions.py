# -*- coding: utf-8 -*-
# Test G functions impleeted by muselli PhD

import numpy as np

from eko.anomalous_dimensions import harmonics
import eko.matching_conditions.n3lo.f_functions as f
import eko.matching_conditions.n3lo.s_functions as sf

zeta2 = harmonics.zeta2
zeta3 = harmonics.zeta3
zeta4 = harmonics.zeta4

log2 = np.log(2)

# reference values coming fom mathematica
testN = [1, 2, 2 + 2j, 10 + 5j, 100]
refvals = {
    "S41": [1.0, 1.09375, 1.13674 + 0.0223259j, 1.13322 + 0.000675408j, 1.13348],
    "S21m2": [
        -1.0 + 2.32792e-17,
        -1.34375 - 5.19668e-18j,
        -1.55003 - 0.349276j,
        -1.96639 - 0.100789j,
        -2.19769,
    ],
    "S221": [1, 1.34375, 1.5542 + 0.326975j, 1.89896 + 0.0770909j, 2.05026],
    "Sm221": [
        -1.0,
        -0.65625,
        -0.767023 - 0.000133792j,
        -0.767089 - 8.56713e-10j,
        -0.766973,
    ],
    "S311": [1.0, 1.21875, 1.35078 + 0.142143j, 1.43773 + 0.0168713j, 1.45799],
    "S2111": [1.0, 1.46875, 1.71628 + 0.591696j, 2.70015 + 0.291842j, 3.66588],
    "Sm2111": [
        -1.0,
        -0.53125,
        -0.705997 - 0.000177561j,
        -0.706186 - 2.32698e-9j,
        -0.704801,
    ],
    'Sm21': [-1., -0.625, -0.751192 - 0.000147181j, -0.751286 - 1.17067e-9j, -0.751029]    
}


# copare the exact values of some harmonics
# All the harmonics defition are coming from :cite`:Bl_mlein_2009` section 9.
def test_F9():
    for N, vals in zip(testN, refvals["S41"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        S41 = -f.F9(N, S1) + S1 * zeta4 - S2 * zeta3 + S3 * zeta2
        np.testing.assert_allclose(S41, vals, atol=1e-05)


def test_F11():
    for N, vals in zip(testN, refvals["S311"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S311 = f.F11(N, S1, S2) + zeta3 * S2 - zeta4 / 4 * S1
        np.testing.assert_allclose(S311, vals, atol=1e-05)


def test_F13():
    for N, vals in zip(testN, refvals["S221"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S21 = sf.harmonic_S21(N, S1, S2)
        S221 = (
            -2 * f.F11(N, S1, S2)
            + 1 / 2 * f.F13(N, S1, S2)
            + zeta2 * S21
            - 3 / 10 * zeta2 ** 2 * S1
        )
        np.testing.assert_allclose(S221, vals, atol=1e-05)


def test_F12_F14():
    # here there is a typo in eq (9.25) of Bl_mlein_2009
    for N, vals in zip(testN, refvals["Sm221"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        Sm1 = sf.harmonic_Sm1(N, harmonics.harmonic_S1(N / 2))
        S21 = sf.harmonic_S21(N, S1, S2)
        Sm21 = sf.harmonic_Sm21(N, Sm1)
        Sm221 = (
            (-1) ** (N + 1) * (2 * f.F12(N, S1, S2) - 1 / 2 * f.F14(N, S1, S2, S21))
            + zeta2 * Sm21
            - 3 / 10 * zeta2 ** 2 * Sm1
            - 0.119102 + 0.0251709
        )
        np.testing.assert_allclose(Sm221, vals, atol=1e-05)

def test_F16():
    for N, vals in zip(testN, refvals["S21m2"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        Sm1 = sf.harmonic_Sm1(N, harmonics.harmonic_S1(N / 2))
        Sm2 = sf.harmonic_Sm2(N, harmonics.harmonic_S2(N / 2))
        Sm3 = sf.harmonic_Sm3(N, harmonics.harmonic_S3(N / 2))
        S21 = sf.harmonic_S21(N, S1, S2)
        S2m1 = sf.harmonic_S2m1(N, S2, Sm1, Sm2)
        Sm21 = sf.harmonic_Sm21(N, Sm1)
        S21m2 = (
            (-1) ** (N) * f.F16(N, S1, Sm1, Sm2, Sm3, Sm21)
            - 1 / 2 * zeta2 * (S21 - S2m1)
            - (1 / 8 * zeta3 - 1 / 2 * zeta2 * log2) * (S2 - Sm2)
            + 1 / 8 * zeta2 ** 2 * Sm1
            + 0.0854806
        )
        np.testing.assert_allclose(S21m2, vals, atol=1e-04)

def test_F17():
    for N, vals in zip(testN, refvals["S2111"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        S2111 = -f.F17(N, S1, S2, S3) + zeta4 * S1
        np.testing.assert_allclose(S2111, vals, atol=1e-05)


def test_F18():
    for N, vals in zip(testN, refvals["Sm2111"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        Sm1 = sf.harmonic_Sm1(N, harmonics.harmonic_S1(N / 2))
        Sm2111 = (
            (-1) ** (N + 1) * f.F18(N, S1, S2, S3)
            + zeta4 * Sm1
            - 0.706186
            + 0.693147 * zeta4
        )
        np.testing.assert_allclose(Sm2111, vals, atol=1e-05)

def test_Sm21():
    for N, vals in zip(testN, refvals["Sm21"]):
        Sm1 = sf.harmonic_Sm1(N,harmonics.harmonic_S1(N / 2))
        np.testing.assert_allclose(sf.harmonic_Sm21(N, Sm1), vals, atol=1e-05)
