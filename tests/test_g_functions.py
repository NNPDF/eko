# -*- coding: utf-8 -*-
# Test G functions impleeted by muselli PhD

import numpy as np

from eko.anomalous_dimensions import harmonics
import eko.matching_conditions.n3lo.g_functions as gf
import eko.matching_conditions.n3lo.s_functions as sf

zeta3 = harmonics.zeta3
log2 = np.log(2)


testN = [1, 10, 100]


# copare the exact values of some harmonics with Muselli parametrisations
def test_g4():
    refvals = [-1, -1.34359, -1.40286]
    for N, vals in zip(testN, refvals):
        S2 = harmonics.harmonic_S2(N)
        Sm1 = sf.harmonic_Sm1(N)
        Sm2 = sf.harmonic_Sm2(N)
        S2m1 = sf.harmonic_S2m1(N, S2, Sm1, Sm2)
        np.testing.assert_allclose(S2m1, vals, atol=1e-05)


def test_g6():
    refvals = [-1, -0.857976, -0.859245]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.harmonic_Sm1(N)
        Sm2 = sf.harmonic_Sm2(N)
        Sm31 = sf.harmonic_Sm31(N, Sm1, Sm2)
        np.testing.assert_allclose(Sm31, vals, atol=1e-05)


def test_g5():
    refvals = [-1, -0.777375, -0.784297]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.harmonic_Sm1(N)
        Sm2 = sf.harmonic_Sm2(N)
        Sm31 = sf.harmonic_Sm31(N, Sm1, Sm2)
        Sm22 = sf.harmonic_Sm22(N, Sm31)
        np.testing.assert_allclose(Sm22, vals, atol=1e-05)


def test_g8():
    refvals = [-1, -0.696836, -0.719637]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.harmonic_Sm1(N)
        Sm211 = sf.harmonic_Sm211(N, Sm1)
        np.testing.assert_allclose(Sm211, vals, atol=1e-05)


def test_g18():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.375, 1.5787, 2.0279, 2.34252]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S21 = sf.harmonic_S21(N, S1, S2)
        np.testing.assert_allclose(S21, vals, atol=1e-05)


def test_g19():
    refvals = [1, 0.953673, 0.958928]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        Sm2 = sf.harmonic_Sm2(N)
        Sm2m1 = sf.harmonic_Sm2m1(N, S1, S2, Sm2)
        np.testing.assert_allclose(Sm2m1, vals, atol=1e-05)


def test_g21():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.4375, 1.69985, 2.38081, 3.04323]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        S211 = sf.harmonic_S211(N, S1, S2, S3)
        np.testing.assert_allclose(S211, vals, atol=1e-05)


def test_g22():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.1875, 1.2554, 1.33724, 1.35262]
    for N, vals in zip(testN, refvals):
        S2 = harmonics.harmonic_S2(N)
        S4 = harmonics.harmonic_S4(N)
        S31 = sf.harmonic_S31(N, S2, S4)
        np.testing.assert_allclose(S31, vals, atol=1e-05)
