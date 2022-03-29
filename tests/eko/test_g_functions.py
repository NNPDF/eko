# -*- coding: utf-8 -*-
# Test G functions implemented by Muselli PhD

import numpy as np

import eko.harmonics as sf
from eko.anomalous_dimensions import harmonics

zeta3 = harmonics.constants.zeta3
log2 = harmonics.constants.log2


testN = [1, 10, 100]

# compare the exact values of some harmonics with Muselli parametrization
def test_g3():
    ns = [1.0, 2.0, 1 + 1j]
    # NIntegrate[x^({1, 2, 1 + I} - 1) PolyLog[2, x]/(1 + x), {x, 0, 1}]
    mma_ref_values = [0.3888958462, 0.2560382207, 0.3049381491 - 0.1589060625j]
    for n, r in zip(ns, mma_ref_values):
        np.testing.assert_almost_equal(sf.g_functions.mellin_g3(n), r, decimal=6)


def test_g4():
    refvals = [-1, -1.34359, -1.40286]
    for N, vals in zip(testN, refvals):
        S2 = harmonics.S2(N)
        Sm1 = sf.Sm1(N)
        Sm2 = sf.Sm2(N)
        S2m1 = sf.S2m1(N, S2, Sm1, Sm2)
        np.testing.assert_allclose(S2m1, vals, atol=1e-05)


def test_g6():
    refvals = [-1, -0.857976, -0.859245]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.Sm1(N)
        Sm2 = sf.Sm2(N)
        Sm31 = sf.Sm31(N, Sm1, Sm2)
        np.testing.assert_allclose(Sm31, vals, atol=1e-05)


def test_g5():
    refvals = [-1, -0.777375, -0.784297]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.Sm1(N)
        Sm2 = sf.Sm2(N)
        Sm31 = sf.Sm31(N, Sm1, Sm2)
        Sm22 = sf.Sm22(N, Sm2, Sm31)
        np.testing.assert_allclose(Sm22, vals, atol=1e-05)


def test_g8():
    refvals = [-1, -0.696836, -0.719637]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.Sm1(N)
        Sm211 = sf.Sm211(N, Sm1)
        np.testing.assert_allclose(Sm211, vals, atol=1e-05)


def test_g18():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.375, 1.5787, 2.0279, 2.34252]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.S1(N)
        S2 = harmonics.S2(N)
        S21 = sf.S21(N, S1, S2)
        np.testing.assert_allclose(S21, vals, atol=1e-05)


def test_g19():
    refvals = [1, 0.953673, 0.958928]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.S1(N)
        S2 = harmonics.S2(N)
        Sm2 = sf.Sm2(N)
        Sm2m1 = sf.Sm2m1(N, S1, S2, Sm2)
        np.testing.assert_allclose(Sm2m1, vals, atol=1e-05)


def test_g21():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.4375, 1.69985, 2.38081, 3.04323]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.S1(N)
        S2 = harmonics.S2(N)
        S3 = harmonics.S3(N)
        S211 = sf.S211(N, S1, S2, S3)
        np.testing.assert_allclose(S211, vals, atol=1e-05)


def test_g22():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.1875, 1.2554, 1.33724, 1.35262]
    for N, vals in zip(testN, refvals):
        S2 = harmonics.S2(N)
        S4 = harmonics.S4(N)
        S31 = sf.S31(N, S2, S4)
        np.testing.assert_allclose(S31, vals, atol=1e-05)
