# Test G functions implemented by Muselli PhD

import numpy as np

from eko.anomalous_dimensions import harmonics as h

zeta3 = h.constants.zeta3
log2 = h.constants.log2


# Reference values comes form Mathematica, and are
# obtained using the function S of HarmonicSums package

testN = [1, 10, 100]

# compare the exact values of some harmonics with Muselli parametrization
def test_g3():
    ns = [1.0, 2.0, 1 + 1j]
    # NIntegrate[x^({1, 2, 1 + I} - 1) PolyLog[2, x]/(1 + x), {x, 0, 1}]
    refvals = [0.3888958462, 0.2560382207, 0.3049381491 - 0.1589060625j]
    for n, r in zip(ns, refvals):
        S1 = h.S1(n)
        np.testing.assert_almost_equal(h.g_functions.mellin_g3(n, S1), r, decimal=6)


def test_g4():
    refvals = [-1, -1.34359, -1.40286]
    for N, vals in zip(testN, refvals):
        is_singlet = (-1) ** N == 1
        S1 = h.S1(N)
        S2 = h.S2(N)
        Sm1 = h.Sm1(N, S1, is_singlet)
        Sm2 = h.Sm2(N, S2, is_singlet)
        S2m1 = h.S2m1(N, S2, Sm1, Sm2, is_singlet)
        np.testing.assert_allclose(S2m1, vals, atol=1e-05)


def test_g6():
    refvals = [-1, -0.857976, -0.859245]
    for N, vals in zip(testN, refvals):
        is_singlet = (-1) ** N == 1
        S1 = h.S1(N)
        S2 = h.S2(N)
        Sm1 = h.Sm1(N, S1, is_singlet)
        Sm2 = h.Sm2(N, S2, is_singlet)
        Sm31 = h.Sm31(N, S1, Sm1, Sm2, is_singlet)
        np.testing.assert_allclose(Sm31, vals, atol=1e-05)


def test_g5():
    refvals = [-1, -0.777375, -0.784297]
    for N, vals in zip(testN, refvals):
        is_singlet = (-1) ** N == 1
        S1 = h.S1(N)
        S2 = h.S2(N)
        Sm1 = h.Sm1(N, S1, is_singlet)
        Sm2 = h.Sm2(N, S2, is_singlet)
        Sm31 = h.Sm31(N, S1, Sm1, Sm2, is_singlet)
        Sm22 = h.Sm22(N, S1, S2, Sm2, Sm31, is_singlet)
        np.testing.assert_allclose(Sm22, vals, atol=1e-05)


def test_g8():
    refvals = [-1, -0.696836, -0.719637]
    for N, vals in zip(testN, refvals):
        is_singlet = (-1) ** N == 1
        S1 = h.S1(N)
        S2 = h.S2(N)
        Sm1 = h.Sm1(N, S1, is_singlet)
        Sm211 = h.Sm211(N, S1, S2, Sm1, is_singlet)
        np.testing.assert_allclose(Sm211, vals, atol=1e-05)


def test_g18():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.375, 1.5787, 2.0279, 2.34252]
    for N, vals in zip(testN, refvals):
        S1 = h.S1(N)
        S2 = h.S2(N)
        S21 = h.S21(N, S1, S2)
        np.testing.assert_allclose(S21, vals, atol=1e-05)


def test_g19():
    refvals = [1, 0.953673, 0.958928]
    for N, vals in zip(testN, refvals):
        is_singlet = (-1) ** N == 1
        S1 = h.S1(N)
        S2 = h.S2(N)
        Sm2 = h.Sm2(N, S2, is_singlet)
        Sm2m1 = h.Sm2m1(N, S1, S2, Sm2)
        np.testing.assert_allclose(Sm2m1, vals, atol=1e-05)


def test_g21():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.4375, 1.69985, 2.38081, 3.04323]
    for N, vals in zip(testN, refvals):
        S1 = h.S1(N)
        S2 = h.S2(N)
        S3 = h.S3(N)
        S211 = h.S211(N, S1, S2, S3)
        np.testing.assert_allclose(S211, vals, atol=1e-05)


def test_g22():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.1875, 1.2554, 1.33724, 1.35262]
    for N, vals in zip(testN, refvals):
        S1 = h.S1(N)
        S2 = h.S2(N)
        S3 = h.S3(N)
        S4 = h.S4(N)
        S31 = h.S31(N, S1, S2, S3, S4)
        np.testing.assert_allclose(S31, vals, atol=1e-05)
