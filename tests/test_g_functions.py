# -*- coding: utf-8 -*-
# Test G functions impleeted by muselli PhD

import numpy as np

from eko.anomalous_dimensions import harmonics
import eko.matching_conditions.n3lo.g_functions as gf
import eko.matching_conditions.n3lo.s_functions as sf

zeta2 = harmonics.zeta2
zeta3 = harmonics.zeta3

li4half = 0.517479
li5half = 0.508401
log2 = np.log(2)


testN = [1, 10, 100]


def harmonic_Sm31(N):
    Sm1 = sf.harmonic_Sm1(N, harmonics.harmonic_S1(N/2))
    Sm2 = sf.harmonic_Sm2(N, harmonics.harmonic_S2(N/2))
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


# copare the exact values of some harmonics with Muselli parametrisations
def test_g4():
    refvals = [-1, -1.34359, -1.40286]
    for N, vals in zip(testN, refvals):
        S2 = harmonics.harmonic_S2(N)
        Sm1 = sf.harmonic_Sm1(N, harmonics.harmonic_S1(N/2))
        Sm2 = sf.harmonic_Sm2(N, harmonics.harmonic_S2(N/2))
        S2m1 = sf.harmonic_S2m1(N, S2, Sm1, Sm2)
        np.testing.assert_allclose(S2m1, vals, atol=1e-05)


def test_g6():
    refvals = [-1, -0.857976, -0.859245]
    for N, vals in zip(testN, refvals):

        Sm31 = harmonic_Sm31(N)
        np.testing.assert_allclose(Sm31, vals, atol=1e-05)


def test_g5():
    refvals = [-1, -0.777375, -0.784297]
    for N, vals in zip(testN, refvals):
        Sm31 = harmonic_Sm31(N)
        Sm22 = (
            (-1) ** N * gf.mellin_g5(N)
            - 2 * Sm31
            + 2 * zeta2 * sf.harmonic_Sm2(N, harmonics.harmonic_S2(N/2))
            + 3 / 40 * zeta2 ** 2
        )
        np.testing.assert_allclose(Sm22, vals, atol=1e-05)


def test_g8():
    refvals = [-1, -0.696836, -0.719637]
    for N, vals in zip(testN, refvals):
        Sm1 = sf.harmonic_Sm1(N, harmonics.harmonic_S1(N/2))
        Sm211 = (
            -((-1) ** N) * gf.mellin_g8(N)
            + zeta3 * Sm1
            - li4half
            + 1 / 8 * zeta2 ** 2
            + 1 / 8 * zeta3 * log2
            + 1 / 4 * zeta2 * log2 ** 2
            - 1 / 24 * log2 ** 4
        )
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
        Sm2m1 = (
            -gf.mellin_g19(N, S1)
            + log2 * (harmonics.harmonic_S2(N) - sf.harmonic_Sm2(N, harmonics.harmonic_S2(N/2)))
            - 5 / 8 * zeta3
        )
        np.testing.assert_allclose(Sm2m1, vals, atol=1e-05)


def test_g21():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.4375, 1.69985, 2.38081, 3.04323]
    for N, vals in zip(testN, refvals):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        S211 = -gf.mellin_g21(N, S1, S2, S3) + 6 / 5 * zeta2 ** 2
        np.testing.assert_allclose(S211, vals, atol=1e-05)


def test_g22():
    testN = [1, 2, 3, 10, 100]
    refvals = [1, 1.1875, 1.2554, 1.33724, 1.35262]
    for N, vals in zip(testN, refvals):
        S2 = harmonics.harmonic_S2(N)
        S31 = (
            1 / 2 * gf.mellin_g22(N)
            - 1 / 4 * harmonics.harmonic_S4(N)
            - 1 / 4 * S2 ** 2
            + zeta2 * S2
            - 3 / 20 * zeta2 ** 2
        )
        np.testing.assert_allclose(S31, vals, atol=1e-05)
