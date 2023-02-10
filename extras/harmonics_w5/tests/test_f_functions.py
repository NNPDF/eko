# Test F functions implementing w5 harmonics sums

import harmonics_w5 as w5
import numpy as np

from ekore.harmonics import cache as c

log2 = np.log(2)

# reference values coming fom Mathematica:
# they are computed inverting the harmonics to x-space
# and then doing numerical integration.

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
    "Sm21": [
        -1.0,
        -0.625,
        -0.751192 - 0.000147181j,
        -0.751286 - 1.17067e-9j,
        -0.751029,
    ],
    "S23": [1.0, 1.28125, 1.45296 + 0.227775j, 1.65521 + 0.0442804j, 1.73653],
    "S2m3": [-1.0, -1.21875, -1.35491 - 0.173841j, -1.50575 - 0.0332824j, -1.56676],
    "Sm23": [-1.0, -0.71875, -0.802462 - 0.000100325j, -0.802494, -0.802435],
}


# compare the exact values of some harmonics
# All the harmonics definition are coming from :cite`:Bl_mlein_2009` section 9.
# F19, F20,F21 are not present in that paper.
def test_F9():
    for N, vals in zip(testN, refvals["S41"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        S41 = w5.S41(N, S1, S2, S3)
        np.testing.assert_allclose(S41, vals, atol=1e-05)


def test_F11():
    for N, vals in zip(testN, refvals["S311"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S311 = w5.S311(N, S1, S2)
        np.testing.assert_allclose(S311, vals, atol=1e-05)


def test_F13():
    for N, vals in zip(testN, refvals["S221"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S21 = c.get(c.S21, cache, N, None)
        S221 = w5.S221(N, S1, S2, S21)
        np.testing.assert_allclose(S221, vals, atol=1e-05)


def test_F12_F14():
    # here there is a typo in eq (9.25) of Bl_mlein_2009
    for N, vals in zip(testN, refvals["Sm221"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        Sm1 = c.get(c.Sm1, cache, N, None)
        S21 = c.get(c.S21, cache, N, None)
        Sm21 = c.get(c.Sm21, cache, N, None)
        Sm221 = w5.Sm221(N, S1, Sm1, S21, Sm21)
        np.testing.assert_allclose(Sm221, vals, atol=1e-05)


def test_F16():
    for N, vals in zip(testN, refvals["S21m2"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        Sm1 = c.get(c.Sm1, cache, N, None)
        Sm2 = c.get(c.Sm2, cache, N, None)
        Sm3 = c.get(c.Sm3, cache, N, None)
        S21 = c.get(c.S21, cache, N, None)
        S2m1 = c.get(c.S2m1, cache, N, None)
        Sm21 = c.get(c.Sm21, cache, N, None)
        S21m2 = w5.S21m2(N, S1, S2, Sm1, Sm2, Sm3, S21, Sm21, S2m1)
        np.testing.assert_allclose(S21m2, vals, atol=1e-04)


def test_F17():
    for N, vals in zip(testN, refvals["S2111"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        S2111 = w5.S2111(N, S1, S2, S3)
        np.testing.assert_allclose(S2111, vals, atol=1e-05)


def test_F18():
    for N, vals in zip(testN, refvals["Sm2111"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        Sm1 = c.get(c.Sm1, cache, N, None)
        Sm2111 = w5.Sm2111(N, S1, S2, S3, Sm1)
        np.testing.assert_allclose(Sm2111, vals, atol=1e-05)


# different parametrization, less accurate
def test_F19():
    for N, vals in zip(testN, refvals["S23"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        S23 = w5.S23(N, S1, S2, S3)
        np.testing.assert_allclose(S23, vals, rtol=2e-03)


# different parametrization, less accurate
def test_F20():
    for N, vals in zip(testN, refvals["Sm23"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        Sm3 = c.get(c.Sm3, cache, N, None)
        Sm2 = c.get(c.Sm2, cache, N, None)
        Sm1 = c.get(c.Sm1, cache, N, None)
        Sm23 = w5.Sm23(N, Sm1, Sm2, Sm3)
        np.testing.assert_allclose(Sm23, vals, rtol=1e-03)


# different parametrization, less accurate
def test_F21():
    for N, vals in zip(testN, refvals["S2m3"]):
        cache = c.reset()
        S1 = c.get(c.S1, cache, N, None)
        S2 = c.get(c.S2, cache, N, None)
        S3 = c.get(c.S3, cache, N, None)
        Sm3 = c.get(c.Sm3, cache, N, None)
        Sm2 = c.get(c.Sm2, cache, N, None)
        Sm1 = c.get(c.Sm1, cache, N, None)
        S2m3 = w5.S2m3(N, S2, Sm1, Sm2, Sm3)
        np.testing.assert_allclose(S2m3, vals, rtol=1e-03)
