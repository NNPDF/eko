# -*- coding: utf-8 -*-
# Test cs functions
import numpy as np

from eko.anomalous_dimensions import harmonics
import eko.matching_conditions.n3lo.cs_functions as csf
import eko.matching_conditions.n3lo.h_functions as hf


# reference values coming fom mathematica
testNinteger = [1, 2, 10, 100]
refvalsinteger = {
    "S1l05": [0.5, 0.625, 0.693065, 0.693147],
    "S111l211": [2.0, 5.5, 1092.6, 3.64201e29],
    "S12l21": [2.0, 4.5, 361.913, 4.18766e28],
    "S21l21": [2.0, 3.5, 80.0103, 1.33999e27],
    "S3l2": [2.0, 2.5, 6.1981, 2.61629e24],
    "S1111l20511": [1.0, 2.4375, 211.939, 2.30926e28],
    "S1111l21051": [1.0, 2.6875, 438.767, 9.83117e28],
    "S121l2051": [1.0, 2.3125, 177.276, 1.92438e28],
    "S13l205": [1.0, 2.0625, 127.392, 1.37604e28],
    "S211l2051": [1.0, 1.6875, 24.6954, 2.1289e26],
    "S22l205": [1.0, 1.5625, 17.9259, 1.50709e26],
}


testN = [1.2, 2 + 2j, 10 - 10j, 100 + 5j]
refvals = {
    "S1l05": [
        0.538137,
        0.712215 + 0.0553881j,
        0.693133 - 0.0000611141j,
        0.693147 - 2.81649e-33j,
    ],
    "S111l211": [
        2.54956,
        0.757758 + 7.14874j,
        909.774 - 228.694j,
        -3.48531e29 - 1.04752e29j,
    ],
    "S12l21": [
        2.44162,
        2.49213 + 5.02065j,
        237.335 + 37.1463j,
        -4.02641e28 - 1.13156e28j,
    ],
    "S21l21": [
        2.30848,
        3.16611 + 2.64925j,
        27.0649 + 23.3928j,
        -1.30088e27 - 3.08519e26j,
    ],
    "S3l2": [
        2.13581,
        2.68836 + 0.514803j,
        2.47645 + 0.00409976j,
        -2.56754e24 - 4.47673e23j,
    ],
    "S1111l20511": [
        1.24503,
        1.1248 + 3.00684j,
        131.725 + 24.7849j,
        -2.22049e28 - 6.23296e27j,
    ],
    "S1111l21051": [
        1.26935,
        0.553945 + 3.4886j,
        348.464 - 34.3895j,
        -9.42727e28 - 2.75605e28j,
    ],
    "S1111l21105": [
        1.28092,
        0.240202 + 3.69116j,
        518.932 - 150.299j,
        -2.12395e29 - 6.4095e28j,
    ],
    "S112l2051": [
        1.23026 - 7.87292e-16j,
        1.26939 + 2.68422j,
        110.53 + 20.4704j,
        -1.85041e28 - 5.19414e27j,
    ],
    "S112l2105": [
        1.25606 + 1.90605e-16j,
        0.778372 + 3.21344j,
        276.555 - 18.8422j,
        -7.17239e28 - 2.09103e28j,
    ],
    "S121l2051": [
        1.23026 - 7.87292e-16j,
        1.26939 + 2.68422j,
        110.53 + 20.4704j,
        -1.85041e28 - 5.19414e27j,
    ],
    "S121l2105": [
        1.22865 - 5.91011e-18j,
        1.18968 + 2.66297j,
        132.763 + 19.9346j,
        -2.26363e28 - 6.36336e27j,
    ],
    "S13l205": [
        1.19667,
        1.39248 + 2.05457j,
        79.7069 + 14.5882j,
        -1.32315e28 - 3.71411e27j,
    ],
    "S211l2051": [
        1.14671,
        1.64933 + 1.16434j,
        5.5855 + 6.21788j,
        -2.07082e26 - 4.69937e25j,
    ],
    "S211l2105": [
        1.1634,
        1.57541 + 1.47654j,
        16.9764 + 14.4009j,
        -8.41216e26 - 2.00123e26j,
    ],
    "S22l205": [
        1.12596,
        1.5831 + 0.877058j,
        4.40596 + 4.39592j,
        -1.46597e26 - 3.32677e25j,
    ],
    "S31l205": [
        1.08185,
        1.42797 + 0.349638j,
        1.28279 + 0.00292271j,
        -1.77969e24 - 3.10303e23j,
    ],
}

# compare the exact values of some harmonics
def test_S1l05():
    for N, vals in zip(testN, refvals["S1l05"]):
        test = csf.S1l05(N)
        np.testing.assert_allclose(test, vals, rtol=1e-05)
    for N, vals in zip(testNinteger, refvalsinteger["S1l05"]):
        test = csf.S1l05(N)
        np.testing.assert_allclose(test, vals, rtol=1e-05)


# TODO: still on going
# def test_S111l211():
#     for N, vals in zip(testN, refvals["S111l211"]):
#         S1 = harmonics.harmonic_S1(N)
#         H24 = hf.H24(n,S1)
#         test = csf.S111l211(H24)
#         np.testing.assert_allclose(test, vals), rtol=1e-05)
#
#
# def test_S12l21():
#     for N, vals in zip(testN, refvals["S12l21"]):
#         test = csf.S12l21(H25)
#         np.testing.assert_allclose(test - 1.0j * imtest, vals), rtol=8e-03)


def test_S21l21():
    for N, vals in zip(testN, refvals["S21l21"]):
        S1 = harmonics.harmonic_S1(N)
        test = csf.S21l21(N, S1)
        rtol = 15e-02
        if N == 100 + 5j:
            rtol = 25e-01
        np.testing.assert_allclose(test, vals, rtol=rtol)


def test_S3l2():
    for N, vals in zip(testN, refvals["S3l2"]):
        test = csf.S3l2(N)
        # TODO: here Mathematica and MPmath do not agree on LerchPhi,
        # when the 3rd argument is complez and first real...
        np.testing.assert_allclose(np.real(test), np.real(vals), rtol=1e-05)

    for N, vals in zip(testNinteger, refvalsinteger["S3l2"]):
        test = csf.S3l2(N)
        np.testing.assert_allclose(test, vals, rtol=1e-05)


def test_S1111l20511():
    for N, vals in zip(testN, refvals["S1111l20511"]):
        S1 = harmonics.harmonic_S1(N)
        test = csf.S1111l20511(N, S1)
        np.testing.assert_allclose(test, vals, rtol=3e-03)


def test_S1111l21051():
    for N, vals in zip(testN, refvals["S1111l21051"]):
        S1 = harmonics.harmonic_S1(N)
        H21 = hf.H21(N, S1)
        test = csf.S1111l21051(N, S1, H21)
        rtol = 6e-03
        if N == 100 + 5j:
            rtol = 2e-01
        np.testing.assert_allclose(test, vals, rtol=rtol)


# TODO: still on going
# def test_S1111l21105():
#     for N, vals in zip(testN, refvals["S1111l21105"]):
#         S1 = harmonics.harmonic_S1(N)
#         H21 = hf.H21(N,S1)
#         H23 = hf.H23(N,S1)
#         H24 = hf.H24(N,S1)
#         test = csf.S1111l21105(N, S1, H21,H23,H24)
#         np.testing.assert_allclose(test, vals), rtol=15e-03)


def test_S112l2051():
    for N, vals in zip(testN, refvals["S112l2051"]):
        S1 = harmonics.harmonic_S1(N)
        test = csf.S112l2051(N, S1)
        np.testing.assert_allclose(test, vals, rtol=5e-03)


def test_S112l2105():
    for N, vals in zip(testN, refvals["S112l2105"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        H21 = hf.H21(N, S1)
        H23 = hf.H23(N, S1)
        test = csf.S112l2105(N, S1, S2, H21, H23)
        rtol = 6e-03
        if N == 100 + 5j:
            rtol = 2e-01
        np.testing.assert_allclose(test, vals, rtol=rtol)


def test_S121l2051():
    for N, vals in zip(testN, refvals["S121l2051"]):
        S1 = harmonics.harmonic_S1(N)
        H23 = hf.H23(N, S1)
        test = csf.S121l2051(N, S1, H23)
        np.testing.assert_allclose(test, vals, rtol=15e-01)


# TODO: still on going
# def test_S121l2105():
#     for N, vals in zip(testN, refvals["S121l2105"]):
#         S1 = harmonics.harmonic_S1(N)
#         S2 = harmonics.harmonic_S2(N)
#         test = csf.S121l2105(N, S1, S2,H23,H25)
#         np.testing.assert_allclose(test, vals), rtol=2e-05)


def test_S13l205():
    for N, vals in zip(testN, refvals["S13l205"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        test = csf.S13l205(N, S1, S2, S3)
        np.testing.assert_allclose(test, vals, rtol=3e-03)


# not given alone
# def test_S211l2051():
#     for N, vals in zip(testNinteger, refvalsinteger["S211l2051"]):
#         S1 = harmonics.harmonic_S1(N)
#         S2 = harmonics.harmonic_S2(N)
#         test = csf.S211l2051(N,S1,S2)
#         np.testing.assert_allclose(test, vals, rtol=1e-03)
#     for N, vals in zip(testN, refvals["S211l2051"]):
#         S1 = harmonics.harmonic_S1(N)
#         S2 = harmonics.harmonic_S2(N)
#         test = csf.S211l2051(N,S1,S2)
#         # same as S3l2 for complex part
#         np.testing.assert_allclose(np.real(test), np.real(vals), rtol=1e-03)

# not given alone
# def test_S211l2105():
#     for N, vals in zip(testN, refvals["S211l2105"]):
#         S1 = harmonics.harmonic_S1(N)
#         S2 = harmonics.harmonic_S2(N)
#         test = csf.S211l2105(N, S1, S2)
#         if N == 100 + 5j:
#             continue
#         np.testing.assert_allclose(test, vals, rtol=2e-02)

# not given alone
# def test_S22l205():
#     for N, vals in zip(testN, refvals["S22l205"]):
#         S1 = harmonics.harmonic_S1(N)
#         S2 = harmonics.harmonic_S2(N)
#         test = csf.S22l205(N, S1, S2)
#         np.testing.assert_allclose(test, vals, rtol=1e-04)


def test_S31l205():
    for N, vals in zip(testN, refvals["S31l205"]):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        S3 = harmonics.harmonic_S3(N)
        test = csf.S31l205(N, S1, S2, S3)
        # same as S3l2 for complex part
        np.testing.assert_allclose(np.real(test), np.real(vals), rtol=1e-04)


def test_S211l2051_S211l2105_S22l205():
    for N, vals in zip(
        testN,
        np.array(refvals["S211l2051"])
        + np.array(refvals["S211l2105"])
        - np.array(refvals["S22l205"]),
    ):
        S1 = harmonics.harmonic_S1(N)
        S2 = harmonics.harmonic_S2(N)
        test = hf.H26(N, S1, S2)
        rtol = 2e-02
        if N == 5j + 100:
            rtol = 25e-01
        np.testing.assert_allclose(test, vals, rtol=rtol)
