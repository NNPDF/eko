import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as2 as as2
import ekore.anomalous_dimensions.unpolarized.time_like.as2mela as as2m
import ekore.harmonics as h
from eko import constants

ca = 3
cf = 4 / 3
nf = 5

# def test_nsp():
#     N = complex(1.0, 0.0)
# #    val = [3.11111, 18.0753, 51.3206, 64.5425, 69.9012]
# #    for i, j in enumerate[1, 2, 5, 7, 8]:
# #        np.testing.assert_almost_equal(as2.gamma_nsp(j, nf), val[i])
#     np.testing.assert_almost_equal(as2.gamma_nsp(N, nf), 3.11111)

############################# mathematica based unit test ############################
# def test_nsp():
#     val = [
#         14.914763950248524,
#         27.170573816296294,
#         36.20843001367968,
#         43.6337052707661,
#         49.77278324218862,
#         55.10505809829998,
#         59.755406196991686,
#         63.922130588519025,
#     ]
#     for i, j in enumerate(val):
#         np.testing.assert_almost_equal(as2.gamma_nsp(i + 2, nf, None), j, decimal=6)


# def test_nsm():
#     val = [
#         14.437397695104497,
#         27.008022375967073,
#         36.13331890256858,
#         43.592750538255814,
#         49.7479719067197,
#         55.08888138952125,
#         59.74427031831462,
#         63.914136685150666,
#     ]
#     for i, j in enumerate(val):
#         np.testing.assert_almost_equal(as2.gamma_nsm(i + 2, nf), j, decimal=6)


# def test_qqs():
#     val = [
#         1280 / 81,
#         4391 / 810,
#         11867 / 4050,
#         186064 / 99225,
#         36490 / 27783,
#         3898835 / 4000752,
#         922333 / 1224720,
#         3969344 / 6615675,
#     ]
#     for i, j in enumerate(val):
#         np.testing.assert_almost_equal(as2.gamma_qqs(i + 2, nf), j, decimal=6)


# def test_qg():
#     val = [
#         4.394040436034569,
#         1.2283097867056794,
#         1.224271622535062,
#         0.5682223219230416,
#         0.5146142100999999,
#         0.20949896137930538,
#         0.15630406923958376,
#         -0.015229173167235112,
#     ]
#     for i, j in enumerate(val):
#         np.testing.assert_almost_equal(as2.gamma_qg(i + 2, nf), j, decimal=6)


# def test_gq():
#     val = [
#         -307.1723308605102,
#         -382.1730799366584,
#         -319.66481897017945,
#         -226.2889349719677,
#         -188.00245135408588,
#         -148.32874785657552,
#         -127.67168089632983,
#         -106.46381918972175,
#     ]
#     for i, j in enumerate(val):
#         np.testing.assert_almost_equal(as2.gamma_gq(i + 2, nf, None), j, decimal=6)


# def test_gg():
#     val = [-43.94040436034575,
#            -63.52992705212047,
#            -49.340293336279544,
#            -21.931061116073508,
#            -6.6250317132958685,
#            10.14906155587218,
#            21.959513930070727,
#            34.02767982616993]
#     for i, j in enumerate(val):
#         np.testing.assert_almost_equal(as2.gamma_gg(i + 2, nf), j, decimal=4)


def test_nsp():
    for N in 3, 5, 7, 9:
        for nf in 3, 4, 5:
            np.testing.assert_almost_equal(
                as2.gamma_nsp(N, nf, None, None), as2m.gamma_nsp(N, nf), decimal=4
            )


def test_nsm():
    for N in 3, 5, 7, 9:
        for nf in 3, 4, 5:
            np.testing.assert_almost_equal(
                as2.gamma_nsm(N, nf), as2m.gamma_nsm(N, nf), decimal=5
            )


def notest_qq():
    for N in 2, 4, 6, 8:
        for nf in 3, 4, 5:
            np.testing.assert_almost_equal(
                as2.gamma_singlet(N, nf, None)[0, 0],
                as2m.gamma_singlet(N, nf)[0, 0],
                decimal=5,
            )


def test_gq():
    for N in 2, 3, 4, 5, 6, 7, 8, 9:
        for nf in 3, 4, 5:
            np.testing.assert_almost_equal(
                as2.gamma_singlet(N, nf, None)[0, 1],
                as2m.gamma_singlet(N, nf)[0, 1],
                decimal=6,
            )


def test_qg():
    for N in 2, 4, 6, 8:
        for nf in 3, 4, 5:
            np.testing.assert_almost_equal(
                as2.gamma_singlet(N, nf, None)[1, 0],
                as2m.gamma_singlet(N, nf)[1, 0],
                decimal=5,
            )


def test_gg():
    for N in 2, 4, 6, 8:
        for nf in 3, 4, 5:
            np.testing.assert_almost_equal(
                as2.gamma_singlet(N, nf, None)[1, 1],
                as2m.gamma_singlet(N, nf)[1, 1],
                decimal=4,
            )
