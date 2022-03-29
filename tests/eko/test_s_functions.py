# -*- coding: utf-8 -*-
# Test some harmonics

import numpy as np

import eko.harmonics as sf

# reference values coming fom mathematica
testN = [1, 2, 2 + 2j, 10 + 5j, 100]
refvals = {
    "Sm1": [-1.0, -0.5, -0.692917 - 0.000175788j, -0.693147 - 2.77406e-9j, -0.688172],
    "Sm2": [
        -1.0,
        -0.75,
        -0.822442 - 0.0000853585j,
        -0.822467 - 4.29516e-10j,
        -0.822418,
    ],
    "Sm3": [
        -1.0,
        -0.875,
        -0.901551 - 0.0000255879j,
        -0.901543 - 4.61382e-11j,
        -0.901542,
    ],
    "Sm4": [
        -1.0,
        -0.9375,
        -0.947039 - 4.84597e-6j,
        -0.947033 - 3.99567e-12j,
        -0.947033,
    ],
    "Sm5": [-1.0, -0.96875, -0.972122 - 1.13162e-7j, -0.97212 - 2.81097e-13j, -0.97212],
    "Sm21": [
        -1.0,
        -0.625,
        -0.751192 - 0.000147181j,
        -0.751286 - 1.17067e-9j,
        -0.751029,
    ],
}


def test_Sm21():
    for N, vals in zip(testN, refvals["Sm21"]):
        Sm1 = sf.Sm1(N)
        np.testing.assert_allclose(sf.Sm21(N, Sm1), vals, atol=1e-06)


def test_Smx():
    for j, N in enumerate(testN):
        smx = [
            sf.Sm1(N),
            sf.Sm2(N),
            sf.Sm3(N),
            sf.Sm4(N),
            sf.Sm5(N),
        ]
        for i, sm in enumerate(smx):
            np.testing.assert_allclose(sm, refvals[f"Sm{i+1}"][j], atol=1e-06)


def test_Sx():
    """test harmonic sums S_x on real axis"""
    # test on real axis
    def sx(n, m):
        return np.sum([1 / k**m for k in range(1, n + 1)])

    ls = [
        sf.S1,
        sf.S2,
        sf.S3,
        sf.S4,
        sf.S5,
    ]
    for k in range(1, 5 + 1):
        for n in range(1, 4 + 1):
            np.testing.assert_almost_equal(ls[k - 1](n), sx(n, k))
