# -*- coding: utf-8 -*-
# Test some harmomics and the binomial factor

import numpy as np
import scipy.special as sp

import eko.matching_conditions.n3lo.s_functions as sf

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
        Sm1 = sf.harmonic_Sm1(N)
        np.testing.assert_allclose(sf.harmonic_Sm21(N, Sm1), vals, atol=1e-06)


def test_Smx():
    for j, N in enumerate(testN):
        smx = [
            sf.harmonic_Sm1(N),
            sf.harmonic_Sm2(N),
            sf.harmonic_Sm3(N),
            sf.harmonic_Sm4(N),
            sf.harmonic_Sm5(N),
        ]
        for i, sm in enumerate(smx):
            np.testing.assert_allclose(sm, refvals[f"Sm{i+1}"][j], atol=1e-06)

def test_binomial():
    r1 = np.random.randint(1000)
    r2 = np.random.randint(1000)
    np.testing.assert_allclose(sf.binomial(r1,r2), sp.binom(r1, r2))
