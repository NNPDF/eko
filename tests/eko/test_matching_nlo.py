# -*- coding: utf-8 -*-
# Test NLO OME
import numpy as np

from eko.matching_conditions.as1 import A_ns, A_singlet
from eko.matching_conditions.operator_matrix_element import compute_harmonics_cache


def test_A_1_intrinsic():

    L = 100.0
    N = 2
    sx = compute_harmonics_cache(N, 2, True)
    aS1 = A_singlet(N, sx, L)
    # heavy quark momentum conservation
    np.testing.assert_allclose(aS1[0, 2] + aS1[1, 2] + aS1[2, 2], 0.0, atol=1e-10)

    # gluon momentum conservation
    np.testing.assert_allclose(aS1[0, 0] + aS1[1, 0] + aS1[2, 0], 0.0)


def test_A_1_shape():

    N = 2
    L = 3.0
    sx = compute_harmonics_cache(N, 2, (-1) ** N == 1)
    aNS1i = A_ns(N, sx, L)
    aS1i = A_singlet(N, sx, L)

    assert aNS1i.shape == (2, 2)
    assert aS1i.shape == (3, 3)

    # check intrinsic hh is the same
    assert aNS1i[1, 1] == aS1i[2, 2]


def test_Blumlein_1():
    # Test against Blumlein OME implementation :cite:`Bierenbaum:2009mv`.
    # Only even moments are available in that code.
    # Note there is a minus sign in the definition of L.

    N_vals = 5
    ref_val_gg = {10: np.full(N_vals, -6.66667)}
    ref_val_Hg = {
        10: [6.66667, 3.66667, 2.61905, 2.05556, 1.69697],
    }

    for n in range(N_vals):
        N = 2 * n + 2
        sx = compute_harmonics_cache(N, 2, True)
        for L, ref_gg in ref_val_gg.items():
            aS1 = A_singlet(N, sx, L)
            np.testing.assert_allclose(aS1[0, 0], ref_gg[n], rtol=1e-6)
            np.testing.assert_allclose(aS1[2, 0], ref_val_Hg[L][n], rtol=3e-6)
