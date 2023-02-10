# Test NLO OME
import numpy as np

from ekore.operator_matrix_elements.unpolarized.space_like.as1 import A_ns, A_singlet

from ekore.harmonics import cache as c

def test_A_1_intrinsic():

    cache = c.reset()
    L = 100.0
    N = 2
    aS1 = A_singlet(N, L, cache, None)
    # heavy quark momentum conservation
    np.testing.assert_allclose(aS1[0, 2] + aS1[1, 2] + aS1[2, 2], 0.0, atol=1e-10)

    # gluon momentum conservation
    np.testing.assert_allclose(aS1[0, 0] + aS1[1, 0] + aS1[2, 0], 0.0)


def test_A_1_shape():

    cache = c.reset()
    N = 2
    L = 3.0
    aNS1i = A_ns(N, L, cache, None)
    aS1i = A_singlet(N, L, cache, None)

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
        cache = c.reset()
        for L, ref_gg in ref_val_gg.items():
            aS1 = A_singlet(N, L, cache, None)
            np.testing.assert_allclose(aS1[0, 0], ref_gg[n], rtol=1e-6)
            np.testing.assert_allclose(aS1[2, 0], ref_val_Hg[L][n], rtol=3e-6)
