# Test NLO polarized OME
import numpy as np

from ekore.operator_matrix_elements.polarized.space_like.as1 import (
    A_gg,
    A_hg,
    A_singlet,
)
from ekore.operator_matrix_elements.unpolarized.space_like import as1 as as1unp


def test_A_1_shape():
    N = 2
    L = 3.0
    aS1i = A_singlet(N, L)
    assert aS1i.shape == (3, 3)


def test_gg():
    L = 10
    np.testing.assert_allclose(as1unp.A_gg(L), A_gg(L))


def test_hg():
    L = 10
    refs = [10 / 3, 3, 50 / 21, 35 / 18, 18 / 11]
    vals = []
    for i, _ in enumerate(refs):
        n = 2 * i + 2
        vals.append(A_hg(n, L))
    np.testing.assert_allclose(vals, refs)
