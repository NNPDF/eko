import numpy as np
import pytest

import ekore.operator_matrix_elements.polarized.space_like as ome_ps


def test_implemeted_orders():
    n = 1.234
    L = 4.567
    nf = 4
    for order in [2]:
        A_ns = ome_ps.A_non_singlet((order, 0), n, L)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(A_ns, np.zeros_like(A_ns))
    for order in [1, 2]:
        A_s = ome_ps.A_singlet((order, 0), n, nf, L)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(A_s, np.zeros_like(A_s))
        ome_ps.A_non_singlet((1, 0), 1.0, 4, 0.0)
    with pytest.raises(NotImplementedError):
        ome_ps.A_singlet((1, 0), 1.0, 4, 0.0, False)
