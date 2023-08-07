r"""Test |OME|."""

import ekore.operator_matrix_elements.unpolarized.space_like as ome


def test_shapes():
    for k in range(1, 3 + 1):
        assert ome.A_singlet((k, 0), complex(0, 1), 4, 0.0, False).shape == (k, 3, 3)
        assert ome.A_non_singlet((k, 0), complex(0, 1), 4, 0.0).shape == (k, 2, 2)
