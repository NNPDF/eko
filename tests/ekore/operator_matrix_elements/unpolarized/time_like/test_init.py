import ekore.operator_matrix_elements.unpolarized.time_like as ome


def test_shapes():
    assert ome.A_singlet((1, 0), complex(0, 1), 0).shape == (1, 3, 3)
    assert ome.A_non_singlet((1, 0), complex(0, 1), 0).shape == (1, 2, 2)
