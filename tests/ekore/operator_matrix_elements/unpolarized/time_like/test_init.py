import pytest

import ekore.operator_matrix_elements.unpolarized.time_like as ome_ut


def test_init():
    with pytest.raises(NotImplementedError):
        ome_ut.A_non_singlet((1, 0), 1.0, [], 4, 0.0)
    with pytest.raises(NotImplementedError):
        ome_ut.A_singlet((1, 0), 1.0, [], 4, 0.0, False)
