import pytest

import ekore.operator_matrix_elements.polarized.space_like as ome_ps


def test_init():
    with pytest.raises(NotImplementedError):
        ome_ps.A_non_singlet((1, 0), 1.0, 4, 0.0)
    with pytest.raises(NotImplementedError):
        ome_ps.A_singlet((1, 0), 1.0, 4, 0.0, False)
