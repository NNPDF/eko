import pytest

import ekore.anomalous_dimensions.polarized.space_like as ad_ps


def test_init():
    with pytest.raises(NotImplementedError):
        ad_ps.gamma_ns((1, 0), 0, 1.0, 4)
    with pytest.raises(NotImplementedError):
        ad_ps.gamma_singlet((1, 0), 1.0, 4)
