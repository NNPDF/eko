import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

import ekore.anomalous_dimensions.polarized.space_like as ad_ps
from eko import basis_rotation as br


def test_shapes():
    for k in range(1, 3 + 1):
        assert ad_ps.gamma_ns((k, 0), 10101, 2.0, 4).shape == (k,)
        assert ad_ps.gamma_singlet((k, 0), 2.0, 4).shape == (k, 2, 2)


def test_gamma_ns():
    nf = 3
    # LO
    assert_almost_equal(
        ad_ps.gamma_ns((3, 0), br.non_singlet_pids_map["ns+"], 1, nf)[0], 0.0
    )
    # NLO
    assert_allclose(
        ad_ps.gamma_ns((2, 0), br.non_singlet_pids_map["ns+"], 1, nf),
        np.zeros(2),
        atol=2e-6,
    )
    # NNLO
    assert_allclose(
        ad_ps.gamma_ns((3, 0), br.non_singlet_pids_map["ns+"], 1, nf),
        np.zeros(3),
        atol=2e-4,
    )
    assert not np.array_equal(
        ad_ps.gamma_ns((3, 0), br.non_singlet_pids_map["nsV"], 1, nf), np.zeros(3)
    )


def test_not_implemeted():
    with pytest.raises(NotImplementedError):
        ad_ps.gamma_ns((4, 0), br.non_singlet_pids_map["ns-"], 1.234, 4)
    with pytest.raises(NotImplementedError):
        ad_ps.gamma_ns((2, 0), 10202, 1.234, 4)
