# Test LO splitting functions
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_raises

from eko import anomalous_dimensions as ad
from eko import basis_rotation as br
from eko.anomalous_dimensions import as1 as ad_as1
from eko.anomalous_dimensions import harmonics

NF = 5


def test_eigensystem_gamma_singlet_0_values():
    n = 3
    s1 = harmonics.S1(n)
    gamma_S_0 = ad_as1.gamma_singlet(3, s1, NF)
    res = ad.exp_singlet(gamma_S_0)
    lambda_p = complex(12.273612971466964, 0)
    lambda_m = complex(5.015275917421917, 0)
    e_p = np.array(
        [
            [0.07443573 + 0.0j, -0.32146941 + 0.0j],
            [-0.21431294 + 0.0j, 0.92556427 + 0.0j],
        ]
    )
    e_m = np.array(
        [[0.92556427 + 0.0j, 0.32146941 + 0.0j], [0.21431294 + 0.0j, 0.07443573 + 0.0j]]
    )
    assert_almost_equal(lambda_p, res[1])
    assert_almost_equal(lambda_m, res[2])
    assert_allclose(e_p, res[3])
    assert_allclose(e_m, res[4])


def test_eigensystem_gamma_singlet_projectors_EV():
    nf = 3
    for N in [3, 4]:  # N=2 seems close to 0, so test fails
        for o in [(2, 0), (3, 0), (4, 0)]:
            # NNLO and N3LO too big numbers,
            # ignore Runtime Warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            for gamma_S in ad.gamma_singlet(o, N, nf, n3lo_ad_variation=[0, 0, 0, 0]):
                _exp, l_p, l_m, e_p, e_m = ad.exp_singlet(gamma_S)
                # projectors behave as P_a . P_b = delta_ab P_a
                assert_allclose(np.dot(e_p, e_p), e_p)
                assert_almost_equal(np.dot(e_p, e_m), np.zeros((2, 2)))
                assert_allclose(np.dot(e_m, e_m), e_m)
                # check EVs
                assert_allclose(np.dot(e_p, gamma_S), l_p * e_p)
                assert_allclose(np.dot(e_m, gamma_S), l_m * e_m)


def test_gamma_ns():
    nf = 3
    # LO
    assert_almost_equal(
        ad.gamma_ns((3, 0), br.non_singlet_pids_map["ns+"], 1, nf)[0], 0.0
    )
    # NLO
    assert_allclose(
        ad.gamma_ns((2, 0), br.non_singlet_pids_map["ns-"], 1, nf),
        np.zeros(2),
        atol=2e-6,
    )
    # NNLO
    assert_allclose(
        ad.gamma_ns((3, 0), br.non_singlet_pids_map["ns-"], 1, nf),
        np.zeros(3),
        atol=2e-4,
    )
    assert_allclose(
        ad.gamma_ns((3, 0), br.non_singlet_pids_map["nsV"], 1, nf),
        np.zeros(3),
        atol=8e-4,
    )
    # N3LO
    assert_allclose(
        ad.gamma_ns((4, 0), br.non_singlet_pids_map["ns-"], 1, nf),
        np.zeros(4),
        atol=2e-4,
    )
    # N3LO valence has a spurious pole, need to add a small shift
    assert_allclose(
        ad.gamma_ns((4, 0), br.non_singlet_pids_map["nsV"], 1 + 1e-6, nf),
        np.zeros(4),
        atol=5e-4,
    )
    assert_raises(
        AssertionError,
        assert_allclose,
        ad.gamma_ns((4, 0), br.non_singlet_pids_map["ns+"], 1, nf),
        np.zeros(4),
    )
