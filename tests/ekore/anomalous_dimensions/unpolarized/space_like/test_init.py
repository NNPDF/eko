# Test LO splitting functions
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_raises
from scipy.linalg import expm

import ekore.anomalous_dimensions.unpolarized.space_like as ad_us
from eko import basis_rotation as br
from ekore import anomalous_dimensions as ad
from ekore import harmonics as h

NF = 5


def test_eigensystem_gamma_singlet_0_values():
    N = 3
    cache = h.cache.reset()
    gamma_S_0 = ad_us.as1.gamma_singlet(N, cache, NF)
    res = ad.exp_matrix_2D(gamma_S_0)
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


def test_exp_matrix():
    N = 3
    cache = h.cache.reset()
    gamma_S_0 = ad_us.as1.gamma_singlet(N, cache, NF)
    res = ad.exp_matrix_2D(gamma_S_0)[0]
    res2 = ad.exp_matrix(gamma_S_0)[0]
    assert_allclose(res, res2)
    gamma_S_0_qed = ad_us.as1.gamma_singlet_qed(3, cache, NF)
    res = expm(gamma_S_0_qed)
    res2 = ad.exp_matrix(gamma_S_0_qed)[0]
    assert_allclose(res, res2)
    gamma_2D = np.random.rand(2, 2) + np.random.rand(2, 2) * 1j
    res1 = expm(gamma_2D)
    res2 = ad.exp_matrix(gamma_2D)[0]
    assert_almost_equal(res1, res2)
    diag = np.diag([1, 2, 3, 4])
    assert_allclose(np.diag(np.exp([1, 2, 3, 4])), ad.exp_matrix(diag)[0])
    id_ = np.identity(4, np.complex_)
    zero = np.zeros((4, 4), np.complex_)
    assert_allclose(id_, ad.exp_matrix(zero)[0])
    sigma2 = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    exp = ad.exp_matrix(sigma2)[0]
    exp_m = ad.exp_matrix(-sigma2)[0]
    assert_almost_equal(np.identity(2, np.complex_), exp @ exp_m)


def test_eigensystem_gamma_singlet_projectors_EV():
    nf = 3
    for N in [3, 4]:  # N=2 seems close to 0, so test fails
        for o in [(2, 0), (3, 0), (4, 0)]:
            # NNLO and N3LO too big numbers,
            # ignore Runtime Warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            for gamma_S in ad_us.gamma_singlet(
                o, N, nf, n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0)
            ):
                _exp, l_p, l_m, e_p, e_m = ad.exp_matrix_2D(gamma_S)
                # projectors behave as P_a . P_b = delta_ab P_a
                assert_allclose(np.dot(e_p, e_p), e_p)
                assert_almost_equal(np.dot(e_p, e_m), np.zeros((2, 2)))
                assert_allclose(np.dot(e_m, e_m), e_m)
                # check EVs
                assert_allclose(np.dot(e_p, gamma_S), l_p * e_p)
                assert_allclose(np.dot(e_m, gamma_S), l_m * e_m)


def test_gamma_ns():
    nf = 3
    n3lo_ad_variation = (0, 0, 0, 0, 0, 0, 0)
    # ad_us.as1
    assert_almost_equal(
        ad_us.gamma_ns(
            (3, 0), br.non_singlet_pids_map["ns+"], 1, nf, n3lo_ad_variation
        )[0],
        0.0,
    )
    # ad_us.as2
    assert_allclose(
        ad_us.gamma_ns(
            (2, 0), br.non_singlet_pids_map["ns-"], 1, nf, n3lo_ad_variation
        ),
        np.zeros(2),
        atol=2e-6,
    )
    # ad_us.as3
    assert_allclose(
        ad_us.gamma_ns(
            (3, 0), br.non_singlet_pids_map["ns-"], 1, nf, n3lo_ad_variation
        ),
        np.zeros(3),
        atol=2e-4,
    )
    assert_allclose(
        ad_us.gamma_ns(
            (3, 0), br.non_singlet_pids_map["nsV"], 1, nf, n3lo_ad_variation
        ),
        np.zeros(3),
        atol=8e-4,
    )
    # as4
    assert_allclose(
        ad_us.gamma_ns(
            (4, 0), br.non_singlet_pids_map["ns-"], 1, nf, n3lo_ad_variation
        ),
        np.zeros(4),
        atol=2e-4,
    )
    # N3LO valence has a spurious pole, need to add a small shift
    assert_allclose(
        ad_us.gamma_ns(
            (4, 0), br.non_singlet_pids_map["nsV"], 1 + 1e-6, nf, n3lo_ad_variation
        ),
        np.zeros(4),
        atol=5e-4,
    )
    assert_raises(
        AssertionError,
        assert_allclose,
        ad_us.gamma_ns(
            (4, 0), br.non_singlet_pids_map["ns+"], 1, nf, n3lo_ad_variation
        ),
        np.zeros(4),
    )
    with pytest.raises(NotImplementedError):
        ad_us.gamma_ns((2, 0), 10106, 2.0, nf, n3lo_ad_variation)


def test_gamma_ns_qed():
    nf = 3
    # aem1
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 1), br.non_singlet_pids_map["ns-u"], 1, nf),
        np.zeros((2, 2)),
        decimal=5,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 1), br.non_singlet_pids_map["ns-d"], 1, nf),
        np.zeros((2, 2)),
        decimal=5,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 1), br.non_singlet_pids_map["ns+u"], 1, nf)[0, 1],
        0,
        decimal=5,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 1), br.non_singlet_pids_map["ns+d"], 1, nf)[0, 1],
        0,
        decimal=5,
    )
    # as1aem1
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 2), br.non_singlet_pids_map["ns-u"], 1, nf),
        np.zeros((2, 3)),
        decimal=5,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 2), br.non_singlet_pids_map["ns-d"], 1, nf),
        np.zeros((2, 3)),
        decimal=5,
    )
    # aem2
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 2), br.non_singlet_pids_map["ns-u"], 1, nf),
        np.zeros((2, 3)),
        decimal=5,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((1, 2), br.non_singlet_pids_map["ns-d"], 1, nf),
        np.zeros((2, 3)),
        decimal=5,
    )
    # ad_us.as2
    assert_almost_equal(
        ad_us.gamma_ns_qed((2, 1), br.non_singlet_pids_map["ns-u"], 1, nf),
        np.zeros((3, 2)),
        decimal=5,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((2, 1), br.non_singlet_pids_map["ns-d"], 1, nf),
        np.zeros((3, 2)),
        decimal=5,
    )
    # ad_us.as3
    assert_almost_equal(
        ad_us.gamma_ns_qed((3, 1), br.non_singlet_pids_map["ns-u"], 1, nf),
        np.zeros((4, 2)),
        decimal=3,
    )
    assert_almost_equal(
        ad_us.gamma_ns_qed((3, 1), br.non_singlet_pids_map["ns-d"], 1, nf),
        np.zeros((4, 2)),
        decimal=3,
    )


def test_errors():
    cache = h.cache.reset()
    with pytest.raises(NotImplementedError):
        ad_us.choose_ns_ad_as1aem1(10106, 2.0, cache)
    with pytest.raises(NotImplementedError):
        ad_us.choose_ns_ad_aem2(10106, 2.0, 4, cache)


def test_dim_singlet():
    nf = 3
    N = 2
    cache = h.cache.reset()
    gamma_singlet = ad_us.gamma_singlet_qed((3, 2), N, nf, (0, 0, 0, 0))
    assert gamma_singlet.shape == (4, 3, 4, 4)
    gamma_singlet_as1 = ad_us.as1.gamma_singlet_qed(N, cache, nf)
    assert gamma_singlet_as1.shape == (4, 4)
    gamma_singlet_as2 = ad_us.as2.gamma_singlet_qed(N, nf, cache)
    assert gamma_singlet_as2.shape == (4, 4)
    gamma_singlet_as3 = ad_us.as3.gamma_singlet_qed(N, nf, cache)
    assert gamma_singlet_as3.shape == (4, 4)


def test_dim_valence():
    nf = 3
    N = 2
    cache = h.cache.reset()
    gamma_valence = ad_us.gamma_valence_qed((3, 2), N, nf)
    assert gamma_valence.shape == (4, 3, 2, 2)
    gamma_valence_as1 = ad_us.as1.gamma_valence_qed(N, cache)
    assert gamma_valence_as1.shape == (2, 2)
    gamma_valence_as2 = ad_us.as2.gamma_valence_qed(N, nf, cache)
    assert gamma_valence_as2.shape == (2, 2)
    gamma_valence_as3 = ad_us.as3.gamma_valence_qed(N, nf, cache)
    assert gamma_valence_as3.shape == (2, 2)


def test_dim_nsp():
    nf = 3
    N = 2
    gamma_nsup = ad_us.gamma_ns_qed((3, 2), 10102, N, nf)
    assert gamma_nsup.shape == (4, 3)
    gamma_nsdp = ad_us.gamma_ns_qed((3, 2), 10103, N, nf)
    assert gamma_nsdp.shape == (4, 3)
    with pytest.raises(NotImplementedError):
        ad_us.gamma_ns_qed((2, 0), 10106, N, nf)
