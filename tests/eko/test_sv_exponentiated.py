import numpy as np

from eko.scale_variations.exponentiated import gamma_variation, gamma_variation_qed


def test_gamma_ns_fact():
    gamma_ns = np.array([1.0, 0.5, 0.25, 0.125])
    gamma_ns_LO_0 = gamma_variation(gamma_ns.copy(), (1, 0), 3, 0)
    np.testing.assert_allclose(gamma_ns_LO_0, gamma_ns)
    gamma_ns_LO_1 = gamma_variation(gamma_ns.copy(), (1, 0), 3, 1)
    np.testing.assert_allclose(gamma_ns_LO_1, gamma_ns)
    gamma_ns_NLO_1 = gamma_variation(gamma_ns.copy(), (2, 0), 3, 1)
    assert gamma_ns_NLO_1[1] < gamma_ns[1]
    gamma_ns_NNLO_1 = gamma_variation(gamma_ns.copy(), (3, 0), 3, 1)
    assert gamma_ns_NNLO_1[2] - gamma_ns[2] == 8.0
    gamma_ns_N3LO_0 = gamma_variation(gamma_ns.copy(), (4, 0), 3, 0)
    assert gamma_ns_N3LO_0[3] == gamma_ns[3]


def test_gamma_singlet_fact():
    gamma_s = np.array([1.0, 0.5, 0.25, 0.125])
    gamma_s_LO_0 = gamma_variation(gamma_s.copy(), (1, 0), 3, 0)
    np.testing.assert_allclose(gamma_s_LO_0, gamma_s)
    gamma_s_LO_1 = gamma_variation(gamma_s.copy(), (1, 0), 3, 1)
    np.testing.assert_allclose(gamma_s_LO_1, gamma_s)
    gamma_s_NLO_1 = gamma_variation(gamma_s.copy(), (2, 0), 3, 1)
    assert gamma_s_NLO_1[1] < gamma_s[1]
    gamma_s_NNLO_1 = gamma_variation(gamma_s.copy(), (3, 0), 3, 1)
    assert gamma_s_NNLO_1[2] - gamma_s[2] == 8.0
    gamma_s_N3LO_0 = gamma_variation(gamma_s.copy(), (4, 0), 3, 0)
    assert gamma_s_N3LO_0[3] == gamma_s[3]


def test_gamma_ns_qed_fact():
    gamma_ns = np.array(
        [
            [0.0, 1.0, 0.5, 0.25, 0.125],
            [0.2, 0.1, 0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0, 0.0, 0.0],
        ]
    ).transpose()
    gamma_ns_as1aem1_0 = gamma_variation_qed(gamma_ns.copy(), (1, 1), 3, 0, True)
    np.testing.assert_allclose(gamma_ns_as1aem1_0, gamma_ns)
    gamma_ns_as1aem1_1 = gamma_variation_qed(gamma_ns.copy(), (1, 1), 3, 1, True)
    np.testing.assert_allclose(gamma_ns_as1aem1_1, gamma_ns)
    gamma_ns_as2aem2_1 = gamma_variation_qed(gamma_ns.copy(), (2, 2), 3, 1, True)
    assert gamma_ns_as2aem2_1[2, 0] < gamma_ns[2, 0]
    assert gamma_ns_as2aem2_1[0, 2] > gamma_ns[0, 2]  # beta0qed < 0
    gamma_ns_as4aem2_1 = gamma_variation_qed(gamma_ns.copy(), (4, 2), 3, 1, True)
    gamma_ns_N3LO_1 = gamma_variation(gamma_ns.copy()[1:, 0], (4, 0), 3, 1)
    np.testing.assert_allclose(gamma_ns_as4aem2_1[1:, 0], gamma_ns_N3LO_1)
