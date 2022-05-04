# -*- coding: utf-8 -*-

import numpy as np

from eko.scale_variations.exponentiated import gamma_variation


def test_gamma_ns_fact():
    gamma_ns = np.array([1.0, 0.5, 0.25, 0.125])
    gamma_ns_LO_0 = gamma_variation(gamma_ns.copy(), (0, 0), 3, 0)
    np.testing.assert_allclose(gamma_ns_LO_0, gamma_ns)
    gamma_ns_LO_1 = gamma_variation(gamma_ns.copy(), (0, 0), 3, 1)
    np.testing.assert_allclose(gamma_ns_LO_1, gamma_ns)
    gamma_ns_NLO_1 = gamma_variation(gamma_ns.copy(), (1, 0), 3, 1)
    assert gamma_ns_NLO_1[1] < gamma_ns[1]
    gamma_ns_NNLO_1 = gamma_variation(gamma_ns.copy(), (2, 0), 3, 1)
    assert gamma_ns_NNLO_1[2] - gamma_ns[2] == 8.0
    gamma_ns_N3LO_0 = gamma_variation(gamma_ns.copy(), (3, 0), 3, 0)
    assert gamma_ns_N3LO_0[3] == gamma_ns[3]


def test_gamma_singlet_fact():
    gamma_s = np.array([1.0, 0.5, 0.25, 0.125])
    gamma_s_LO_0 = gamma_variation(gamma_s.copy(), (0, 0), 3, 0)
    np.testing.assert_allclose(gamma_s_LO_0, gamma_s)
    gamma_s_LO_1 = gamma_variation(gamma_s.copy(), (0, 0), 3, 1)
    np.testing.assert_allclose(gamma_s_LO_1, gamma_s)
    gamma_s_NLO_1 = gamma_variation(gamma_s.copy(), (1, 0), 3, 1)
    assert gamma_s_NLO_1[1] < gamma_s[1]
    gamma_s_NNLO_1 = gamma_variation(gamma_s.copy(), (2, 0), 3, 1)
    assert gamma_s_NNLO_1[2] - gamma_s[2] == 8.0
    gamma_s_N3LO_0 = gamma_variation(gamma_s.copy(), (3, 0), 3, 0)
    assert gamma_s_N3LO_0[3] == gamma_s[3]
