# -*- coding: utf-8 -*-

import numpy as np

from eko import anomalous_dimensions as ad
from eko.scale_variations.scheme_A import gamma_ns_fact, gamma_singlet_fact


def test_gamma_ns_fact():
    gamma_ns = np.array([1.0, 0.5, 0.25])
    gamma_ns_LO_0 = gamma_ns_fact(gamma_ns.copy(), 0, 3, 0)
    np.testing.assert_allclose(gamma_ns_LO_0, gamma_ns)
    gamma_ns_LO_1 = gamma_ns_fact(gamma_ns.copy(), 0, 3, 1)
    np.testing.assert_allclose(gamma_ns_LO_1, gamma_ns)
    gamma_ns_NLO_1 = gamma_ns_fact(gamma_ns.copy(), 1, 3, 1)
    assert gamma_ns_NLO_1[1] < gamma_ns[1]
    gamma_ns_NNLO_1 = gamma_ns_fact(gamma_ns.copy(), 2, 3, 1)
    assert gamma_ns_NNLO_1[2] - gamma_ns[2] == 8.0


def test_gamma_singlet_fact():
    gamma_s = np.array([1.0, 0.5, 0.25])
    gamma_s_LO_0 = gamma_singlet_fact(gamma_s.copy(), 0, 3, 0)
    np.testing.assert_allclose(gamma_s_LO_0, gamma_s)
    gamma_s_LO_1 = gamma_singlet_fact(gamma_s.copy(), 0, 3, 1)
    np.testing.assert_allclose(gamma_s_LO_1, gamma_s)
    gamma_s_NLO_1 = gamma_singlet_fact(gamma_s.copy(), 1, 3, 1)
    assert gamma_s_NLO_1[1] < gamma_s[1]
    gamma_s_NNLO_1 = gamma_singlet_fact(gamma_s.copy(), 2, 3, 1)
    assert gamma_s_NNLO_1[2] - gamma_s[2] == 8.0
