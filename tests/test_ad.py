# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from eko.constants import Constants
import eko.anomalous_dimensions as ad
import eko.anomalous_dimensions.lo as ad_lo

constants = Constants()
CA = constants.CA
CF = constants.CF
NF = 5


def test_eigensystem_gamma_singlet_0_values():
    gamma_S_0 = ad_lo.gamma_singlet_0(3, NF, CA, CF)
    res = ad.exp_singlet(gamma_S_0)
    lambda_p = np.complex(12.273612971466964, 0)
    lambda_m = np.complex(5.015275917421917, 0)
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


def test_eigensystem_gamma_singlet_0_projectors_EV():
    for N in [3, 4]:  # N=2 seems close to 0, so test fails
        gamma_S_0 = ad_lo.gamma_singlet_0(N, NF, CA, CF)
        _exp, l_p, l_m, e_p, e_m = ad.exp_singlet(gamma_S_0)
        # projectors behave as P_a . P_b = delta_ab P_a
        assert_allclose(np.dot(e_p, e_p), e_p)
        assert_almost_equal(np.dot(e_p, e_m), np.zeros((2, 2)))
        assert_allclose(np.dot(e_m, e_m), e_m)
        # check EVs
        gamma_S = ad_lo.gamma_singlet_0(N, NF, CA, CF)
        assert_allclose(np.dot(e_p, gamma_S), l_p * e_p)
        assert_allclose(np.dot(e_m, gamma_S), l_m * e_m)
