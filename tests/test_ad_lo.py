# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from eko.constants import Constants
import eko.anomalous_dimensions.lo as ad_lo

constants = Constants()
CA = constants.CA
CF = constants.CF
NF = 5


def check_values(function, inputs, known_values):
    """ Takes advantages of the unified signature for all coefficients
    to check the value for N == `N1` """
    for N, val in zip(inputs, known_values):
        result = function(N, NF, CA, CF)
        assert_almost_equal(result, val)


def test_number_momentum_conservation():
    """test number/momentum conservation"""
    # number
    input_N = [complex(1.0, 0.0)]
    known_vals = [complex(0.0, 0.0)]
    check_values(ad_lo.gamma_ns_0, input_N, known_vals)

    # quark momentum
    input_N = [complex(2.0, 0.0)]
    known_vals = [complex(0.0, 0.0)]

    def _sum(*args):
        return ad_lo.gamma_ns_0(  # pylint: disable=no-value-for-parameter
            *args
        ) + ad_lo.gamma_gq_0(  # pylint: disable=no-value-for-parameter
            *args
        )

    check_values(_sum, input_N, known_vals)

    # gluon momentum
    def _sum(*args):
        return ad_lo.gamma_qg_0(  # pylint: disable=no-value-for-parameter
            *args
        ) + ad_lo.gamma_gg_0(  # pylint: disable=no-value-for-parameter
            *args
        )

    check_values(_sum, input_N, known_vals)


def test_gamma_qg_0():
    input_N = [complex(1.0, 0.0)]
    known_vals = [complex(-20.0 / 3.0, 0.0)]
    check_values(ad_lo.gamma_qg_0, input_N, known_vals)


def test_gamma_gq_0():
    input_N = [complex(0.0, 1.0)]
    known_vals = [complex(4.0, -4.0) / 3.0]
    check_values(ad_lo.gamma_gq_0, input_N, known_vals)


def test_gamma_gg_0():
    input_N = [complex(0.0, 1.0)]
    known_vals = [complex(5.195725159621, 10.52008856962)]
    check_values(ad_lo.gamma_gg_0, input_N, known_vals)


def test_get_Eigensystem_gamma_singlet_0_values():
    res = ad_lo.get_Eigensystem_gamma_singlet_0(3, NF, CA, CF)
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
    assert_almost_equal(lambda_p, res[0])
    assert_almost_equal(lambda_m, res[1])
    assert_allclose(e_p, res[2])
    assert_allclose(e_m, res[3])


def test_get_Eigensystem_gamma_singlet_0_projectors_EV():
    for N in [3, 4]:  # N=2 seems close to 0, so test fails
        l_p, l_m, e_p, e_m = ad_lo.get_Eigensystem_gamma_singlet_0(N, NF, CA, CF)
        # projectors behave as P_a . P_b = delta_ab P_a
        assert_allclose(np.dot(e_p, e_p), e_p)
        assert_almost_equal(np.dot(e_p, e_m), np.zeros((2, 2)))
        assert_allclose(np.dot(e_m, e_m), e_m)
        # check EVs
        gamma_S = ad_lo.get_gamma_singlet_0(N, NF, CA, CF)
        assert_allclose(np.dot(e_p, gamma_S), l_p * e_p)
        assert_allclose(np.dot(e_m, gamma_S), l_m * e_m)
