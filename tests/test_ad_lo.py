# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

import eko.constants as consts
import eko.anomalous_dimensions.lo as ad_lo

constants = consts.Constants()
CA = constants.CA
CF = constants.CF
NF = 5


def check_values(function, inputs, known_values):
    """Takes advantages of the unified signature for all coefficients
    to check the value for N == `N1`"""
    for N, val in zip(inputs, known_values):
        result = function(N, NF, CA, CF)
        np.testing.assert_almost_equal(result, val)


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
