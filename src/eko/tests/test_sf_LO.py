# -*- coding: utf-8 -*-
# Test splitting functions
from numpy.testing import assert_approx_equal, assert_almost_equal

from eko.constants import Constants
import eko.splitting_functions_LO as spf_LO

constants = Constants()
CA = constants.CA
CF = constants.CF
Tf = constants.TF
NF = 5


def check_values(function, inputs, known_values):
    """ Takes advantages of the unified signature for all coefficients
    to check the value for N == `N1` """
    for N, val in zip(inputs, known_values):
        result = function(N, NF, CA, CF)
        assert_almost_equal(result, val)


def test__S1():
    """test harmonic sum _S1"""
    # test on real axis
    known_vals = [1.0, 1.0 + 1.0 / 2.0, 1.0 + 1.0 / 2.0 + 1.0 / 3.0]
    for i, val in enumerate(known_vals):
        result = spf_LO._S1(1 + i)
        assert_approx_equal(result, val, significant=8)


def test_gamma_ns_0():
    """test gamma_ns_0"""
    # momentum conservation
    input_N = [complex(1.0, 0.0)]
    known_vals = [complex(0.0, 0.0)]
    check_values(spf_LO.gamma_ns_0, input_N, known_vals)


def test_gamma_ps_0():
    input_N = [complex(1.0, 0.0)]
    known_vals = [complex(0.0, 0.0)]
    check_values(spf_LO.gamma_ps_0, input_N, known_vals)

def test_gamma_qg_0():
    input_N = [complex(1.0, 0.0)]
    known_vals = [complex(-20.0/3.0, 0.0)]
    check_values(spf_LO.gamma_qg_0, input_N, known_vals)


def test_gamma_gq_0():
    input_N = [complex(0.0, 1.0)]
    known_vals = [complex(4.0,-4.0)/3.0]
    check_values(spf_LO.gamma_gq_0, input_N, known_vals)


def test_gamma_gg_0():
    input_N = [complex(0.0, 1.0)]
    known_vals = [complex(5.195725159621,10.52008856962)]
    check_values(spf_LO.gamma_gg_0, input_N, known_vals)
