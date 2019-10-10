# -*- coding: utf-8 -*-
# Testing the beta functions
from eko import t_float
from eko.alpha_s import beta_0, beta_1, beta_2
from eko.constants import Constants

# the test will only pass for the default set of constants
constants = Constants()
CA = constants.CA
CF = constants.CF
Tf = constants.TF

# for the isinstance-tests to work, keep in mind that in Python3 the default precision is np.float64

def test_beta_0():
    """Test first beta function coefficient"""
    for ngen in range(8):
        result = beta_0(2*ngen, CA, CF, Tf)
        assert result > 0.
        assert isinstance(result,t_float)

def test_beta_1():
    """Test second beta function coefficient"""
    for ngen in range(4):
        result = beta_1(2*ngen, CA, CF, Tf)
        assert result > 0.
        assert isinstance(result,t_float)

def test_beta_2():
    """Test third beta function coefficient"""
    for nf in range(5):
        result = beta_2(nf, CA, CF, Tf)
        assert result > 0.
        assert isinstance(result,t_float)
