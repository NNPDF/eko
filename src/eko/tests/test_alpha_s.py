# -*- coding: utf-8 -*-
# Testing the beta functions
from eko.alpha_s import beta_0, beta_1

def test_beta_0():
    """Test first beta function coefficient"""
    for ngen in range(8):
        result = beta_0(2*ngen)
        assert result > 0.
        assert isinstance(result,float)

def test_beta_1():
    """Test second beta function coefficient"""
    for ngen in range(4):
        result = beta_1(2*ngen)
        assert result > 0.
        assert isinstance(result,float)

def test_beta_2():
    """Test third beta function coefficient"""
    for nf in range(5):
        result = beta_1(nf)
        assert result > 0.
        assert isinstance(result,float)
