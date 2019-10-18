# -*- coding: utf-8 -*-
# Test splitting functions
import numpy as np

from eko.constants import Constants
from eko.splitting_functions_LO import _S1,gamma_ns_0

def test__S1():
    """test harmonic sum _S1"""
    # test on real axis
    r = [1., 1.+1./2., 1.+1./2.+1./3.]
    l = len(r) # trick pylint
    for j in range(l):
        a = _S1(1+j)
        e = r[j]
        assert np.abs(a-e) < 1e-6

def test_gamma_ns_0():
    """test gamma_ns_0"""
    c = Constants()
    # momentum conservation
    zero = gamma_ns_0(1,4,c.CA,c.CF)
    assert np.abs(0. - zero) < 1e-6