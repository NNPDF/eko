# -*- coding: utf-8 -*-
# Test Mellin stuff
from numpy.testing import assert_almost_equal

import numba as nb
import numpy as np
import eko.mellin as mellin


def check_path_derivation(path, jacobian):
    """ Check the derivatives of the path """
    epss = [1e-2, 1e-3, 1e-4, 1e-5]
    for t0 in [0.2, 0.4, 0.6, 0.8]:  # avoid 0.5 due to Talbot+edge
        # compute numeric derivative + extrapolation
        num = []
        for eps in epss:
            derivative = (path(t0 + eps) - path(t0 - eps)) / (2.0 * eps)
            num.append(derivative)
        ex = jacobian(t0)
        f = np.polyfit(epss, num, 2)
        assert_almost_equal(ex, f[2], decimal=4)


def test_inverse_mellin_transform():
    """test inverse_mellin_transform"""

    def function_x(x):
        return x

    @nb.njit
    def function_N(N, lnx):
        return np.exp(- N * lnx) / (N + 1)

    xgrid = [0.1, 0.3, 0.5, 0.7]
    p, j = mellin.get_path_Talbot()
    for x in xgrid:
        xresult = function_x(x)
        integrand = mellin.compile_integrand(function_N, p, j)
        nresult = mellin.inverse_mellin_transform(integrand, 1e-2, extra_args = np.log(x))
        assert_almost_equal(xresult, nresult[0])
        assert_almost_equal(0.0, nresult[1])

def test_get_path_Talbot():
    scales = [1, 2]
    for s in scales:
        path, jacobian = mellin.get_path_Talbot(s)
        check_path_derivation(path, jacobian)

def test_get_path_line():
    params = [(1, 1), (2, 2)]
    for m, c in params:
        path, jacobian = mellin.get_path_line(m, c)
        check_path_derivation(path, jacobian)

def test_get_path_edge():
    params = [(2, 1), (2, 2)]
    for m, c in params:
        path, jacobian = mellin.get_path_line(m, c)
        check_path_derivation(path, jacobian)

def test_get_path_Cauchy_tan():
    params = [(1, 1), (2, 2)]
    for c,o in params:
        path, jacobian = mellin.get_path_Cauchy_tan(c,o)
        check_path_derivation(path, jacobian)

def test_path_similarity():
    # these two should be identical
    p1,j1 = mellin.get_path_line(10.0,1.0)
    p2,j2 = mellin.get_path_edge(20.0,1.0,np.pi/2.0)
    for t in [0.1,0.2,0.3,0.4,0.6]:
        assert_almost_equal(p1(t),p2(t))
        assert_almost_equal(j1(t),j2(t))

def test_Mellin_transform():
    f = lambda x: x
    g = lambda N: 1.0 / (N + 1.0)
    for N in [1.0, 1.0 + 1j, 0.5 - 2j]:
        e = g(N)
        a = mellin.mellin_transform(f, N)
        assert_almost_equal(e, a[0])
        assert_almost_equal(0.0, a[1])
