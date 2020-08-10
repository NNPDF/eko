# -*- coding: utf-8 -*-
"""Test Mellin module"""

from numpy.testing import assert_almost_equal

import numba as nb
import numpy as np

import eko.mellin as mellin


def _numbify_list(extra_args):
    """ convert list to numba """
    extra_args_nb = nb.typed.List()
    for arg in extra_args:
        extra_args_nb.append(arg)
    return extra_args_nb


def check_path_derivation(path, jacobian, extra_args):
    """ Check the derivatives of the path """
    extra_args_nb = _numbify_list(extra_args)

    epss = [1e-2, 1e-3, 1e-4, 1e-5]
    for t0 in [0.2, 0.4, 0.6, 0.8]:  # avoid 0.5 due to Talbot+edge
        # compute numeric derivative + extrapolation
        num = []
        for eps in epss:
            derivative = (
                path(t0 + eps, extra_args_nb) - path(t0 - eps, extra_args_nb)
            ) / (2.0 * eps)
            num.append(derivative)
        ex = jacobian(t0, extra_args_nb)
        f = np.polyfit(epss, num, 2)
        assert_almost_equal(ex, f[2], decimal=4)


def check_path_symmetry(path, jac, extra_args):
    """ Check symmetry arount 1/2 """
    extra_args_nb = _numbify_list(extra_args)
    for t in [0.1, 0.2, 0.3]:
        plow = path(0.5 - t, extra_args_nb)
        phigh = path(0.5 + t, extra_args_nb)
        assert_almost_equal(plow, np.conjugate(phigh))
        jlow = jac(0.5 - t, extra_args_nb)
        jhigh = jac(0.5 + t, extra_args_nb)
        assert_almost_equal(jlow, -np.conjugate(jhigh))


def test_inverse_mellin_transform():
    """test inverse_mellin_transform"""

    def function_x(x):
        return x

    @nb.njit
    def function_N(N, lnx, a1, a0):  # pylint:disable=unused-argument
        return np.exp(-N * lnx) / (N + 1)

    xgrid = [0.1, 0.3, 0.5, 0.7]
    p, j = mellin.get_path_Talbot()
    for x in xgrid:
        xresult = function_x(x)
        for compile_numba in [True, False]:
            integrand = mellin.compile_integrand(function_N, p, j, compile_numba)
            extra_args = _numbify_list([np.log(x), 0.0, 0.0, 1.0, 0.0])
            nresult = mellin.inverse_mellin_transform(integrand, 1e-2, extra_args)
            assert_almost_equal(xresult, nresult[0])
            assert_almost_equal(0.0, nresult[1])


def test_get_path_Talbot():
    params = [[1, 0], [2, 0]]
    for p in params:
        path, jacobian = mellin.get_path_Talbot()
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)
        # assert special points
        extra_args_nb = _numbify_list(p)
        assert_almost_equal(path(0.5, extra_args_nb), p[0] + p[1])
        assert_almost_equal(
            jacobian(0.5, extra_args_nb), np.complex(0, p[0] * 2.0 * np.pi)
        )


def test_get_path_line():
    params = [[1, 1], [2, 2]]
    for p in params:
        path, jacobian = mellin.get_path_line()
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)


def test_get_path_edge():
    params = [[2, 1], [2, 2]]
    for p in params:
        path, jacobian = mellin.get_path_line()
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)


def test_get_path_Cauchy_tan():
    params = [[1, 1], [2, 2]]
    for p in params:
        path, jacobian = mellin.get_path_Cauchy_tan()
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)


def test_path_similarity():
    # these two should be identical
    p1, j1 = mellin.get_path_line()
    a1 = nb.typed.List()
    a1.append(10.0)
    a1.append(1.0)
    p2, j2 = mellin.get_path_edge()
    a2 = nb.typed.List()
    a2.append(20.0)
    a2.append(1.0)
    a2.append(np.pi / 2.0)
    for t in [0.1, 0.2, 0.3, 0.4, 0.6]:
        assert_almost_equal(p1(t, a1), p2(t, a2))
        assert_almost_equal(j1(t, a1), j2(t, a2))


def test_Mellin_transform():
    f = lambda x: x
    g = lambda N: 1.0 / (N + 1.0)
    for N in [1.0, 1.0 + 1j, 0.5 - 2j]:
        e = g(N)
        a = mellin.mellin_transform(f, N)
        assert_almost_equal(e, a[0])
        assert_almost_equal(0.0, a[1])
