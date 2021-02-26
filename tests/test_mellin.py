# -*- coding: utf-8 -*-
"""Test Mellin module"""

from numpy.testing import assert_almost_equal

import numpy as np

import eko.mellin as mellin


def check_path_derivation(path, jacobian, extra_args):
    """ Check the derivatives of the path """

    epss = [1e-2, 1e-3, 1e-4, 1e-5]
    for t0 in [0.2, 0.4, 0.6, 0.8]:  # avoid 0.5 due to Talbot+edge
        # compute numeric derivative + extrapolation
        num = []
        for eps in epss:
            derivative = (path(t0 + eps, *extra_args) - path(t0 - eps, *extra_args)) / (
                2.0 * eps
            )
            num.append(derivative)
        ex = jacobian(t0, *extra_args)
        f = np.polyfit(epss, num, 2)
        assert_almost_equal(ex, f[2], decimal=4)


def check_path_symmetry(path, jac, extra_args):
    """ Check symmetry arount 1/2 """
    for t in [0.1, 0.2, 0.3]:
        plow = path(0.5 - t, *extra_args)
        phigh = path(0.5 + t, *extra_args)
        assert_almost_equal(plow, np.conjugate(phigh))
        jlow = jac(0.5 - t, *extra_args)
        jhigh = jac(0.5 + t, *extra_args)
        assert_almost_equal(jlow, -np.conjugate(jhigh))


def test_Talbot():
    params = [[1, 0], [2, 0]]
    for p in params:
        path, jacobian = mellin.Talbot_path, mellin.Talbot_jac
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)
        # assert special points
        assert_almost_equal(path(0.5, *p), p[0] + p[1])
        assert_almost_equal(jacobian(0.5, *p), complex(0, p[0] * 2.0 * np.pi))


def test_get_path_line():
    params = [[1, 1], [2, 2]]
    for p in params:
        path, jacobian = mellin.line_path, mellin.line_jac
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)


def test_get_path_edge():
    params = [[2, 1, np.pi / 2.0], [2, 2, np.pi / 2.0]]
    for p in params:
        path, jacobian = mellin.edge_path, mellin.edge_jac
        check_path_derivation(path, jacobian, p)
        check_path_symmetry(path, jacobian, p)


def test_path_similarity():
    # these two should be identical
    p1, j1 = mellin.line_path, mellin.line_jac
    a1 = [10.0, 1.0]
    p2, j2 = mellin.edge_path, mellin.edge_jac
    a2 = [20.0, 1.0, np.pi / 2.0]
    for t in [0.1, 0.2, 0.3, 0.4, 0.6]:
        assert_almost_equal(p1(t, *a1), p2(t, *a2))
        assert_almost_equal(j1(t, *a1), j2(t, *a2))
