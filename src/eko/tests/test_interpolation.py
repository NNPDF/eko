# -*- coding: utf-8 -*-
# Test interpolation
from numpy.polynomial import Chebyshev
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np
import scipy.integrate as integrate


from eko import t_float
from eko.interpolation import (
    get_xgrid_linear_at_id,
    get_xgrid_linear_at_log,
    get_xgrid_Chebyshev_at_id,
    get_xgrid_Chebyshev_at_log,
    get_Lagrange_interpolators_x,
    get_Lagrange_interpolators_N,
    get_Lagrange_interpolators_log_x,
    get_Lagrange_interpolators_log_N,
)

# for the numeric comparision to work, keep in mind that in Python3 the default precision is
# np.float64


def check_is_tfloat(function, grid_size=3, xmin=0.0, xmax=1.0):
    """ Checks all members of the return value of function are of type t_float """
    result = function(grid_size, xmin, xmax)
    for i in result:
        assert isinstance(i, t_float)
    return result


def check_xgrid(function, grid, grid_size=3, xmin=0.0, xmax=1.0):
    """ Checks the grid that function returns correspond to `grid` """
    result = function(grid_size, xmin, xmax)
    assert_allclose(result, grid)
    return result


def check_interpolator(function, points, values, xmin=0.0, j=3):
    arr = np.linspace(xmin, 1, 5)
    for x, val in zip(points, values):
        result = function(x, arr, j)
        assert_almost_equal(result, val, decimal=4)


def test_get_xgrid_linear_at_id():
    """test linear@id grids"""
    grid_result = np.array([0.0, 0.5, 1.0])
    check_xgrid(get_xgrid_linear_at_id, grid_result)
    check_is_tfloat(get_xgrid_linear_at_id)


def test_get_xgrid_Chebyshev_at_id():
    """test get_xgrid_Chebyshev_at_id"""
    for n in [3, 5, 7]:
        check_is_tfloat(get_xgrid_Chebyshev_at_id, grid_size=n)
        # test that grid points correspond indeed to nodes of the polynomial
        cheb_n = Chebyshev(np.append(np.zeros(n), 1), domain=[0, 1])
        check_xgrid(get_xgrid_Chebyshev_at_id, cheb_n.roots(), grid_size=n)


def test_get_xgrid_linear_at_log10():
    """test linear@log10 grids"""
    grid_result = np.array([1e-2, 1e-1, 1.0])
    check_is_tfloat(get_xgrid_linear_at_log, xmin=1e-2)
    check_xgrid(get_xgrid_linear_at_log, grid_result, xmin=1e-2)


def test_get_xgrid_Chebyshev_at_log():
    xmin = 1e-2
    for n in [3, 5, 7]:
        check_is_tfloat(get_xgrid_Chebyshev_at_log, grid_size=n, xmin=xmin)
        cheb_n = Chebyshev(np.append(np.zeros(n), 1), domain=[0, 1])
        exp_arg = np.log(xmin) - cheb_n.roots() * np.log(xmin)
        nodes = np.exp(exp_arg)
        check_xgrid(get_xgrid_Chebyshev_at_log, nodes, grid_size=n, xmin=xmin)


def test_get_Lagrange_interpolators_x():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [0.3]
    values = [-504 / 5625]
    check_interpolator(get_Lagrange_interpolators_x, points, values)


def test_get_Lagrange_interpolators_N():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [complex(0.5, 0.5)]
    values = [complex(0.381839, -0.1408880)]
    check_interpolator(get_Lagrange_interpolators_N, points, values)


def test_get_Lagrange_interpolators_log_x():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [0.3]
    values = [-0.6199271485409041]
    check_interpolator(get_Lagrange_interpolators_log_x, points, values, xmin=1e-2)


def test_get_Lagrange_interpolators_log_N():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [complex(0.5, 0.5)]
    values = [complex(-42.24104240911104, -120.36554908750743)]
    check_interpolator(get_Lagrange_interpolators_log_N, points, values, xmin=1e-2)


def _Mellin_transform(f, N):
    """straight implementation of the Mellin transform"""
    r, re = integrate.quad(lambda x, f=f, N=N: np.real(x ** (N - 1) * f(x)), 0, 1)
    i, ie = integrate.quad(lambda x, f=f, N=N: np.imag(x ** (N - 1) * f(x)), 0, 1)
    return np.complex(r, i), np.complex(re, ie)


def test__Mellin_transform():
    """prevent circular reasoning"""
    f = lambda x: x
    g = lambda N: 1.0 / (N + 1.0)
    for N in [1.0, 1.0 + 1j, 0.5 - 2j]:
        e = g(N)
        a = _Mellin_transform(f, N)
        assert_almost_equal(e, a[0])
        assert_almost_equal(0.0, a[1])


def test_f_xN():
    """test correspondence of interpolators in x- and N-space"""
    for fxfNg in [
        (get_Lagrange_interpolators_x, get_Lagrange_interpolators_N, [0.0, 0.5, 1.0]),
        (
            get_Lagrange_interpolators_log_x,
            get_Lagrange_interpolators_log_N,
            [1e-4, 1e-2, 1.0],
        ),
    ]:
        fx, fN, g = fxfNg
        l = len(g)
        for j in range(l):
            for N in [1.0, 1.0 + 1j, 0.5 - 2j]:
                a = fN(N, g, j)
                e = _Mellin_transform(lambda y, fx=fx, g=g, j=j: fx(y, g, j), N)
                assert_almost_equal(a, e[0])
                assert_almost_equal(0.0, e[1])


def test_is_interpolators_x():
    """test that the interpolator functions are indeed interpolators"""
    for fg in [
        (get_Lagrange_interpolators_x, [0.0, 0.5, 1.0]),
        (get_Lagrange_interpolators_log_x, [1e-4, 1e-2, 1.0]),
    ]:
        f, g = fg
        l = len(g)
        # sum needs to be one
        for x in [1e-4, 1e-2, 0.2, 0.4, 0.6, 0.8]:
            s = np.sum([f(x, g, j) for j in range(l)])
            assert_almost_equal(1.0, s)
        # polynoms need to be "orthogonal" at grid points
        for j in range(l):
            one = f(g[j], g, j)
            assert np.abs(1.0 - one) < 1e-6
            for k in range(l):
                if j == k:
                    continue
                zero = f(g[k], g, j)
                assert_almost_equal(0.0, zero)
