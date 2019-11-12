# -*- coding: utf-8 -*-
# Test interpolation
from numpy.polynomial import Chebyshev
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np


from eko import t_float, t_complex
from eko.interpolation import (
    get_xgrid_linear_at_id,
    get_xgrid_linear_at_log,
    get_xgrid_Chebyshev_at_id,
    get_xgrid_Chebyshev_at_log,
    InterpolatorDispatcher,
    get_Lagrange_basis_functions,
    evaluate_Lagrange_basis_function_x,
    evaluate_Lagrange_basis_function_N,
    get_Lagrange_basis_functions_log,
    evaluate_Lagrange_basis_function_log_x,
    evaluate_Lagrange_basis_function_log_N,
)
import eko.mellin as mellin

# for the numeric comparision to work, keep in mind that in Python3 the default precision is
# np.float64

# Checking utilities
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


def check_interpolator_with_cache(
    interpolator, variable, points, values, xmin=0.0, j=3
):
    arr = np.linspace(xmin, 1, 5)
    function = InterpolatorDispatcher(interpolator, variable, arr)
    for x, val in zip(points, values):
        result = function(x, j)
        assert_almost_equal(result, val, decimal=4)


def check_is_interpolator(interpolator, variable, xgrid):
    """ Check whether the function `inter_x` is indeed an interpolator """
    inter_x = InterpolatorDispatcher(interpolator, variable, xgrid)
    values = [1e-4, 1e-2, 0.2, 0.4, 0.6, 0.8]
    for v in values:
        one = 0.0
        # Check it sums to one
        for j in range(len(xgrid)):
            one += inter_x(v, j)
        assert_almost_equal(one, 1.0)

    # polynoms need to be "orthogonal" at grid points
    for j, x in enumerate(xgrid):
        one = inter_x(x, j)
        assert_almost_equal(one, 1.0)

        for k, y in enumerate(xgrid):
            if j == k:
                continue
            zero = inter_x(y, j)
            assert_almost_equal(zero, 0.0)


def check_correspondence_interpolators(interpolator, mode, xgrid):
    """ Check the correspondece between x and N space of the interpolators
    inter_x and inter_N"""
    inter_N = InterpolatorDispatcher(interpolator, f"{mode}N", xgrid)
    inter_x = InterpolatorDispatcher(interpolator, f"{mode}x", xgrid)
    ngrid = [complex(1.0), complex(1.0 + 1j), t_complex(0.5 - 2j)]
    for j in range(len(xgrid)):

        def ker(x):
            return inter_x(x, j)

        for N in ngrid:
            result_N = inter_N(N, j)
            result_x = mellin.mellin_transform(ker, N)
            assert_almost_equal(result_x[0], result_N)


def check_is_interpolator2(getConfs, xgrid, polynom_rank, evaluate_x):
    """ Check whether the functions are indeed interpolators"""
    confs = getConfs(xgrid, polynom_rank)
    values = [0.1, 0.2, 0.4, 0.6, 0.8]
    # has to be in the range of the interpolation, but for the numerical integration of the
    # logartithmic interpolation to work it has to be setup in a rather larger area
    for v in values:
        one = 0.0
        # Check it sums to one
        for conf in confs:
            one += evaluate_x(v, conf)
        assert_almost_equal(one, 1.0)

    # polynoms need to be "orthogonal" at grid points
    for j, xj in enumerate(xgrid):
        one = evaluate_x(xj, confs[j])
        assert_almost_equal(one, 1.0)

        for k, confk in enumerate(confs):
            if j == k:
                continue
            zero = evaluate_x(xj, confk)
            assert_almost_equal(zero, 0.0)


def check_correspondence_interpolators2(confs, evaluate_x, evaluate_N):
    """ Check the correspondece between x and N space of the interpolators
    inter_x and inter_N"""
    ngrid = [t_complex(1.0), t_complex(1.0 + 1j), t_complex(2.5 - 2j)]
    for conf in confs:

        def ker(x):
            return evaluate_x(x, conf)

        for N in ngrid:
            result_N = evaluate_N(N, conf, 0)
            result_x = mellin.mellin_transform(ker, N)
            assert_almost_equal(result_x[0], result_N)


def test_get_Lagrange_basis_functions():
    xgrid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for polynom_rank in range(1, len(xgrid)):
        # check interpolation
        check_is_interpolator2(
            get_Lagrange_basis_functions,
            xgrid,
            polynom_rank,
            evaluate_Lagrange_basis_function_x,
        )
        # check correspondence
        confs = get_Lagrange_basis_functions(xgrid, polynom_rank)
        check_correspondence_interpolators2(
            confs,
            evaluate_Lagrange_basis_function_x,
            evaluate_Lagrange_basis_function_N,
        )
    # do some more explicit testing here?
    # as we know all polynomials in an easy test case, say [0,.5,1]


def test_get_Lagrange_basis_functions_log():
    xgrid = np.array(np.logspace(-1, 0, num=5))
    for polynom_rank in range(1, len(xgrid)):
        # check interpolation
        check_is_interpolator2(
            get_Lagrange_basis_functions_log,
            xgrid,
            polynom_rank,
            evaluate_Lagrange_basis_function_log_x,
        )
        # check correspondence
        confs = get_Lagrange_basis_functions_log(xgrid, polynom_rank)
        check_correspondence_interpolators2(
            confs,
            evaluate_Lagrange_basis_function_log_x,
            evaluate_Lagrange_basis_function_log_N,
        )


# TEST functions
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
    check_interpolator_with_cache("Lagrange", "x", points, values)
    check_is_interpolator("Lagrange", "x", [0.0, 0.5, 1.0])


def test_get_Lagrange_interpolators_N():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [complex(0.5, 0.5)]
    values = [complex(0.381839, -0.1408880)]
    check_interpolator_with_cache("Lagrange", "N", points, values)


def test_correspondence_lagrange_xN():
    """test correspondence of interpolators in x- and N-space"""
    check_correspondence_interpolators("Lagrange", "", [0.0, 0.5, 1.0])


def test_get_Lagrange_interpolators_log_x():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [0.3]
    values = [-0.6199271485409041]
    check_interpolator_with_cache("Lagrange", "logx", points, values, xmin=1e-2)
    check_is_interpolator("Lagrange", "logx", [1e-4, 1e-2, 1.0])


def test_get_Lagrange_interpolators_log_N():
    # TODO: this assumes implementation at f61b238602db5a43f1945fb015dbc88cdfee0dd0 is ok
    # try some external way?
    points = [complex(0.5, 0.5)]
    values = [complex(-42.24104240911104, -120.36554908750743)]
    check_interpolator_with_cache("Lagrange", "logN", points, values, xmin=1e-2)


def test_correspondence_lagrange_log_xN():
    check_correspondence_interpolators("Lagrange", "log", [1e-4, 1e-2, 1.0])
