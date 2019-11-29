# -*- coding: utf-8 -*-
# Test interpolation
from numpy.polynomial import Chebyshev
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np


from eko import t_float, t_complex
from eko import interpolation
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


def check_is_interpolator(getConfs, xgrid, polynom_rank, evaluate_x):
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


def check_correspondence_interpolators(confs, evaluate_x, evaluate_N):
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


# def test_get_Lagrange_basis_functions():
#     xgrid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
#     for polynom_rank in range(1, len(xgrid)):
#         # check interpolation
#         check_is_interpolator(
#             get_Lagrange_basis_functions,
#             xgrid,
#             polynom_rank,
#             evaluate_Lagrange_basis_function_x,
#         )
#         # check correspondence
#         confs = get_Lagrange_basis_functions(xgrid, polynom_rank)
#         check_correspondence_interpolators(
#             confs,
#             evaluate_Lagrange_basis_function_x,
#             evaluate_Lagrange_basis_function_N,
#         )
#     # do some more explicit testing here?
#     # as we know all polynomials in an easy test case, say [0,.5,1]


# def test_get_Lagrange_basis_functions_log():
#     xgrid = np.array(np.logspace(-1, 0, num=5))
#     for polynom_rank in range(1, len(xgrid)):
#         # check interpolation
#         check_is_interpolator(
#             get_Lagrange_basis_functions_log,
#             xgrid,
#             polynom_rank,
#             evaluate_Lagrange_basis_function_log_x,
#         )
#         # check correspondence
#         confs = get_Lagrange_basis_functions_log(xgrid, polynom_rank)
#         check_correspondence_interpolators(
#             confs,
#             evaluate_Lagrange_basis_function_log_x,
#             evaluate_Lagrange_basis_function_log_N,
#         )


def test_get_xgrid_linear_at_id():
    """test linear@id grids"""
    grid_result = np.array([0.0, 0.5, 1.0])
    check_xgrid(interpolation.get_xgrid_linear_at_id, grid_result)
    check_is_tfloat(interpolation.get_xgrid_linear_at_id)


def test_get_xgrid_linear_at_log10():
    """test linear@log10 grids"""
    grid_result = np.array([1e-2, 1e-1, 1.0])
    check_is_tfloat(interpolation.get_xgrid_linear_at_log, xmin=1e-2)
    check_xgrid(interpolation.get_xgrid_linear_at_log, grid_result, xmin=1e-2)
