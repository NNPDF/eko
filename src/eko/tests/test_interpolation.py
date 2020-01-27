# -*- coding: utf-8 -*-
# Test interpolation
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


from eko import t_float, t_complex
from eko import interpolation
import eko.mellin as mellin

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")

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


def check_is_interpolator(interpolator, xgrid):
    """ Check whether the functions are indeed interpolators"""
    values = [0.1, 0.2, 0.4, 0.6, 0.8]
    # has to be in the range of the interpolation, but for the numerical integration of the
    # logartithmic interpolation to work it has to be setup in a rather larger area
    for v in values:
        one = 0.0
        # Check it sums to one
        for basis in interpolator:
            one += basis(v)
        assert_almost_equal(one, 1.0)

    # polynoms need to be "orthogonal" at grid points
    for j, (basis, xj) in enumerate(zip(interpolator, xgrid)):
        one = basis(xj)
        assert_almost_equal(one, 1.0)

        for k, basis in enumerate(interpolator):
            if j == k:
                continue
            zero = basis(xj)
            assert_almost_equal(zero, 0.0)


def check_correspondence_interpolators(inter_x, inter_N):
    """ Check the correspondece between x and N space of the interpolators
    inter_x and inter_N"""
    ngrid = [t_complex(1.0), t_complex(1.0 + 1j), t_complex(2.5 - 2j)]
    logxinv = np.log(.9e-2) # < 1e-2, to trick skipping
    for fun_x, fun_N in zip(inter_x, inter_N):
        for N in ngrid:
            result_N = fun_N(N, logxinv) * np.exp(N*logxinv)
            result_x = mellin.mellin_transform(fun_x, N)
            assert_almost_equal(result_x[0], result_N)


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


def test_Lagrange_interpolation_basis():
    """ test the InterpolatorDispatcher
    for the evaluation of N by checking the coefficients are
    the same as before """
    xgrid = np.logspace(-3, -1, 10)
    xgrid_lin = np.exp(xgrid)
    polrank = 4
    # Read json file with the old values of the interpolator coefficients
    regression_file = REGRESSION_FOLDER / "Lagrange_interpol.json"
    with open(regression_file, "r") as f:
        reference = json.load(f)
    new_inter = interpolation.InterpolatorDispatcher(xgrid_lin, polrank, log=True)
    for ref, new in zip(reference, new_inter.basis):
        assert ref["polynom_number"] == new.poly_number
        ref_areas = ref["areas"]
        for ref_area, new_area in zip(ref_areas, new):
            assert_almost_equal(ref_area["xmin"], new_area.xmin)
            assert_almost_equal(ref_area["xmax"], new_area.xmax)
            assert_allclose(ref_area["coeffs"], new_area.coefs)


def test_Lagrange_interpolation_log():
    """ Test several points of the Lagrange interpolator """
    xgrid = np.linspace(1e-2, 1, 10)
    polrank = 4
    inter_x = interpolation.InterpolatorDispatcher(
        xgrid, polrank, log=True, mode_N=False
    )
    check_is_interpolator(inter_x, xgrid)
    inter_N = interpolation.InterpolatorDispatcher(
        xgrid, polrank, log=True, mode_N=True
    )
    check_correspondence_interpolators(inter_x, inter_N)


def test_Lagrange_interpolation():
    """ Test several points of the Lagrange interpolator """
    xgrid = np.linspace(0, 1, 10)
    polrank = 4
    inter_x = interpolation.InterpolatorDispatcher(
        xgrid, polrank, log=False, mode_N=False
    )
    check_is_interpolator(inter_x, xgrid)
    inter_N = interpolation.InterpolatorDispatcher(
        xgrid, polrank, log=False, mode_N=True
    )
    check_correspondence_interpolators(inter_x, inter_N)

def test_eq():
    a = interpolation.InterpolatorDispatcher(np.linspace(0, 1, 10), 4, log=False, mode_N=False)
    b = interpolation.InterpolatorDispatcher(np.linspace(0, 1,  9), 4, log=False, mode_N=False)
    assert a != b
    c = interpolation.InterpolatorDispatcher(np.linspace(0, 1, 10), 3, log=False, mode_N=False)
    assert a != c
    d = interpolation.InterpolatorDispatcher(np.linspace(0, 1, 10), 4, log=True,  mode_N=False)
    assert a != d
    e = interpolation.InterpolatorDispatcher(np.linspace(0, 1, 10), 4, log=False, mode_N=False)
    assert a == e
