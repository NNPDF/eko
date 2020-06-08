# -*- coding: utf-8 -*-
# pylint:disable=protected-access
# Test interpolation
import json
import pathlib

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from eko import t_complex
from eko import interpolation
import eko.mellin as mellin

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")

# for the numeric comparision to work, keep in mind that in Python3 the default precision is
# np.float64

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
    logxinv = np.log(0.9e-2)  # < 1e-2, to trick skipping
    for fun_x, fun_N in zip(inter_x, inter_N):
        for N in ngrid:
            result_N = fun_N(N, logxinv) * np.exp(N * logxinv)
            result_x = mellin.mellin_transform(fun_x, N)
            assert_almost_equal(result_x[0], result_N)

def test_Lagrange_interpolation_basis():
    """ test the InterpolatorDispatcher
    for the evaluation of N by checking the coefficients are
    the same as before """
    xgrid = np.logspace(-3, -1, 10)
    xgrid_lin = np.exp(xgrid)
    poly_deg = 4
    # Read json file with the old values of the interpolator coefficients
    regression_file = REGRESSION_FOLDER / "Lagrange_interpol.json"
    with open(regression_file, "r") as f:
        reference = json.load(f)
    new_inter = interpolation.InterpolatorDispatcher(xgrid_lin, poly_deg, log=True)
    for ref, new in zip(reference, new_inter.basis):
        assert ref["polynom_number"] == new.poly_number
        ref_areas = ref["areas"]
        for ref_area, new_area in zip(ref_areas, new):
            assert_almost_equal(ref_area["xmin"], new_area.xmin)
            assert_almost_equal(ref_area["xmax"], new_area.xmax)
            assert_allclose(ref_area["coeffs"], new_area.coefs)

class TestInterpolatorDispatcher:
    def test_init(self):
        # errors
        with pytest.raises(ValueError):
            interpolation.InterpolatorDispatcher([0.1, 0.1, 0.2], 1)
        with pytest.raises(ValueError):
            interpolation.InterpolatorDispatcher([0.1], 1)
        with pytest.raises(ValueError):
            interpolation.InterpolatorDispatcher([0.1, 0.2], 0)
        with pytest.raises(ValueError):
            interpolation.InterpolatorDispatcher([0.1, 0.2], 2)

    def test_eq(self):
        a = interpolation.InterpolatorDispatcher(
            np.linspace(0.1, 1, 10), 4, log=False, mode_N=False
        )
        b = interpolation.InterpolatorDispatcher(
            np.linspace(0.1, 1, 9), 4, log=False, mode_N=False
        )
        assert a != b
        c = interpolation.InterpolatorDispatcher(
            np.linspace(0.1, 1, 10), 3, log=False, mode_N=False
        )
        assert a != c
        d = interpolation.InterpolatorDispatcher(
            np.linspace(0.1, 1, 10), 4, log=True, mode_N=False
        )
        assert a != d
        e = interpolation.InterpolatorDispatcher(
            np.linspace(0.1, 1, 10), 4, log=False, mode_N=False
        )
        assert a == e
        # via dict
        dd = a.to_dict()
        assert isinstance(dd,dict)
        assert a == interpolation.InterpolatorDispatcher.from_dict(dd)

    def test_iter(self):
        xgrid = np.linspace(0.1, 1, 10)
        poly_degree = 4
        inter_x = interpolation.InterpolatorDispatcher(xgrid, poly_degree)
        for bf,k in zip(inter_x, range(len(xgrid))):
            assert bf == inter_x[k]

    def test_get_interpolation(self):
        xg = [0.5, 1.0]
        inter_x = interpolation.InterpolatorDispatcher(xg, 1,False,False)
        i = inter_x.get_interpolation(xg)
        np.testing.assert_array_almost_equal(i,np.eye(len(xg)))
        # .75 is exactly inbetween
        i = inter_x.get_interpolation([0.75])
        np.testing.assert_array_almost_equal(i,[[0.5,0.5]])

    def test_evaluate_x(self):
        xgrid = np.linspace(0.1, 1, 10)
        poly_degree = 4
        for log in [True, False]:
            inter_x = interpolation.InterpolatorDispatcher(
                xgrid, poly_degree, log=log, mode_N=False
            )
            inter_N = interpolation.InterpolatorDispatcher(
                xgrid, poly_degree, log=log, mode_N=True
            )
            for x in [0.2, 0.5]:
                for bx, bN in zip(inter_x, inter_N):
                    assert_almost_equal(bx.evaluate_x(x), bN.evaluate_x(x))

    def test_math(self):
        """ Test math properties of interpolator """
        xgrid = np.linspace(0.09, 1, 10)
        poly_deg = 4
        for log in [True, False]:
            inter_x = interpolation.InterpolatorDispatcher(
                xgrid, poly_deg, log=log, mode_N=False
            )
            check_is_interpolator(inter_x, xgrid)
            inter_N = interpolation.InterpolatorDispatcher(
                xgrid, poly_deg, log=log, mode_N=True
            )
            check_correspondence_interpolators(inter_x, inter_N)

@pytest.mark.skip # TODO will fix tomorrow
class TestBasisFunction:
    def test_is_below_x(self):
        xgrid = [.1,.4,.7,1.0]
        poly_deg = 2
        for log in [False, True]:
            inter_x = interpolation.InterpolatorDispatcher(
                xgrid, poly_deg, log=log
            )
            for bf in inter_x:
                assert bf.is_below_x(xgrid[0]) == (bf.poly_number == len(xgrid) - 1)

def test_reference_indices():
    xgrid = np.linspace(0.1, 1, 10)
    # polynomial_degree = 2
    a = interpolation.Area(0, 0, (0, 1), xgrid)
    assert list(a._reference_indices()) == [1]
    a = interpolation.Area(0, 1, (0, 1), xgrid)
    assert list(a._reference_indices()) == [0]
    # polynomial_degree = 3
    a = interpolation.Area(0, 0, (0, 2), xgrid)
    assert list(a._reference_indices()) == [1, 2]
    a = interpolation.Area(0, 1, (0, 2), xgrid)
    assert list(a._reference_indices()) == [0, 2]
    a = interpolation.Area(0, 2, (0, 2), xgrid)
    assert list(a._reference_indices()) == [0, 1]
    # errors
    with pytest.raises(ValueError):
        a = interpolation.Area(0, 3, (0, 2), xgrid)
