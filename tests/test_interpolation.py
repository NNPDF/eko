# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko import t_complex
from eko import interpolation
import eko.mellin as mellin

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
        assert isinstance(dd, dict)
        assert a == interpolation.InterpolatorDispatcher.from_dict(dd)

    def test_iter(self):
        xgrid = np.linspace(0.1, 1, 10)
        poly_degree = 4
        inter_x = interpolation.InterpolatorDispatcher(xgrid, poly_degree)
        for bf, k in zip(inter_x, range(len(xgrid))):
            assert bf == inter_x[k]

    def test_get_interpolation(self):
        xg = [0.5, 1.0]
        inter_x = interpolation.InterpolatorDispatcher(xg, 1, False, False)
        i = inter_x.get_interpolation(xg)
        np.testing.assert_array_almost_equal(i, np.eye(len(xg)))
        # .75 is exactly inbetween
        i = inter_x.get_interpolation([0.75])
        np.testing.assert_array_almost_equal(i, [[0.5, 0.5]])

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


class TestBasisFunction:
    def test_init(self):
        # errors
        with pytest.raises(ValueError):
            interpolation.BasisFunction([.1,1.],0,[])

    def test_eval_N(self):
        xg = [0., 1.]
        inter_N = interpolation.InterpolatorDispatcher(xg,1,log=False)
        # p_0(x) = 1-x -> \tilde p_0(N) = 1/N - 1/(N+1)
        p0N = inter_N[0]
        assert len(p0N.areas) == 1
        p0_cs_ref = [1, -1]
        for act_c, res_c in zip(p0N.areas[0], p0_cs_ref):
            assert_almost_equal(act_c,res_c)
        p0Nref = lambda N,lnx:(1/N-1/(N+1))*np.exp(-N*lnx)
        #assert_almost_equal(p0N(1.,0),p0Nref(1.,0))
        # p_1(x) = x -> \tilde p_1(N) = 1/(N+1)
        p1N = inter_N[1]
        assert len(p1N.areas) == 1
        p1_cs_ref = [0,1]
        for act_c, res_c in zip(p1N.areas[0], p1_cs_ref):
            assert_almost_equal(act_c,res_c)
        p1Nref = lambda N,lnx:(1/(N+1))*np.exp(-N*lnx)
        assert_almost_equal(p1N(1.,0),p1Nref(1.,0))

    def test_is_below_x(self):
        for log in [False, True]:
            for cfg in [
                {"xg": [0.1, 0.2, 1.0], "pd": 1, "x": 0.2, "res": [True, False, False]},
                {
                    "xg": [0.1, 0.2, 1.0],
                    "pd": 2,
                    "x": 0.2,
                    "res": [False, False, False],
                },
                {  # this tests the correct attributions of blocks
                    # i.e. in case of doubt use the one higher
                    # | | | | |
                    # |B0-|
                    #   |B1-|
                    #     |B2-|
                    # A0 -> B0, A1 -> B1, A2 -> B2, A3 -> B2
                    "xg": [0.1, 0.2, 0.3, 0.4, 1.0],
                    "pd": 2,
                    "x": 0.35,  # -> A2
                    "res": [True, True, False, False, False],
                },
            ]:
                inter_x = interpolation.InterpolatorDispatcher(
                    cfg["xg"], cfg["pd"], log=log
                )
                actual = [bf.is_below_x(cfg["x"]) for bf in inter_x]
                assert actual == cfg["res"]

class TestArea:
    def test_iter(self):
        xg = [.1, 1.]
        a = interpolation.Area(0, 0, (0, 1), xg) # = p_0(x) with degree O(x)
        # p_0(x) = 1/0.9(1-x) because p_0(.1)=.9/.9=1 and p_0(1) = 0
        res_cs = [1/0.9, -1/0.9]
        for act_c, res_c in zip(a, res_cs):
            assert_almost_equal(act_c,res_c)

    def test_reference_indices(self):
        xgrid = np.linspace(0.1, 1, 10)
        # polynomial_degree = 1
        a = interpolation.Area(0, 0, (0, 1), xgrid)
        assert list(a._reference_indices()) == [1] #pylint: disable=protected-access
        a = interpolation.Area(0, 1, (0, 1), xgrid)
        assert list(a._reference_indices()) == [0] #pylint: disable=protected-access
        # polynomial_degree = 2
        a = interpolation.Area(0, 0, (0, 2), xgrid)
        assert list(a._reference_indices()) == [1, 2] #pylint: disable=protected-access
        a = interpolation.Area(0, 1, (0, 2), xgrid)
        assert list(a._reference_indices()) == [0, 2] #pylint: disable=protected-access
        a = interpolation.Area(0, 2, (0, 2), xgrid)
        assert list(a._reference_indices()) == [0, 1] #pylint: disable=protected-access
        # errors
        with pytest.raises(ValueError):
            a = interpolation.Area(0, 3, (0, 2), xgrid)
