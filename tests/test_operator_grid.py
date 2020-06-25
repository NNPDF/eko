# -*- coding: utf-8 -*-
"""
    Checks that the operator grid works as intended
    These test can be slow as they require the computation of several values of Q
    But they should be fast as the grid is very small.
    It does *not* test whether the result is correct, it can just test that it is sane
"""

import pytest
import numpy as np
import eko.interpolation as interpolation
from eko.strong_coupling import StrongCoupling
from eko.constants import Constants
from eko.thresholds import ThresholdsConfig
from eko.kernel_generation import KernelDispatcher
from eko.operator_grid import OperatorGrid


class TestOperatorGrid:
    def _get_setup(self, use_FFNS):
        setup = {
            "alphas": 0.35,
            "PTO": 0,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "interpolation_xgrid": [0.1, 1.0],
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
        }
        if use_FFNS:
            setup["FNS"] = "FFNS"
            setup["NfFF"] = 3
        else:
            setup["FNS"] = "ZM-VFNS"
            setup["mc"] = 2
            setup["mb"] = 4
            setup["mt"] = 100
        return setup

    def _get_pdf(self):
        basis = ["V", "V3", "V8", "V15", "T3", "T15", "S", "g"]
        len_grid = len(self._get_setup(True)["interpolation_xgrid"])
        pdf_m = {}
        for i in basis:
            pdf_m[i] = np.random.rand(len_grid)
            pdf_m[i].sort()
        pdf = {"metadata": "evolbasis", "members": pdf_m}
        return pdf

    def _get_operator_grid(self, use_FFNS=True):
        setup = self._get_setup(use_FFNS)
        # create objects
        basis_function_dispatcher = interpolation.InterpolatorDispatcher.from_dict(
            setup
        )
        threshold_holder = ThresholdsConfig.from_dict(setup)
        constants = Constants()
        kernel_dispatcher = KernelDispatcher(basis_function_dispatcher, constants)
        a_s = StrongCoupling.from_dict(setup, constants, threshold_holder)
        return OperatorGrid(
            threshold_holder, a_s, kernel_dispatcher, setup["interpolation_xgrid"]
        )

    def test_sanity(self):
        """ Sanity checks for the input"""
        opgrid = self._get_operator_grid(False)
        # Check that an operator grid with the correct number of regions was created
        nregs = len(opgrid._op_masters)  # pylint: disable=protected-access
        assert nregs == 3 + 1
        # errors
        with pytest.raises(ValueError):
            opgrid.set_q2_limits(-1, 4)
        with pytest.raises(ValueError):
            opgrid.set_q2_limits(-1, -4)
        with pytest.raises(ValueError):
            opgrid.set_q2_limits(4, 1)

    def test_compute_q2grid(self):
        opgrid = self._get_operator_grid()
        # q2 has not be precomputed - but should work nevertheless
        opgrid.get_op_at_q2(3)
        # we can also pass a single number
        opgrid.compute_q2grid(3)
        # errors
        with pytest.raises(ValueError):
            bad_grid = [100, -6, 3]
            _ = opgrid.compute_q2grid(bad_grid)

    def test_grid_computation_VFNS(self):
        """ Checks that the grid can be computed """
        opgrid = self._get_operator_grid(False)
        qgrid_check = [3, 5]
        operators = opgrid.compute_q2grid(qgrid_check)
        assert len(operators) == len(qgrid_check)
        # Check that the operators can act on pdfs
        pdf = self._get_pdf()
        _return_1 = operators[0](pdf)
