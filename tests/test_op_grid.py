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
from eko.thresholds import ThresholdsAtlas
from eko.operator.grid import OperatorGrid


class TestOperatorGrid:
    def _get_setup(self, use_FFNS):
        theory_card = {
            "alphas": 0.35,
            "PTO": 0,
            "ModEv": "TRN",
            "XIF": 1.0,
            "XIR": 1.0,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "FNS": "FFNS",
            "NfFF": 3,
            "IC": 1,
            "mc": 1.0,
            "mb": 4.75,
            "mt": 173.0,
            "kcThr": 0,
            "kbThr": np.inf,
            "ktThr": np.inf,
        }
        operators_card = {
            "Q2grid": [1, 10],
            "interpolation_xgrid": [0.1, 1.0],
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "debug_skip_singlet": True,
            "debug_skip_non_singlet": False,
            "ev_op_max_order": 1,
            "ev_op_iterations": 1,
        }
        if use_FFNS:
            theory_card["FNS"] = "FFNS"
            theory_card["NfFF"] = 3
        else:
            theory_card["FNS"] = "ZM-VFNS"
            theory_card["mc"] = 2
            theory_card["mb"] = 4
            theory_card["mt"] = 100
            theory_card["kcThr"] = 1
            theory_card["kbThr"] = 1
            theory_card["ktThr"] = 1
        return theory_card, operators_card

    def _get_operator_grid(self, use_FFNS=True):
        theory_card, operators_card = self._get_setup(use_FFNS)
        # create objects
        basis_function_dispatcher = interpolation.InterpolatorDispatcher.from_dict(
            operators_card
        )
        threshold_holder = ThresholdsAtlas.from_dict(theory_card)
        a_s = StrongCoupling.from_dict(theory_card, threshold_holder)
        return OperatorGrid.from_dict(
            theory_card,
            operators_card,
            threshold_holder,
            a_s,
            basis_function_dispatcher,
        )

    def test_sanity(self):
        """ Sanity checks for the input"""
        opgrid = self._get_operator_grid(False)

        # errors
        with pytest.raises(ValueError):
            opgrid.set_q2_limits(-1, 4)
        with pytest.raises(ValueError):
            opgrid.set_q2_limits(-1, -4)
        with pytest.raises(ValueError):
            opgrid.set_q2_limits(4, 1)
        with pytest.raises(ValueError):
            theory_card, operators_card = self._get_setup(True)
            basis_function_dispatcher = interpolation.InterpolatorDispatcher.from_dict(
                operators_card
            )
            threshold_holder = ThresholdsAtlas.from_dict(theory_card)
            a_s = StrongCoupling.from_dict(theory_card, threshold_holder)
            theory_card.update({"ModEv": "wrong"})
            OperatorGrid.from_dict(
                theory_card,
                operators_card,
                threshold_holder,
                a_s,
                basis_function_dispatcher,
            )

    def test_compute_q2grid(self):
        opgrid = self._get_operator_grid()
        # q2 has not be precomputed - but should work nevertheless
        opgrid.get_op_at_q2(3)
        # we can also pass a single number
        opg = opgrid.compute_q2grid()
        assert len(opg) == 2
        opg = opgrid.compute_q2grid(3)
        assert len(opg) == 1
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
