# -*- coding: utf-8 -*-
"""
    Checks that the operator grid works as intended
    These test can be slow as they require the computation of several values of Q
    But they should be fast as the grid is very small.
    It does *not* test whether the result is correct, it can just test that it is sane
"""

import numpy as np
import pytest

from eko import interpolation
from eko.couplings import Couplings
from eko.evolution_operator.grid import OperatorGrid
from eko.thresholds import ThresholdsAtlas


class TestOperatorGrid:
    def _get_setup(self, use_FFNS):
        theory_card = {
            "alphas": 0.35,
            "alphaem": 0.00781,
            "order": (1, 0),
            "ModEv": "TRN",
            "fact_to_ren_scale_ratio": 1.0,
            "Qref": np.sqrt(2),
            "nfref": None,
            "Q0": np.sqrt(100),
            "nf0": 3,
            "FNS": "FFNS",
            "NfFF": 3,
            "IC": 1,
            "IB": 1,
            "mc": 2.0,
            "mb": 4.0,
            "mt": 105.0,
            "MaxNfPdf": 6,
            "MaxNfAs": 6,
            "HQ": "POLE",
            "ModSV": None,
        }
        operators_card = {
            "Q2grid": [1, 100**2],
            "xgrid": [0.1, 1.0],
            "configs": {
                "interpolation_polynomial_degree": 1,
                "interpolation_is_log": True,
                "ev_op_max_order": (2, 0),
                "ev_op_iterations": 1,
                "backward_inversion": "exact",
                "n_integration_cores": 1,
            },
            "debug": {
                "skip_singlet": True,
                "skip_non_singlet": False,
            },
        }
        if use_FFNS:
            theory_card["FNS"] = "FFNS"
            theory_card["NfFF"] = 3
            theory_card["kcThr"] = np.inf
            theory_card["kbThr"] = np.inf
            theory_card["ktThr"] = np.inf
        else:
            theory_card["FNS"] = "ZM-VFNS"
            theory_card["kcThr"] = 1
            theory_card["kbThr"] = 1
            theory_card["ktThr"] = 1
        return theory_card, operators_card

    def _get_operator_grid(self, use_FFNS=True, theory_update=None):
        theory_card, operators_card = self._get_setup(use_FFNS)
        if theory_update is not None:
            theory_card.update(theory_update)
        # create objects
        basis_function_dispatcher = interpolation.InterpolatorDispatcher.from_dict(
            operators_card
        )
        threshold_holder = ThresholdsAtlas.from_dict(theory_card)
        a_s = Couplings.from_dict(theory_card)
        return OperatorGrid.from_dict(
            theory_card,
            operators_card,
            threshold_holder,
            a_s,
            basis_function_dispatcher,
        )

    def test_sanity(self):
        """Sanity checks for the input"""
        # errors
        with pytest.raises(ValueError):
            theory_card, operators_card = self._get_setup(True)
            basis_function_dispatcher = interpolation.InterpolatorDispatcher.from_dict(
                operators_card
            )
            threshold_holder = ThresholdsAtlas.from_dict(theory_card)
            a_s = Couplings.from_dict(theory_card)
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
        opgrid.compute(3)
        # we can also pass a single number
        opg = opgrid.compute()
        assert len(opg) == 2
        assert all(
            k in op for k in ["operators", "operator_errors"] for op in opg.values()
        )
        opg = opgrid.compute(3)
        assert len(opg) == 1
        assert all(
            k in op for k in ["operators", "operator_errors"] for op in opg.values()
        )

    def test_grid_computation_VFNS(self):
        """Checks that the grid can be computed"""
        opgrid = self._get_operator_grid(False)
        qgrid_check = [3, 5, 200**2]
        operators = opgrid.compute(qgrid_check)
        assert len(operators) == len(qgrid_check)

    def test_mod_expanded(self):
        theory_update = {
            "order": (1, 0),
            "ModSV": "expanded",
        }
        epsilon = 1e-1
        for ffns, nf0 in zip([False, True], [5, 3]):
            theory_update["nf0"] = nf0
            opgrid = self._get_operator_grid(use_FFNS=ffns, theory_update=theory_update)
            opg = opgrid.compute(3)
            theory_update["fact_to_ren_scale_ratio"] = 1.0 + epsilon
            sv_opgrid = self._get_operator_grid(
                use_FFNS=ffns, theory_update=theory_update
            )
            sv_opg = sv_opgrid.compute(3)
            np.testing.assert_allclose(
                opg[3]["operators"], sv_opg[3]["operators"], atol=0.7 * epsilon
            )
