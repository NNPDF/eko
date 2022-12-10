"""
    Checks that the operator grid works as intended
    These test can be slow as they require the computation of several values of Q
    But they should be fast as the grid is very small.
    It does *not* test whether the result is correct, it can just test that it is sane
"""

import numpy as np

import eko.io.types
from eko.runner import legacy


class TestOperatorGrid:
    def test_compute_q2grid(self, theory_ffns, operator_card, tmp_path):
        mugrid = np.array([10.0, 100.0])
        operator_card._mugrid = mugrid
        operator_card._mu2grid = None
        opgrid = legacy.Runner(
            theory_ffns(3), operator_card, path=tmp_path / "eko.tar"
        ).op_grid
        # q2 has not be precomputed - but should work nevertheless
        opgrid.compute(3)
        # we can also pass a single number
        opg = opgrid.compute()
        assert len(opg) == len(mugrid)
        assert all(k in op for k in ["operator", "error"] for op in opg.values())
        opg = opgrid.compute(3)
        assert len(opg) == 1
        assert all(k in op for k in ["operator", "error"] for op in opg.values())

    def test_grid_computation_VFNS(self, theory_card, operator_card, tmp_path):
        """Checks that the grid can be computed"""
        opgrid = legacy.Runner(
            theory_card, operator_card, path=tmp_path / "eko.tar"
        ).op_grid
        qgrid_check = [3, 5, 200**2]
        operators = opgrid.compute(qgrid_check)
        assert len(operators) == len(qgrid_check)

    def test_mod_expanded(self, theory_card, theory_ffns, operator_card, tmp_path):
        operator_card.configs.scvar_method = eko.io.types.ScaleVariationsMethod.EXPANDED
        theory_update = {
            "order": (1, 0),
            "ModSV": "expanded",
        }
        epsilon = 1e-1
        for is_ffns, nf0 in zip([False, True], [5, 3]):
            if is_ffns:
                theory = theory_ffns(nf0)
            else:
                theory = theory_card
            theory.order = (1, 0)
            theory.num_flavs_init = nf0
            theory.matching
            theory.fact_to_ren = 1.0
            opgrid = legacy.Runner(
                theory, operator_card, path=tmp_path / "eko.tar"
            ).op_grid
            opg = opgrid.compute(3)
            theory.fact_to_ren = 1.0 + epsilon
            theory_update["fact_to_ren_scale_ratio"] = 1.0 + epsilon
            sv_opgrid = legacy.Runner(
                theory, operator_card, path=tmp_path / "eko.tar"
            ).op_grid
            sv_opg = sv_opgrid.compute(3)
            np.testing.assert_allclose(
                opg[3]["operator"], sv_opg[3]["operator"], atol=2.5 * epsilon
            )
