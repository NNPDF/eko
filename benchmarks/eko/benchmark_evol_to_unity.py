import pathlib
from math import nan

import numpy as np
import pytest

from eko import basis_rotation as br
from eko.evolution_operator import Operator
from eko.interpolation import XGrid
from eko.io import types
from eko.io.runcards import OperatorCard, TheoryCard
from eko.runner.legacy import Runner

# from eko.matching_conditions.operator_matrix_element import OperatorMatrixElement


def update_cards(theory: TheoryCard, operator: OperatorCard):
    theory.couplings = types.CouplingsRef(
        alphas=types.FloatRef(value=0.35, scale=float(np.sqrt(2))),
        alphaem=types.FloatRef(value=0.007496, scale=nan),
        max_num_flavs=6,
        num_flavs_ref=None,
    )
    theory.num_flavs_init = 4
    theory.intrinsic_flavors = [4, 5]
    theory.quark_masses.c.value = 1.0
    theory.quark_masses.b.value = 4.75
    theory.quark_masses.t.value = 173.0
    operator.mu0 = float(np.sqrt(2))
    operator.mu2grid = [10]
    operator.rotations.xgrid = XGrid(np.linspace(1e-1, 1, 30))
    operator.configs.interpolation_polynomial_degree = 1
    operator.configs.ev_op_max_order = (2, 0)
    operator.configs.ev_op_iterations = 1
    operator.configs.inversion_method = types.InversionMethod.EXACT
    operator.configs.n_integration_cores = 1


@pytest.mark.isolated
class BenchmarkBackwardForward:
    def test_operator_grid(
        self,
        theory_card: TheoryCard,
        operator_card: OperatorCard,
        tmp_path: pathlib.Path,
    ):
        """Test that eko_forward @ eko_backward gives ID matrix or zeros."""
        update_cards(theory_card, operator_card)
        g = Runner(
            theory_card=theory_card,
            operators_card=operator_card,
            path=tmp_path / "eko.tar",
        ).op_grid

        q20 = 30
        q21 = 50
        nf = 4
        o = Operator(g.config, g.managers, nf, q20, q21)
        o_back = Operator(g.config, g.managers, nf, q21, q20)
        o.compute()
        o_back.compute()

        dim = o_back.op_members[(br.non_singlet_pids_map["nsV"], 0)].value.shape
        for k in ["nsV", "ns-", "ns+"]:
            np.testing.assert_allclose(
                o.op_members[(br.non_singlet_pids_map[k], 0)].value
                @ o_back.op_members[(br.non_singlet_pids_map[k], 0)].value,
                np.eye(dim[0]),
                atol=7e-2,
            )
        # qq
        np.testing.assert_allclose(
            o_back.op_members[(100, 100)].value @ o.op_members[(100, 100)].value
            + o_back.op_members[(100, 21)].value @ o.op_members[(21, 100)].value,
            np.eye(dim[0]),
            atol=7e-2,
        )
        # qg
        np.testing.assert_allclose(
            o_back.op_members[(100, 100)].value @ o.op_members[(100, 21)].value
            + o_back.op_members[(100, 21)].value @ o.op_members[(21, 21)].value,
            np.zeros(dim),
            atol=7e-4,
        )
        # gg, check last two rows separately
        gg_id = (
            o_back.op_members[(21, 21)].value @ o.op_members[(21, 21)].value
            + o_back.op_members[(21, 100)].value @ o.op_members[(100, 21)].value
        )
        np.testing.assert_allclose(
            gg_id[:-2],
            np.eye(dim[0])[:-2],
            atol=3e-2,
        )
        np.testing.assert_allclose(
            gg_id[-2:],
            np.eye(dim[0])[-2:],
            atol=11.2e-2,
        )
        # gq
        np.testing.assert_allclose(
            o_back.op_members[(21, 21)].value @ o.op_members[(21, 100)].value
            + o_back.op_members[(21, 100)].value @ o.op_members[(100, 100)].value,
            np.zeros(dim),
            atol=2e-3,
        )

    # def test_matching_grid(
    #     self,
    # ):
    #     # test matching_ome_forward @ matching_ome_backward gives ID matrix or zeros
    #     # Only singlet matrix is tested
    #     q2 = 5
    #     L = 0
    #     ome = OperatorMatrixElement(self.g.config, self.g.managers, is_backward=False)
    #     ome_back = OperatorMatrixElement(self.g.config, self.g.managers, is_backward=True)
    #     ome.compute( q2, L)
    #     ome_back.compute(q2, L)

    #     dim = ome.op_members[(100, 100)].value.shape
    #     ome_tensor = np.zeros((3,3,dim[0],dim[0]))
    #     ome_tensor_back = ome_tensor
    #     idx_dict = dict(zip(["g", "q", "H"],[0,1,2]))
    #     for p1, j in idx_dict.items():
    #         for p2, k in idx_dict.items():
    #             ome_tensor[j,k] = ome.op_members[f"S_{p1}{p2}"].value
    #             ome_tensor_back[j,k] = ome_back.op_members[f"S_{p1}{p2}"].value

    #     ome_product = np.einsum("abjk,bckl -> acjl", ome_tensor_back, ome_tensor)
    #     for j, line in enumerate(ome_product):
    #         for k, elem in enumerate(line):
    #             test_matrix = np.zeros(dim) if j != k else np.eye(dim[0])
    #             np.testing.assert_allclose(
    #                 elem,
    #                 test_matrix,
    #                 atol=7e-2,
    #             )
