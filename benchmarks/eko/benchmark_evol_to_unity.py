import pathlib

import numpy as np
import pytest

from eko import basis_rotation as br
from eko.evolution_operator import Operator
from eko.interpolation import XGrid
from eko.io import types
from eko.io.runcards import OperatorCard, TheoryCard
from eko.matchings import Segment
from eko.quantities.couplings import CouplingsInfo
from eko.runner.legacy import Runner

# from ekore.matching_conditions.operator_matrix_element import OperatorMatrixElement


def update_cards(theory: TheoryCard, operator: OperatorCard):
    theory.couplings = CouplingsInfo(
        alphas=0.35,
        alphaem=0.007496,
        ref=(float(np.sqrt(2)), 4),
    )
    theory.heavy.masses.c.value = 1.0
    theory.heavy.masses.b.value = 4.75
    theory.heavy.masses.t.value = 173.0
    operator.init = (float(np.sqrt(2)), 4)
    operator.mugrid = [(10, 5)]
    operator.xgrid = XGrid(np.linspace(1e-1, 1, 30))
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

        seg = Segment(30, 50, 4)
        seg_back = Segment(50, 30, 4)
        o = Operator(g.config, g.managers, seg)
        o_back = Operator(g.config, g.managers, seg_back)
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
