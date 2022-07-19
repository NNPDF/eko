# -*- coding: utf-8 -*-
import numpy as np
import pytest

from eko import basis_rotation as br
from eko import compatibility
from eko.couplings import Couplings
from eko.evolution_operator import Operator
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher
from eko.thresholds import ThresholdsAtlas

# from eko.matching_conditions.operator_matrix_element import OperatorMatrixElement


@pytest.mark.isolated
class BenchmarkBackwardForward:
    # setup objs
    theory_card = {
        "alphas": 0.35,
        "alphaqed": 0.007496,
        "PTO": 0,
        "QED": 0,
        "ModEv": "EXA",
        "fact_to_ren_scale_ratio": 1.0,
        "Qref": np.sqrt(2),
        "nfref": None,
        "Q0": np.sqrt(2),
        "nf0": 4,
        "IC": 1,
        "IB": 1,
        "mc": 1.0,
        "mb": 4.75,
        "mt": 173.0,
        "kcThr": 1.0,
        "kbThr": 1.0,
        "ktThr": 1.0,
        "MaxNfPdf": 6,
        "MaxNfAs": 6,
        "HQ": "POLE",
        "ModSV": None,
    }
    operators_card = {
        "Q2grid": [10],
        # here you need a very dense grid
        "xgrid": np.linspace(1e-1, 1, 30),
        # "xgrid": make_grid(30,30, x_min=1e-3),
        "configs": {
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "ev_op_max_order": (2, 0),
            "ev_op_iterations": 1,
            "backward_inversion": "exact",
            "n_integration_cores": 1,
        },
        "debug": {
            "skip_singlet": False,
            "skip_non_singlet": False,
        },
    }
    new_theory, new_operators = compatibility.update(theory_card, operators_card)
    g = OperatorGrid.from_dict(
        new_theory,
        new_operators,
        ThresholdsAtlas.from_dict(new_theory),
        Couplings.from_dict(new_theory),
        InterpolatorDispatcher.from_dict(new_operators),
    )

    def test_operator_grid(
        # test that eko_forward @ eko_backward gives ID matrix or zeros
        self,
    ):
        g = OperatorGrid.from_dict(
            self.new_theory,
            self.new_operators,
            ThresholdsAtlas.from_dict(self.new_theory),
            Couplings.from_dict(self.new_theory),
            InterpolatorDispatcher.from_dict(self.new_operators),
        )
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
