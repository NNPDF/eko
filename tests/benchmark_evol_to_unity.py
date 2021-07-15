# -*- coding: utf-8 -*-

import numpy as np

from eko.evolution_operator import Operator
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher, make_grid
from eko.strong_coupling import StrongCoupling
from eko.thresholds import ThresholdsAtlas
from eko.matching_conditions.operator_matrix_element import OperatorMatrixElement


class TestBackwardForward:
    # setup objs
    theory_card = {
        "alphas": 0.35,
        "PTO": 0,
        "ModEv": "EXA",
        "fact_to_ren_scale_ratio": 1.0,
        "Qref": np.sqrt(2),
        "nfref": None,
        "Q0": np.sqrt(2),
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
    }
    operators_card = {
        "Q2grid": [10],
        # here you need a very dense grid
        "interpolation_xgrid": np.linspace(1e-1,5e-1,30),
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "debug_skip_singlet": False,
        "debug_skip_non_singlet": False,
        "ev_op_max_order": 1,
        "ev_op_iterations": 1,
        "backward_inversion": "exact",
    }
    g = OperatorGrid.from_dict(
        theory_card,
        operators_card,
        ThresholdsAtlas.from_dict(theory_card),
        StrongCoupling.from_dict(theory_card),
        InterpolatorDispatcher.from_dict(operators_card),
    )

    def test_operator_grid(
        # test that eko_forward @ eko_backward gives ID matrix or zeros
        self,
    ):
        g = OperatorGrid.from_dict(
            self.theory_card,
            self.operators_card,
            ThresholdsAtlas.from_dict(self.theory_card),
            StrongCoupling.from_dict(self.theory_card),
            InterpolatorDispatcher.from_dict(self.operators_card),
        )
        q20 = 30
        q21 = 50
        nf = 4
        o = Operator(g.config, g.managers, nf, q20, q21)
        o_back = Operator(g.config, g.managers, nf, q21, q20)
        o.compute()
        o_back.compute()

        dim = o_back.op_members["NS_v"].value.shape
        for k in ["NS_v", "NS_m", "NS_p"]:
            np.testing.assert_allclose(
                o.op_members[k].value @ o_back.op_members[k].value,
                np.eye(dim[0]),
                atol=7e-2,
            )
        # qq
        np.testing.assert_allclose(
            o_back.op_members["S_qq"].value @ o.op_members["S_qq"].value
            + o_back.op_members["S_qg"].value @ o.op_members["S_gq"].value,
            np.eye(dim[0]),
            atol=7e-2,
        )
        # qg
        np.testing.assert_allclose(
            o_back.op_members["S_qq"].value @ o.op_members["S_qg"].value
            + o_back.op_members["S_qg"].value @ o.op_members["S_gg"].value,
            np.zeros(dim),
            atol=7e-4,
        )
        # gg
        np.testing.assert_allclose(
            o_back.op_members["S_gg"].value @ o.op_members["S_gg"].value
            + o_back.op_members["S_gq"].value @ o.op_members["S_qg"].value,
            np.eye(dim[0]),
            atol=9e-2,
        )
        # gq
        np.testing.assert_allclose(
            o_back.op_members["S_gg"].value @ o.op_members["S_gq"].value
            + o_back.op_members["S_gq"].value @ o.op_members["S_qq"].value,
            np.zeros(dim),
            atol=3e-4,
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

    #     dim = ome.ome_members["S_qq"].value.shape
    #     ome_tensor = np.zeros((3,3,dim[0],dim[0]))
    #     ome_tensor_back = ome_tensor
    #     idx_dict = dict(zip(["g", "q", "H"],[0,1,2]))
    #     for p1, j in idx_dict.items():
    #         for p2, k in idx_dict.items():
    #             ome_tensor[j,k] = ome.ome_members[f"S_{p1}{p2}"].value
    #             ome_tensor_back[j,k] = ome_back.ome_members[f"S_{p1}{p2}"].value

    #     ome_product = np.einsum("abjk,bckl -> acjl", ome_tensor_back, ome_tensor)
    #     for j, line in enumerate(ome_product):
    #         for k, elem in enumerate(line):
    #             test_matrix = np.zeros(dim) if j != k else np.eye(dim[0])
    #             np.testing.assert_allclose(
    #                 elem,
    #                 test_matrix,
    #                 atol=7e-2,
    #             )
