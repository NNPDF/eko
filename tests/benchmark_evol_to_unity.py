# -*- coding: utf-8 -*-

import numpy as np

from eko.evolution_operator import Operator
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher, make_grid
from eko.strong_coupling import StrongCoupling
from eko.thresholds import ThresholdsAtlas


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
        "interpolation_xgrid": make_grid(20, 10),
        "interpolation_polynomial_degree": 3,
        "interpolation_is_log": True,
        "debug_skip_singlet": False,
        "debug_skip_non_singlet": False,
        "ev_op_max_order": 1,
        "ev_op_iterations": 1,
        "backward_inversion": "exact",
    }

    def test_op_grid(
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
        q20 = 20
        q21 = 30
        o = Operator(g.config, g.managers, 3, q20, q21)
        o_back = Operator(g.config, g.managers, 3, q21, q20)
        o_back.compute()
        o.compute()

        dim = o_back.op_members["NS_v"].value.shape
        for k in ["NS_v"]:
            np.testing.assert_allclose(
                o.op_members[k].value @ o_back.op_members[k].value,
                np.eye(dim[0]),
                atol=3e-2,
                rtol=2e-2,
            )
        # TODO: not passing for singlet?
        a = o_back.op_members["S_qq"].value @ o.op_members["S_qq"].value
        b = o_back.op_members["S_qg"].value @ o.op_members["S_gq"].value
        np.testing.assert_allclose(
            o_back.op_members["S_qq"].value @ o.op_members["S_qq"].value
            + o_back.op_members["S_qg"].value @ o.op_members["S_gq"].value,
            np.eye(dim[0]),
            atol=3e-2,
            rtol=2e-2,
        )
        np.testing.assert_allclose(
            o_back.op_members["S_qq"].value @ o.op_members["S_qg"].value
            + o_back.op_members["S_qg"].value @ o.op_members["S_gg"].value,
            np.zeros(dim),
            atol=3e-2,
            rtol=2e-2,
        )
