# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate

from eko import interpolation, mellin
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher
from eko.kernels import non_singlet as ns
from eko.kernels import singlet as s
from eko.scale_variations import b
from eko.strong_coupling import StrongCoupling
from eko.thresholds import ThresholdsAtlas


def test_ns_sv_dispacher():
    """Test to identity"""
    order = 2
    gamma_ns = np.random.random((order + 1, 2, 2))
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        b.non_singlet_dispatcher(gamma_ns, a_s, order, nf, L), 1
    )


def test_singlet_sv_dispacher():
    """Test to identity"""
    order = 2
    gamma_singlet = np.random.random((order + 1, 2, 2))
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        b.singlet_dispatcher(gamma_singlet, a_s, order, nf, L), np.eye(2)
    )


def test_quad_ker(monkeypatch):
    monkeypatch.setattr(
        mellin, "Talbot_path", lambda *args: 2
    )  # N=2 is a safe evaluation point
    monkeypatch.setattr(
        mellin, "Talbot_jac", lambda *args: complex(0, np.pi)
    )  # negate mellin prefactor
    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 1)
    monkeypatch.setattr(interpolation, "evaluate_Nx", lambda *args: 1)
    monkeypatch.setattr(ns, "dispatcher", lambda *args: 1.0)
    monkeypatch.setattr(s, "dispatcher", lambda *args: np.identity(2))
    for is_log in [True, False]:
        res_ns = b.quad_ker(
            u=0,
            order=2,
            mode="NS_p",
            is_log=is_log,
            logx=1.0,
            areas=np.zeros(3),
            a_s=1,
            nf=3,
            L=0,
        )
        np.testing.assert_allclose(res_ns, 1.0)
        res_s = b.quad_ker(
            u=0,
            order=2,
            mode="S_qq",
            is_log=is_log,
            logx=1.0,
            areas=np.zeros(3),
            a_s=1,
            nf=3,
            L=0,
        )
        np.testing.assert_allclose(res_s, 1.0)
        res_s = b.quad_ker(
            u=0,
            order=2,
            mode="S_qg",
            is_log=is_log,
            logx=1.0,
            areas=np.zeros(3),
            a_s=1,
            nf=3,
            L=0,
        )
        np.testing.assert_allclose(res_s, 0.0)
    res_logx = b.quad_ker(
        u=0,
        order=2,
        mode="S_qq",
        is_log=is_log,
        logx=0.0,
        areas=np.zeros(3),
        a_s=1,
        nf=3,
        L=10,
    )
    np.testing.assert_allclose(res_logx, 0.0)


class TestOperator:
    def test_compute(self, monkeypatch):
        # setup objs
        theory_card = {
            "alphas": 0.35,
            "PTO": 0,
            "ModEv": "TRN",
            "fact_to_ren_scale_ratio": np.sqrt(2),
            "Qref": np.sqrt(2),
            "nfref": None,
            "Q0": np.sqrt(2),
            "nf0": 3,
            "FNS": "FFNS",
            "NfFF": 3,
            "IC": 0,
            "IB": 0,
            "mc": 1.0,
            "mb": 4.75,
            "mt": 173.0,
            "kcThr": np.inf,
            "kbThr": np.inf,
            "ktThr": np.inf,
            "MaxNfPdf": 6,
            "MaxNfAs": 6,
            "HQ": "POLE",
            "SV_scheme": "B",
        }
        operators_card = {
            "Q2grid": [1, 10],
            "interpolation_xgrid": [0.1, 1.0],
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
        o = b.ScaleVariationOperator(g.config, g.managers, nf=3, q2=2)
        # fake quad
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, **kwargs: np.random.rand(2)
        )
        # LO
        o.compute()
        assert "NS_m" in o.op_members
        np.testing.assert_allclose(
            o.op_members["NS_m"].value, o.op_members["NS_p"].value
        )
        np.testing.assert_allclose(
            o.op_members["NS_v"].value, o.op_members["NS_p"].value
        )
        # NLO
        o.config["order"] = 1
        o.compute()
        assert not np.allclose(o.op_members["NS_p"].value, o.op_members["NS_m"].value)
        np.testing.assert_allclose(
            o.op_members["NS_v"].value, o.op_members["NS_m"].value
        )
