# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate

from eko.operator import Operator, gamma_ns_fact, gamma_singlet_fact, quad_ker
from eko.operator.grid import OperatorGrid, OperatorMaster
from eko.operator.member import OpMember
from eko.operator.physical import PhysicalOperator
from eko.thresholds import ThresholdsConfig
from eko.strong_coupling import StrongCoupling
from eko.interpolation import InterpolatorDispatcher
from eko import anomalous_dimensions as ad
from eko.kernels import non_singlet as ns
from eko.kernels import singlet as s
from eko import mellin
from eko import interpolation


def test_gamma_ns_fact(monkeypatch):
    gamma_ns = np.array([1.0, 0.5])
    monkeypatch.setattr(ad, "gamma_ns", lambda *args: gamma_ns.copy())
    gamma_ns_LO_0 = gamma_ns_fact(0, "NS_p", 1, 3, 0)
    np.testing.assert_allclose(gamma_ns_LO_0, gamma_ns)
    gamma_ns_LO_1 = gamma_ns_fact(0, "NS_p", 1, 3, 1)
    np.testing.assert_allclose(gamma_ns_LO_1, gamma_ns)
    gamma_ns_NLO_1 = gamma_ns_fact(1, "NS_p", 1, 3, 1)
    assert gamma_ns_NLO_1[1] < gamma_ns[1]


def test_gamma_singlet_fact(monkeypatch):
    gamma_s = np.array([1.0, 0.5])
    monkeypatch.setattr(ad, "gamma_singlet", lambda *args: gamma_s.copy())
    gamma_s_LO_0 = gamma_singlet_fact(0, 1, 3, 0)
    np.testing.assert_allclose(gamma_s_LO_0, gamma_s)
    gamma_s_LO_1 = gamma_singlet_fact(0, 1, 3, 1)
    np.testing.assert_allclose(gamma_s_LO_1, gamma_s)
    gamma_s_NLO_1 = gamma_singlet_fact(1, 1, 3, 1)
    assert gamma_s_NLO_1[1] < gamma_s[1]


def test_quad_ker(monkeypatch):
    monkeypatch.setattr(mellin, "Talbot_path", lambda *args: 2) # N=2 is a safe evaluation point
    monkeypatch.setattr(mellin, "Talbot_jac", lambda *args: np.complex(0, np.pi)) # negate mellin prefactor
    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 1)
    monkeypatch.setattr(interpolation, "evaluate_Nx", lambda *args: 1)
    monkeypatch.setattr(ns, "dispatcher", lambda *args: 1.0)
    monkeypatch.setattr(s, "dispatcher", lambda *args: np.identity(2))
    for is_log in [True, False]:
        res_ns = quad_ker(
            u=0,
            order=0,
            mode="NS_p",
            method="",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            a1=1,
            a0=2,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=0,
        )
        np.testing.assert_allclose(res_ns, 1.0)
        res_s = quad_ker(
            u=0,
            order=0,
            mode="S_qq",
            method="",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            a1=1,
            a0=2,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=0,
        )
        np.testing.assert_allclose(res_s, 1.0)
        res_s = quad_ker(
            u=0,
            order=0,
            mode="S_qg",
            method="",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            a1=1,
            a0=2,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=0,
        )
        np.testing.assert_allclose(res_s, 0.0)
    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 0)
    res_ns = quad_ker(
        u=0,
        order=0,
        mode="NS_p",
        method="",
        is_log=True,
        logx=0.0,
        areas=np.zeros(3),
        a1=1,
        a0=2,
        nf=3,
        L=0,
        ev_op_iterations=0,
        ev_op_max_order=0,
    )
    np.testing.assert_allclose(res_ns, 0.0)


class TestOperator:
    def test_labels(self):
        o = Operator(
            dict(order=1, debug_skip_non_singlet=False, debug_skip_singlet=False),
            {},
            3,
            1,
            2,
        )
        assert sorted(o.labels()) == sorted(
            ["NS_p", "NS_m", "S_qq", "S_qg", "S_gq", "S_gg"]
        )
        o = Operator(
            dict(order=1, debug_skip_non_singlet=True, debug_skip_singlet=True),
            {},
            3,
            1,
            2,
        )
        assert sorted(o.labels()) == []

    def test_compute(self, monkeypatch):
        # setup objs
        setup = {
            "alphas": 0.35,
            "PTO": 0,
            "ModEv": "TRN",
            "XIF": 1.0,
            "XIR": 1.0,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "Q2grid": [1, 10],
            "FNS": "FFNS",
            "NfFF": 3,
            "interpolation_xgrid": [0.1, 1.0],
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "debug_skip_singlet": True,
        }
        g = OperatorGrid.from_dict(
            setup,
            ThresholdsConfig.from_dict(setup),
            StrongCoupling.from_dict(setup),
            InterpolatorDispatcher.from_dict(setup),
        )
        m = OperatorMaster(g.config, g.managers, 3)
        o = m.get_op(1, 10)
        # fake quad
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, **kwargs: np.random.rand(2)
        )
        # LO
        o.compute()
        np.testing.assert_allclose(
            o.op_members["NS_m"].value, o.op_members["NS_p"].value
        )
        assert o.op_members["NS_m"].name == "NS_m"
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

    def test_compose(self, monkeypatch):
        # fake compute
        op_members = {}
        shape = (2, 2)
        for n in ["S_qq", "S_qg", "S_gq", "S_gg", "NS_p", "NS_m", "NS_v"]:
            op_members[n] = OpMember(np.random.rand(*shape), np.random.rand(*shape), n)

        def fake_compute(op):
            op.op_members = op_members
            pass

        monkeypatch.setattr(Operator, "compute", fake_compute)
        op1 = Operator({}, {}, 3, 1, 2)
        t = ThresholdsConfig(1, "FFNS", nf=3)
        instruction_set = t.get_composition_path(3, 0)
        ph = op1.compose([], instruction_set, 2)
        assert isinstance(ph, PhysicalOperator)
        # V.V is NS_v
        np.testing.assert_allclose(ph.op_members["V.V"].value, op_members["NS_v"].value)
        assert ph.op_members["V.V"].name == "V.V"
