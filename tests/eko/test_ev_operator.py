# -*- coding: utf-8 -*-
import copy
import os

import numpy as np
import scipy.integrate

from eko import anomalous_dimensions as ad
from eko import basis_rotation as br
from eko import interpolation, mellin
from eko.couplings import Couplings
from eko.evolution_operator import Operator, quad_ker
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher
from eko.kernels import non_singlet as ns
from eko.kernels import singlet as s
from eko.thresholds import ThresholdsAtlas


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
        res_ns = quad_ker(
            u=0,
            order=(1, 0),
            mode0=br.non_singlet_pids_map["ns+"],
            mode1=0,
            method="",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_ns, 0.0)
        res_s = quad_ker(
            u=0,
            order=(1, 0),
            mode0=100,
            mode1=100,
            method="",
            is_log=is_log,
            logx=0.123,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_s, 1.0)
        res_s = quad_ker(
            u=0,
            order=(1, 0),
            mode0=100,
            mode1=21,
            method="",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_s, 0.0)
    for label in [(br.non_singlet_pids_map["ns+"], 0), (100, 100)]:
        for sv in [2, 3]:
            res_sv = quad_ker(
                u=0,
                order=(1, 0),
                mode0=label[0],
                mode1=label[1],
                method="",
                is_log=True,
                logx=0.123,
                areas=np.zeros(3),
                as1=1,
                as0=2,
                nf=3,
                L=0,
                ev_op_iterations=0,
                ev_op_max_order=(1, 0),
                sv_mode=sv,
                is_threshold=False,
            )
            np.testing.assert_allclose(res_sv, 1.0)

    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 0)
    res_ns = quad_ker(
        u=0,
        order=(1, 0),
        mode0=br.non_singlet_pids_map["ns+"],
        mode1=0,
        method="",
        is_log=True,
        logx=0.0,
        areas=np.zeros(3),
        as1=1,
        as0=2,
        nf=3,
        L=0,
        ev_op_iterations=0,
        ev_op_max_order=(0, 0),
        sv_mode=1,
        is_threshold=False,
    )
    np.testing.assert_allclose(res_ns, 0.0)


theory_card = {
    "alphas": 0.35,
    "alphaem": 0.00781,
    "order": (1, 0),
    "ModEv": "TRN",
    "fact_to_ren_scale_ratio": 1.0,
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
    "ModSV": None,
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
    "n_integration_cores": 1,
}


class TestOperator:
    def test_labels(self):
        o = Operator(
            dict(
                order=(3, 0),
                debug_skip_non_singlet=False,
                debug_skip_singlet=False,
                n_integration_cores=1,
            ),
            {},
            3,
            1,
            2,
        )
        assert sorted(o.labels) == sorted(br.full_labels)
        o = Operator(
            dict(
                order=(2, 0),
                debug_skip_non_singlet=True,
                debug_skip_singlet=True,
                n_integration_cores=1,
            ),
            {},
            3,
            1,
            2,
        )
        assert sorted(o.labels) == []

    def test_n_pools(self):
        excluded_cores = 3
        # make sure we actually have more the those cores (e.g. on github we don't)
        if os.cpu_count() <= excluded_cores:
            return
        o = Operator(
            dict(
                order=(2, 0),
                debug_skip_non_singlet=True,
                debug_skip_singlet=True,
                n_integration_cores=-excluded_cores,
            ),
            {},
            3,
            1,
            10,
        )
        assert o.n_pools == os.cpu_count() - excluded_cores

    def test_exponentiated(self):
        tcard = copy.deepcopy(theory_card)
        tcard["fact_to_ren_scale_ratio"] = 2.0
        tcard["ModSV"] = "exponentiated"
        ocard = copy.deepcopy(operators_card)
        g = OperatorGrid.from_dict(
            tcard,
            ocard,
            ThresholdsAtlas.from_dict(tcard),
            StrongCoupling.from_dict(tcard),
            InterpolatorDispatcher.from_dict(ocard),
        )
        # setup objs
        o = Operator(g.config, g.managers, 3, 2.0, 10.0)
        np.testing.assert_allclose(o.mur2_shift(40.0), 10.0)
        o.compute()
        self.check_lo(o)

    def test_compute_parallel(self, monkeypatch):
        tcard = copy.deepcopy(theory_card)
        ocard = copy.deepcopy(operators_card)
        ocard["n_integration_cores"] = 2
        g = OperatorGrid.from_dict(
            tcard,
            ocard,
            ThresholdsAtlas.from_dict(tcard),
            Couplings.from_dict(tcard),
            InterpolatorDispatcher.from_dict(ocard),
        )
        # setup objs
        o = Operator(g.config, g.managers, 3, 2.0, 10.0)
        # fake quad
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, **kwargs: np.random.rand(2)
        )
        # LO
        o.compute()
        self.check_lo(o)

    def check_lo(self, o):
        assert (br.non_singlet_pids_map["ns-"], 0) in o.op_members
        np.testing.assert_allclose(
            o.op_members[(br.non_singlet_pids_map["ns-"], 0)].value,
            o.op_members[(br.non_singlet_pids_map["ns+"], 0)].value,
        )
        np.testing.assert_allclose(
            o.op_members[(br.non_singlet_pids_map["nsV"], 0)].value,
            o.op_members[(br.non_singlet_pids_map["ns+"], 0)].value,
        )

    def test_compute(self, monkeypatch):
        tcard = copy.deepcopy(theory_card)
        ocard = copy.deepcopy(operators_card)
        g = OperatorGrid.from_dict(
            tcard,
            ocard,
            ThresholdsAtlas.from_dict(tcard),
            Couplings.from_dict(tcard),
            InterpolatorDispatcher.from_dict(ocard),
        )
        # setup objs
        o = Operator(g.config, g.managers, 3, 2.0, 10.0)
        # fake quad
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, **kwargs: np.random.rand(2)
        )
        # LO
        o.compute()
        self.check_lo(o)
        # NLO
        o.order = (2, 0)
        o.compute()
        assert not np.allclose(
            o.op_members[(br.non_singlet_pids_map["ns+"], 0)].value,
            o.op_members[(br.non_singlet_pids_map["ns-"], 0)].value,
        )
        np.testing.assert_allclose(
            o.op_members[(br.non_singlet_pids_map["nsV"], 0)].value,
            o.op_members[(br.non_singlet_pids_map["ns-"], 0)].value,
        )

        # unity operators
        for n in range(1, 3 + 1):
            o1 = Operator(g.config, g.managers, 3, 2.0, 2.0)
            o1.config["order"] = (n, 0)
            o1.compute()
            for k in br.non_singlet_labels:
                assert k in o1.op_members
                np.testing.assert_allclose(o1.op_members[k].value, np.eye(2), err_msg=k)


def test_pegasus_path():
    def quad_ker_pegasus(
        u, order, mode0, method, logx, areas, a1, a0, nf, ev_op_iterations
    ):
        # compute the mellin inversion as done in pegasus
        phi = 3 / 4 * np.pi
        c = 1.9
        n = complex(c + u * np.exp(1j * phi))
        gamma_ns = ad.gamma_ns(order, mode0, n, nf)
        ker = ns.dispatcher(
            order,
            method,
            gamma_ns,
            a1,
            a0,
            nf,
            ev_op_iterations,
        )
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
        return np.imag(np.exp(1j * phi) / np.pi * pj * ker)

    # It might be useful to test with a different function
    # monkeypatch.setattr(ns, "dispatcher", lambda x, *args: np.exp( - x ** 2 ) )
    xgrid = np.geomspace(1e-7, 1, 10)
    int_disp = InterpolatorDispatcher(xgrid, 1, True)
    order = (2, 0)
    mode0 = br.non_singlet_pids_map["ns+"]
    mode1 = 0
    method = ""
    logxs = np.log(int_disp.xgrid_raw)
    a1 = 1
    a0 = 2
    nf = 3
    L = 0
    ev_op_iterations = 10
    for logx in logxs:
        for bf in int_disp:
            res_ns, _ = scipy.integrate.quad(
                quad_ker,
                0.5,
                1.0,
                args=(
                    order,
                    mode0,
                    mode1,
                    method,
                    int_disp.log,
                    logx,
                    bf.areas_representation,
                    a1,
                    a0,
                    nf,
                    L,
                    ev_op_iterations,
                    10,
                    0,
                    False,
                ),
                epsabs=1e-12,
                epsrel=1e-5,
                limit=100,
                full_output=1,
            )[:2]

            res_test, _ = scipy.integrate.quad(
                quad_ker_pegasus,
                0,
                np.inf,
                args=(
                    order,
                    mode0,
                    method,
                    logx,
                    bf.areas_representation,
                    a1,
                    a0,
                    nf,
                    ev_op_iterations,
                ),
                epsabs=1e-12,
                epsrel=1e-5,
                limit=100,
                full_output=1,
            )[:2]

            np.testing.assert_allclose(res_ns, res_test, rtol=2e-6)
