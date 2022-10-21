# -*- coding: utf-8 -*-
import copy
import os

import numpy as np
import pytest
import scipy.integrate

from eko import anomalous_dimensions as ad
from eko import basis_rotation as br
from eko import interpolation, mellin
from eko.couplings import Couplings
from eko.evolution_operator import Operator, quad_ker
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher, XGrid
from eko.kernels import QEDnon_singlet as qed_ns
from eko.kernels import non_singlet as ns
from eko.kernels import singlet as s
from eko.kernels import utils
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
    monkeypatch.setattr(qed_ns, "dispatcher", lambda *args: 1.0)
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
            aem_list=[0.00058],
            alphaem_running=False,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_ns, 0.0)
        res_ns = quad_ker(
            u=0,
            order=(3, 1),
            mode0=br.non_singlet_pids_map["ns+u"],
            mode1=0,
            method="",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            aem_list=[0.00058],
            alphaem_running=False,
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
            aem_list=[0.00058],
            alphaem_running=False,
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
            order=(1, 1),
            mode0=100,
            mode1=100,
            method="iterate-exact",
            is_log=is_log,
            logx=0.123,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            aem_list=[0.00058],
            alphaem_running=False,
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
            aem_list=[0.00058],
            alphaem_running=False,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_s, 0.0)
        res_s = quad_ker(
            u=0,
            order=(1, 1),
            mode0=100,
            mode1=21,
            method="iterate-exact",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            aem_list=[0.00058],
            alphaem_running=False,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_s, 0.0)
        res_v = quad_ker(
            u=0,
            order=(1, 1),
            mode0=10200,
            mode1=10200,
            method="iterate-exact",
            is_log=is_log,
            logx=0.123,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            aem_list=[0.00058],
            alphaem_running=False,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_v, 1.0)
        res_v = quad_ker(
            u=0,
            order=(1, 1),
            mode0=10200,
            mode1=10204,
            method="iterate-exact",
            is_log=is_log,
            logx=0.123,
            areas=np.zeros(3),
            as1=1,
            as0=2,
            aem_list=[0.00058],
            alphaem_running=False,
            nf=3,
            L=0,
            ev_op_iterations=0,
            ev_op_max_order=(0, 0),
            sv_mode=1,
            is_threshold=False,
        )
        np.testing.assert_allclose(res_v, 0.0)
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
                aem_list=[0.00058],
                alphaem_running=False,
                nf=3,
                L=0,
                ev_op_iterations=0,
                ev_op_max_order=(1, 0),
                sv_mode=sv,
                is_threshold=False,
            )
            np.testing.assert_allclose(res_sv, 1.0)
    for label in [
        (100, 100),
        (21, 21),
        (22, 22),
        (101, 101),
        (10200, 10200),
        (10204, 10204),
        (10202, 0),
    ]:
        for sv in [2, 3]:
            res_sv = quad_ker(
                u=0,
                order=(1, 1),
                mode0=label[0],
                mode1=label[1],
                method="iterate-exact",
                is_log=True,
                logx=0.123,
                areas=np.zeros(3),
                as1=1,
                as0=2,
                aem_list=[0.00058],
                alphaem_running=False,
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
        aem_list=[0.00058],
        alphaem_running=False,
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
    "xgrid": [0.1, 1.0],
    "configs": {
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "ev_op_max_order": [1, 1],
        "ev_op_iterations": 1,
        "backward_inversion": "exact",
        "n_integration_cores": 1,
    },
    "debug": {"skip_singlet": False, "skip_non_singlet": False},
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

    def test_labels_qed(self):
        o = Operator(
            dict(
                order=(3, 1),
                debug_skip_non_singlet=False,
                debug_skip_singlet=False,
                n_integration_cores=1,
            ),
            {},
            3,
            1,
            2,
        )
        assert sorted(o.labels) == sorted(br.full_unified_labels)
        o = Operator(
            dict(
                order=(2, 1),
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
            Couplings.from_dict(tcard),
            InterpolatorDispatcher(
                XGrid(
                    operators_card["xgrid"],
                    log=operators_card["configs"]["interpolation_is_log"],
                ),
                operators_card["configs"]["interpolation_polynomial_degree"],
            ),
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
            InterpolatorDispatcher(
                XGrid(
                    operators_card["xgrid"],
                    log=operators_card["configs"]["interpolation_is_log"],
                ),
                operators_card["configs"]["interpolation_polynomial_degree"],
            ),
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

    def test_aem_list(self):
        tcard = copy.deepcopy(theory_card)
        ocard = copy.deepcopy(operators_card)
        ocard["configs"]["n_integration_cores"] = 2
        ocard["configs"]["ev_op_iterations"] = 10
        for qcd in range(1, 3 + 1):
            for qed in range(1, 2 + 1):
                for q0 in [np.sqrt(2.0), 2.0, 4.5]:
                    for q2to in ocard["Q2grid"]:
                        for aem_running in [True, False]:
                            tcard["order"] = (qcd, qed)
                            tcard["alphaem_running"] = aem_running
                            tcard["Q0"] = q0
                            g = OperatorGrid.from_dict(
                                tcard,
                                ocard,
                                ThresholdsAtlas.from_dict(tcard),
                                Couplings.from_dict(tcard),
                                InterpolatorDispatcher(
                                    XGrid(
                                        operators_card["xgrid"],
                                        log=operators_card["configs"][
                                            "interpolation_is_log"
                                        ],
                                    ),
                                    operators_card["configs"][
                                        "interpolation_polynomial_degree"
                                    ],
                                ),
                            )
                            o = Operator(g.config, g.managers, 3, q0**2, q2to)
                            couplings = Couplings.from_dict(tcard)
                            aem_list = o.aem_list_as
                            (a0, a1) = o.a_s
                            ev_op_iterations = ocard["configs"]["ev_op_iterations"]
                            as_steps = utils.geomspace(a0, a1, 1 + ev_op_iterations)
                            as_l = as_steps[0]
                            for step, as_h in enumerate(as_steps[1:]):
                                as_half = (as_h + as_l) / 2.0
                                aem = couplings.compute_aem_as(
                                    tcard["alphaem"] / 4 / np.pi,
                                    tcard["alphas"] / 4 / np.pi,
                                    as_half,
                                    3,
                                )
                                np.testing.assert_allclose(
                                    aem, aem_list[step], rtol=1e-4
                                )
                                as_l = as_h

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

    def test_compute_no_skip_sv(self, monkeypatch):
        tcard = copy.deepcopy(theory_card)
        tcard["fact_to_ren_scale_ratio"] = 2.0
        tcard["ModSV"] = "expanded"
        ocard = copy.deepcopy(operators_card)
        g = OperatorGrid.from_dict(
            tcard,
            ocard,
            ThresholdsAtlas.from_dict(tcard),
            Couplings.from_dict(tcard),
            InterpolatorDispatcher(
                XGrid(
                    operators_card["xgrid"],
                    log=operators_card["configs"]["interpolation_is_log"],
                ),
                operators_card["configs"]["interpolation_polynomial_degree"],
            ),
        )
        # setup objs
        o = Operator(g.config, g.managers, 3, 2.0, 2.0)
        # fake quad
        v = 0.1234
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, v=v, **kwargs: (v, 0.56)
        )
        o.compute()
        # ns are all diagonal, so they start from an identity matrix
        for k in br.non_singlet_labels:
            assert k in o.op_members
            np.testing.assert_allclose(
                o.op_members[k].value, [[v, v], [v, 1]], err_msg=k
            )

    def test_compute(self, monkeypatch):
        tcard = copy.deepcopy(theory_card)
        ocard = copy.deepcopy(operators_card)
        g = OperatorGrid.from_dict(
            tcard,
            ocard,
            ThresholdsAtlas.from_dict(tcard),
            Couplings.from_dict(tcard),
            InterpolatorDispatcher(
                XGrid(
                    operators_card["xgrid"],
                    log=operators_card["configs"]["interpolation_is_log"],
                ),
                operators_card["configs"]["interpolation_polynomial_degree"],
            ),
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
        o.order = (3, 1)
        o.compute()
        assert not np.allclose(
            o.op_members[(br.non_singlet_pids_map["ns+u"], 0)].value,
            o.op_members[(br.non_singlet_pids_map["ns-u"], 0)].value,
        )

        # unity operators
        for n in range(1, 3 + 1):
            o1 = Operator(g.config, g.managers, 3, 2.0, 2.0)
            o1.config["order"] = (n, 0)
            o1.compute()
            for k in br.non_singlet_labels:
                assert k in o1.op_members
                np.testing.assert_allclose(o1.op_members[k].value, np.eye(2), err_msg=k)

        for n in range(1, 3 + 1):
            for qed in range(1, 2 + 1):
                g.config["order"] = (n, qed)
                o1 = Operator(g.config, g.managers, 3, 2.0, 2.0)
                # o1.config["order"] = (n, qed)
                o1.compute()
                for k in br.non_singlet_unified_labels:
                    assert k in o1.op_members
                    np.testing.assert_allclose(
                        o1.op_members[k].value, np.eye(2), err_msg=k
                    )


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
    logxs = np.log(int_disp.xgrid.raw)
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
                    [0.00058],
                    False,
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
