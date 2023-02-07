import os

import numpy as np
import pytest
import scipy.integrate

import eko.runner.legacy
import ekore.anomalous_dimensions.unpolarized.space_like as ad
from eko import basis_rotation as br
from eko import interpolation, mellin
from eko.evolution_operator import Operator, quad_ker
from eko.interpolation import InterpolatorDispatcher
from eko.io.runcards import OperatorCard, ScaleVariationsMethod, TheoryCard
from eko.kernels import non_singlet as ns
from eko.kernels import singlet as s


def test_quad_ker_errors():
    for p, t in [(True, False), (False, True), (True, True)]:
        for mode0 in [br.non_singlet_pids_map["ns+"], 21]:
            with pytest.raises(NotImplementedError):
                quad_ker(
                    u=0.3,
                    order=(1, 0),
                    mode0=mode0,
                    mode1=0,
                    method="",
                    is_log=True,
                    logx=0.1,
                    areas=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    as1=1,
                    as0=2,
                    nf=3,
                    L=0,
                    ev_op_iterations=1,
                    ev_op_max_order=(1, 0),
                    sv_mode=1,
                    is_threshold=False,
                    is_polarized=p,
                    is_time_like=t,
                    n3lo_ad_variation="best",
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
            n3lo_ad_variation=(0, 0, 0, 0),
            is_polarized=False,
            is_time_like=False,
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
            n3lo_ad_variation=(0, 0, 0, 0),
            is_polarized=False,
            is_time_like=False,
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
            n3lo_ad_variation=(0, 0, 0, 0),
            is_polarized=False,
            is_time_like=False,
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
                n3lo_ad_variation=(0, 0, 0, 0),
                is_polarized=False,
                is_time_like=False,
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
        n3lo_ad_variation=(0, 0, 0, 0),
        is_polarized=False,
        is_time_like=False,
    )
    np.testing.assert_allclose(res_ns, 0.0)


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

    def test_exponentiated(self, theory_ffns, operator_card, tmp_path):
        tcard: TheoryCard = theory_ffns(3)
        tcard.xif = 2.0
        ocard: OperatorCard = operator_card
        ocard.configs.scvar_method = ScaleVariationsMethod.EXPONENTIATED
        r = eko.runner.legacy.Runner(tcard, ocard, path=tmp_path / "eko.tar")
        g = r.op_grid
        # setup objs
        o = Operator(g.config, g.managers, 3, 2.0, 10.0)
        np.testing.assert_allclose(o.sv_exponentiated_shift(40.0), 10.0)
        o.compute()
        self.check_lo(o)

    def test_compute_parallel(self, monkeypatch, theory_ffns, operator_card, tmp_path):
        tcard: TheoryCard = theory_ffns(3)
        ocard: OperatorCard = operator_card
        ocard.configs.n_integration_cores = 2
        r = eko.runner.legacy.Runner(tcard, ocard, path=tmp_path / "eko.tar")
        g = r.op_grid
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

    def test_compute_no_skip_sv(
        self, monkeypatch, theory_ffns, operator_card, tmp_path
    ):
        tcard: TheoryCard = theory_ffns(3)
        tcard.xif = 2.0
        ocard: OperatorCard = operator_card
        ocard.configs.scvar_method = ScaleVariationsMethod.EXPANDED
        r = eko.runner.legacy.Runner(tcard, ocard, path=tmp_path / "eko.tar")
        g = r.op_grid
        # setup objs
        o = Operator(g.config, g.managers, 3, 2.0, 2.0)
        # fake quad
        v = 0.1234
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, v=v, **kwargs: (v, 0.56)
        )
        o.compute()
        lx = len(ocard.rotations.xgrid.raw)
        res = np.full((lx, lx), v)
        res[-1, -1] = 1.0
        # ns are all diagonal, so they start from an identity matrix
        for k in br.non_singlet_labels:
            assert k in o.op_members
            np.testing.assert_allclose(o.op_members[k].value, res, err_msg=k)

    def test_compute(self, monkeypatch, theory_ffns, operator_card, tmp_path):
        tcard: TheoryCard = theory_ffns(3)
        ocard: OperatorCard = operator_card
        r = eko.runner.legacy.Runner(tcard, ocard, path=tmp_path / "eko.tar")
        g = r.op_grid
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
                np.testing.assert_allclose(
                    o1.op_members[k].value,
                    np.eye(len(ocard.rotations.xgrid.raw)),
                    err_msg=k,
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
                    nf,
                    L,
                    ev_op_iterations,
                    10,
                    0,
                    False,
                    "best",
                    False,
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
