# -*- coding: utf-8 -*-
import numpy as np
import pytest


class FakePDF:
    def hasFlavor(self, pid):
        return pid == 1

    def xfxQ2(self, _pid, x, _q2):
        return x


@pytest.fixture
def fake_pdf():
    return FakePDF()


class FakeOutput:
    shape = (2, 2)

    def mkO(self):
        ma, mae = np.random.rand(2, *self.shape)
        return ma, mae

    def mk_g(self, q2s, lpids, lx):
        Q2grid = {}
        for q2 in q2s:
            Q2grid[q2] = {
                "operator": np.random.rand(lpids, lx, lpids, lx),
                "error": np.random.rand(lpids, lx, lpids, lx),
                "alphas": np.random.rand(),
            }
        return Q2grid

    def fake_output(self):
        # build data
        interpolation_xgrid = np.array([0.5, 1.0])
        interpolation_polynomial_degree = 1
        interpolation_is_log = False
        pids = [0, 1]
        q2_ref = 1
        q2_out = 2
        Q2grid = self.mk_g([q2_out], len(pids), len(interpolation_xgrid))
        d = dict(
            xgrid=interpolation_xgrid,
            rotations=dict(
                targetgrid=interpolation_xgrid,
                inputgrid=interpolation_xgrid,
                inputpids=pids,
                targetpids=pids,
            ),
            Q0=np.sqrt(q2_ref),
            couplings=dict(),
            configs=dict(
                ev_op_max_order=1,
                ev_op_iterations=1,
                interpolation_polynomial_degree=interpolation_polynomial_degree,
                interpolation_is_log=interpolation_is_log,
                backward_inversion="exact",
            ),
            Q2grid=Q2grid,
        )
        return d


@pytest.fixture
def fake_factory():
    return FakeOutput()


@pytest.fixture
def fake_output():
    return FakeOutput().fake_output()
