# -*- coding: utf-8 -*-

import numpy as np

import pytest

import eko
import eko.kernel_generation as kg

# fake BasisFunction
class FakeBF:
    def __init__(self):
        self.callable = lambda N, lnx: lnx / N


# fake decomposition
def fake_exp_singlet(*_args):
    return (
        np.array([[1, 1], [1, 1]]),
        0,
        0,
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]]),
    )


class TestKernelDispatcher:
    def test_from_dict(self):
        c = eko.constants.Constants()
        bfs = [FakeBF()]
        for pto in [0, 1]:
            for mod_ev, meth in [("EXA", "iterate-exact"), ("TRN", "truncated")]:
                kd = kg.KernelDispatcher.from_dict(dict(PTO=pto, ModEv=mod_ev), bfs, c)
                assert kd.config["order"] == pto
                assert kd.config["method"] == meth
            with pytest.raises(ValueError):
                _ = kg.KernelDispatcher.from_dict(dict(PTO=pto, ModEv="ERROR"), bfs, c)

    def test_full(self, monkeypatch):
        # fake beta + anomalous dimension + mellin
        c = eko.constants.Constants()
        monkeypatch.setattr(eko.strong_coupling, "beta_0", lambda *_args: 1)
        monkeypatch.setattr(eko.strong_coupling, "beta_1", lambda *_args: 1)
        monkeypatch.setattr(eko.anomalous_dimensions.lo, "gamma_ns_0", lambda *_args: 0)
        monkeypatch.setattr(
            eko.anomalous_dimensions.nlo, "gamma_nsm_1", lambda *_args: 0
        )
        monkeypatch.setattr(
            eko.anomalous_dimensions.nlo, "gamma_nsp_1", lambda *_args: 0
        )
        monkeypatch.setattr(
            eko.anomalous_dimensions.lo,
            "gamma_singlet_0",
            lambda *_args: np.zeros((2, 2)),
        )
        monkeypatch.setattr(
            eko.anomalous_dimensions.nlo,
            "gamma_singlet_1",
            lambda *_args: np.zeros((2, 2)),
        )
        monkeypatch.setattr(
            eko.anomalous_dimensions,
            "exp_singlet",
            fake_exp_singlet,
        )
        monkeypatch.setattr(eko.mellin, "compile_integrand", lambda ker, *_args: ker)
        # fake InterpolationDispatcher
        bfs = [FakeBF()]
        for numba_it in [True, False]:
            for order in [0, 1]:
                for method in ["iterate-exact", "iterate-expanded"]:
                    kd = kg.KernelDispatcher.from_dict(
                        dict(PTO=order, ModEv=method, ev_op_iterations=1),
                        bfs,
                        c,
                        numba_it=numba_it,
                    )
                    nf = 3
                    kd.set_up_all_integrands(nf)
                    # check format
                    assert nf in kd.kernels
                    assert len(kd.kernels[nf]) == len(bfs)
                    for bf in kd.kernels[nf]:
                        assert len(bf) == 4 + 1 + order
                    # check value
                    for k in kd.kernels[nf][0].values():
                        np.testing.assert_almost_equal(k(1, 1, 1, 1), 1)
                    # now it's cached
                    kd.set_up_all_integrands(nf)
                    assert nf in kd.kernels
