# -*- coding: utf-8 -*-

import numpy as np

import eko
import eko.kernel_generation as kg

# fake BasisFunction
class FakeBF:
    def __init__(self):
        self.callable = lambda N, lnx: lnx / N


# fake decomposition
def fake_get_Eigensystem_gamma_singlet_0(*_args):
    return 1, 1, np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]])


class TestKernelDispatcher:
    def test_full(self, monkeypatch):
        # fake beta0 + anomalous dimension + mellin
        c = eko.constants.Constants()
        monkeypatch.setattr(eko.strong_coupling, "beta_0", lambda *_args: 1)
        monkeypatch.setattr(eko.anomalous_dimensions.lo, "gamma_ns_0", lambda *_args: 1)
        monkeypatch.setattr(
            eko.anomalous_dimensions.lo,
            "get_Eigensystem_gamma_singlet_0",
            fake_get_Eigensystem_gamma_singlet_0,
        )
        monkeypatch.setattr(eko.mellin, "compile_integrand", lambda ker, *_args: ker)
        # fake InterpolationDispatcher
        bfs = [FakeBF()]
        for numba_it in [True, False]:
            kd = kg.KernelDispatcher(bfs, c, numba_it=numba_it)
            nf = 3
            kd.set_up_all_integrands(nf)
            # check format
            assert nf in kd.integrands_ns
            assert len(kd.integrands_ns[nf]) == len(bfs)
            assert nf in kd.integrands_s
            assert len(kd.integrands_s[nf]) == len(bfs)
            for bf in kd.integrands_s[nf]:
                assert len(bf) == 4
            # check value
            np.testing.assert_almost_equal(kd.integrands_ns[nf][0](1, 1, 1), np.exp(-1))
            for k in range(4):
                np.testing.assert_almost_equal(
                    kd.integrands_s[nf][0][k](1, 1, 1), np.exp(-1)
                )
