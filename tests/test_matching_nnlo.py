# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np
import scipy.integrate

from eko.evolution_operator.grid import OperatorGrid
from eko.thresholds import ThresholdsAtlas
from eko.strong_coupling import StrongCoupling
from eko.interpolation import InterpolatorDispatcher
from eko import mellin
from eko import interpolation
from eko.matching_conditions.operator_matrix_element import (
    quad_ker,
    OperatorMatrixElement,
)

from eko.matching_conditions.nnlo import A_ns_2, A_singlet_2
from eko.anomalous_dimensions import harmonics


def get_sx(N):
    """Collect the S-cache"""
    sx = np.array(
        [
            harmonics.harmonic_S1(N),
            harmonics.harmonic_S2(N),
            harmonics.harmonic_S3(N),
        ]
    )
    return sx


def test_A_2():
    N = 1
    sx = get_sx(N)
    np.testing.assert_allclose(A_ns_2(N, sx), 0.0, atol=3e-7)

    # get singlet sector
    N = 2
    sx = get_sx(N)
    aS2 = A_singlet_2(N, sx)

    # gluon momentum conservation
    # Reference numbers coming from Mathematica
    np.testing.assert_allclose(aS2[0, 1] + aS2[1, 1], 0.00035576, rtol=1e-6)
    # quark momentum conservation
    np.testing.assert_allclose(A_ns_2(N, sx) + aS2[0, 0] + aS2[1, 0], 0.0, atol=3e-7)

    assert aS2.shape == (2, 2)


# Test operator matriz element integration
def test_quad_ker(monkeypatch):
    monkeypatch.setattr(
        mellin, "Talbot_path", lambda *args: 2
    )  # N=2 is a safe evaluation point
    monkeypatch.setattr(
        mellin, "Talbot_jac", lambda *args: complex(0, np.pi)
    )  # negate mellin prefactor
    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 1)
    monkeypatch.setattr(interpolation, "evaluate_Nx", lambda *args: 1)
    monkeypatch.setattr(
        "eko.matching_conditions.operator_matrix_element.A_ns_2", lambda *args: 1.0
    )
    monkeypatch.setattr(
        "eko.matching_conditions.operator_matrix_element.A_singlet_2",
        lambda *args: np.identity(2),
    )
    for is_log in [True, False]:
        res_ns = quad_ker(
            u=0,
            order=2,
            mode="NS_qq",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=0.0,
        )
        np.testing.assert_allclose(res_ns, 1.0)
        res_s = quad_ker(
            u=0,
            order=2,
            mode="S_qq",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=0.0,
        )
        np.testing.assert_allclose(res_s, 1.0)
        res_s = quad_ker(
            u=0,
            order=2,
            mode="S_qg",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=0.0,
        )
        np.testing.assert_allclose(res_s, 0.0)
    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 0)
    res_ns = quad_ker(
        u=0,
        order=2,
        mode="NS_qq",
        is_log=True,
        logx=0.0,
        areas=np.zeros(3),
        backward_method=None,
        a_s=0.0,
        nf=0.0,
    )
    np.testing.assert_allclose(res_ns, 0.0)


class TestOperatorMatrixElement:
    def test_compute(self, monkeypatch):
        # setup objs
        theory_card = {
            "alphas": 0.35,
            "PTO": 0,
            "ModEv": "TRN",
            "fact_to_ren_scale_ratio": 1.0,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "NfFF": 3,
            "IC": 0,
            "mc": 1.0,
            "mb": 4.75,
            "mt": 173.0,
            "kcThr": np.inf,
            "kbThr": np.inf,
            "ktThr": np.inf,
            "MaxNfPdf": 6,
            "MaxNfAs": 6,
        }
        operators_card = {
            "Q2grid": [1, 10],
            "interpolation_xgrid": [0.1, 1.0],
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "debug_skip_singlet": True,
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
        o = OperatorMatrixElement(g.config, g.managers, 10)
        # fake quad
        monkeypatch.setattr(
            scipy.integrate, "quad", lambda *args, **kwargs: np.random.rand(2)
        )
        o.order = 2
        o.compute()
        assert "NS_qq" in o.ome_members
        assert "S_qq" in o.ome_members
        assert "S_qg" in o.ome_members
        assert "S_gq" in o.ome_members
        assert "S_gg" in o.ome_members
