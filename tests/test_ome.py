# -*- coding: utf-8 -*-
# Test eko.matching_conditions.OperatorMatrixElement
import numpy as np

from eko.evolution_operator.grid import OperatorGrid
from eko.thresholds import ThresholdsAtlas
from eko.strong_coupling import StrongCoupling
from eko.interpolation import InterpolatorDispatcher
from eko import mellin
from eko import interpolation
from eko.matching_conditions.operator_matrix_element import (
    quad_ker,
    OperatorMatrixElement,
    build_ome,
    A_non_singlet,
    A_singlet,
)
from eko.member import singlet_labels


def test_build_ome_as():
    # test that if as = 0 ome is and identity
    N = 2
    L = 0.0
    a_s = 0.0
    sx = np.zeros(3, np.complex_)
    for o in [0, 1, 2]:
        aNS = A_non_singlet(o, N, sx, L)
        aS = A_singlet(o, N, sx, L)

        for a in [aNS, aS]:
            for method in ["", "expanded", "exact"]:
                dim = len(a[0])
                if o != 0:
                    assert len(a) == o

                ome = build_ome(a, o, a_s, method)
                assert ome.shape == (dim, dim)
                assert ome.all() == np.eye(dim).all()


def test_build_ome_nlo():
    # test that the matching is an identity when L=0 and not intrinsic
    N = 2
    L = 0.0
    a_s = 20
    sx = np.array([1,1,1], np.complex_)
    # aNS = A_non_singlet(1, N, sx, L)
    # aS = A_singlet(1, N, sx, L)

    # for a in [aNS, aS]:
    #     for method in ["", "expanded", "exact"]:
    #         dim = len(a[0])
    #         assert len(a) == 1
    #         assert a[0].all() == np.zeros((dim, dim)).all()

    #         ome = build_ome(a, 1, a_s, method)
    #         assert ome.shape == (dim, dim)
    #         assert ome.all() == np.eye(dim).all()

    # test that the matching is not an identity when L=0 and intrinsic
    aNSi = A_non_singlet(1, N, sx, L)
    aSi = A_singlet(1, N, sx, L)
    for a in [aNSi, aSi]:
        for method in ["", "expanded","exact"]:
            dim = len(a[0])
            # hh
            assert a[0, -1, -1] != 0.0
            # qh
            assert a[0, -2, -1] == 0.0
            ome = build_ome(a, 1, a_s, method)
            assert ome.shape == (dim, dim)
            assert ome[-1, -1] != 1.0
            assert ome[-2, -1] == 0.0
            assert ome[-1, -2] == 0.0
            assert ome[-2, -2] == 1.0

    # check gh for singlet
    assert aSi[0, 0, -1] != 0.0
    assert ome[0, -1] != 0.0


# Test OME integration
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
        "eko.matching_conditions.operator_matrix_element.A_non_singlet",
        lambda *args: np.identity(2),
    )
    monkeypatch.setattr(
        "eko.matching_conditions.operator_matrix_element.A_singlet",
        lambda *args: np.identity(3),
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
            L=0.0,
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
            L=0.0,
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
            L=0.0,
        )
        np.testing.assert_allclose(res_s, 0.0)

    # test expanded intrisic inverse kernels
    labels = ["NS_qq", *singlet_labels]
    for label in labels:
        res_ns = quad_ker(
            u=0,
            order=2,
            mode=label,
            is_log=True,
            logx=0.0,
            areas=np.zeros(3),
            backward_method="expanded",
            a_s=0.0,
            L=0.0,
        )
        if label[-1] == label[-2]:
            np.testing.assert_allclose(res_ns, 1.0)
        else:
            np.testing.assert_allclose(res_ns, 0.0)

    # test exact intrisic inverse kernel
    labels.extend(
        [
            "S_Hq",
            "S_Hg",
            "S_HH",
            "S_qH",
            "S_gH",
            "NS_qH",
            "NS_HH",
            "NS_Hq",
        ]
    )
    for label in labels:
        res_ns = quad_ker(
            u=0,
            order=2,
            mode=label,
            is_log=True,
            logx=0.0,
            areas=np.zeros(3),
            backward_method="exact",
            a_s=0.0,
            L=0.0,
        )
        if label[-1] == label[-2]:
            np.testing.assert_allclose(res_ns, 1.0)
        else:
            np.testing.assert_allclose(res_ns, 0.0)

    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 0)
    res_ns = quad_ker(
        u=0,
        order=2,
        mode="NS_qq",
        is_log=True,
        logx=0.0,
        areas=np.array([0.01, 0.1, 1.0]),
        backward_method=None,
        a_s=0.0,
        L=0.0,
    )
    np.testing.assert_allclose(res_ns, 0.0)


class TestOperatorMatrixElement:
    def test_labels(self):
        # setup objs
        theory_card = {
            "alphas": 0.35,
            "PTO": 0,
            "ModEv": "TRN",
            "fact_to_ren_scale_ratio": 1.0,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "NfFF": 3,
            "IC": 1,
            "IB": 0,
            "mc": 1.0,
            "mb": 4.75,
            "mt": 173.0,
            "kcThr": np.inf,
            "kbThr": np.inf,
            "ktThr": np.inf,
            "MaxNfPdf": 6,
            "MaxNfAs": 6,
        }
        for skip_singlet in [True, False]:
            for skip_ns in [True, False]:
                operators_card = {
                    "Q2grid": [1, 10],
                    "interpolation_xgrid": [0.1, 1.0],
                    "interpolation_polynomial_degree": 1,
                    "interpolation_is_log": True,
                    "debug_skip_singlet": skip_singlet,
                    "debug_skip_non_singlet": skip_ns,
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
                labels = o.labels()
                test_labels = ["NS_qq", "NS_Hq"]
                for l in test_labels:
                    if skip_ns:
                        assert l not in labels
                    else:
                        assert l in labels
                test_labels = ["S_qq", "S_Hq", "S_gg", "S_Hg", "S_gH"]
                for l in test_labels:
                    if skip_singlet:
                        assert l not in labels
                    else:
                        assert l in labels
