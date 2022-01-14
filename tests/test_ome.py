# -*- coding: utf-8 -*-
# Test eko.matching_conditions.OperatorMatrixElement
import copy

import numpy as np

from eko import interpolation, mellin
from eko.basis_rotation import singlet_labels
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher
from eko.matching_conditions.operator_matrix_element import (
    A_non_singlet,
    A_singlet,
    OperatorMatrixElement,
    build_ome,
    quad_ker,
)
from eko.strong_coupling import StrongCoupling
from eko.thresholds import ThresholdsAtlas


def test_build_ome_as():
    # test that if as = 0 ome is and identity
    N = 2
    L = 0.0
    a_s = 0.0
    sx = np.zeros(3, np.complex_)
    is_msbar = False
    for o in [0, 1, 2]:
        aNS = A_non_singlet(o, N, sx, L)
        aS = A_singlet(o, N, sx, L, is_msbar)

        for a in [aNS, aS]:
            for method in ["", "expanded", "exact"]:
                dim = len(a[0])
                if o != 0:
                    assert len(a) == o

                ome = build_ome(a, o, a_s, method)
                assert ome.shape == (dim, dim)
                assert ome.all() == np.eye(dim).all()


def test_build_ome_nlo():
    # test that the matching is not an identity when L=0 and intrinsic
    N = 2
    L = 0.0
    a_s = 20
    is_msbar = False

    sx = np.array([1, 1, 1], np.complex_)

    aNSi = A_non_singlet(1, N, sx, L)
    aSi = A_singlet(1, N, sx, L, is_msbar)
    for a in [aNSi, aSi]:
        for method in ["", "expanded", "exact"]:
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
            is_msbar=False,
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
            is_msbar=False,
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
            is_msbar=False,
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
            is_msbar=False,
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
            is_msbar=False,
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
        is_msbar=False,
    )
    np.testing.assert_allclose(res_ns, 0.0)


class TestOperatorMatrixElement:
    # setup objs
    theory_card = {
        "alphas": 0.35,
        "PTO": 0,
        "ModEv": "TRN",
        "fact_to_ren_scale_ratio": 1.0,
        "Qref": np.sqrt(2),
        "nfref": None,
        "Q0": np.sqrt(2),
        "nf0": 3,
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
        "HQ": "POLE",
    }

    def test_labels(self):
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
                    self.theory_card,
                    operators_card,
                    ThresholdsAtlas.from_dict(self.theory_card),
                    StrongCoupling.from_dict(self.theory_card),
                    InterpolatorDispatcher.from_dict(operators_card),
                )
                o = OperatorMatrixElement(g.config, g.managers, is_backward=False)
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

    def test_compute_lo(self):
        operators_card = {
            "Q2grid": [20],
            "interpolation_xgrid": [0.001, 0.01, 0.1, 1.0],
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "debug_skip_singlet": False,
            "debug_skip_non_singlet": False,
            "ev_op_max_order": 1,
            "ev_op_iterations": 1,
            "backward_inversion": "exact",
        }
        g = OperatorGrid.from_dict(
            self.theory_card,
            operators_card,
            ThresholdsAtlas.from_dict(self.theory_card),
            StrongCoupling.from_dict(self.theory_card),
            InterpolatorDispatcher.from_dict(operators_card),
        )
        o = OperatorMatrixElement(g.config, g.managers, is_backward=False)
        o.compute(self.theory_card["mb"] ** 2, L=0, is_msbar=False)

        dim = o.ome_members["NS_qq"].value.shape
        for idx in ["S", "NS"]:
            np.testing.assert_allclose(
                o.ome_members[f"{idx}_qq"].value, np.eye(dim[0]), atol=1e-8
            )
            np.testing.assert_allclose(
                o.ome_members[f"{idx}_HH"].value, np.eye(dim[0]), atol=1e-8
            )
            np.testing.assert_allclose(o.ome_members[f"{idx}_qH"].value, np.zeros(dim))
            np.testing.assert_allclose(o.ome_members[f"{idx}_Hq"].value, np.zeros(dim))
        np.testing.assert_allclose(
            o.ome_members["S_gg"].value, np.eye(dim[0]), atol=1e-8
        )
        np.testing.assert_allclose(
            o.ome_members["S_qg"].value, o.ome_members["S_gq"].value
        )
        np.testing.assert_allclose(
            o.ome_members["S_Hg"].value, o.ome_members["S_gH"].value
        )

    def test_compute_nlo(self):
        operators_card = {
            "Q2grid": [20],
            "interpolation_xgrid": [0.001, 0.01, 0.1, 1.0],
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "debug_skip_singlet": False,
            "debug_skip_non_singlet": False,
            "ev_op_max_order": 1,
            "ev_op_iterations": 1,
            "backward_inversion": "exact",
        }
        t = copy.deepcopy(self.theory_card)
        t["PTO"] = 1
        g = OperatorGrid.from_dict(
            t,
            operators_card,
            ThresholdsAtlas.from_dict(self.theory_card),
            StrongCoupling.from_dict(self.theory_card),
            InterpolatorDispatcher.from_dict(operators_card),
        )
        o = OperatorMatrixElement(g.config, g.managers, is_backward=False)
        o.compute(self.theory_card["mb"] ** 2, L=0, is_msbar=False)

        dim = len(operators_card["interpolation_xgrid"])
        shape = (dim, dim)
        for idx in ["S", "NS"]:
            assert o.ome_members[f"{idx}_qq"].value.shape == shape
            assert o.ome_members[f"{idx}_HH"].value.shape == shape
            assert o.ome_members[f"{idx}_qH"].value.shape == shape
            assert o.ome_members[f"{idx}_Hq"].value.shape == shape
            np.testing.assert_allclose(
                o.ome_members[f"{idx}_qH"].value, np.zeros(shape)
            )
            np.testing.assert_allclose(
                o.ome_members[f"{idx}_Hq"].value, np.zeros(shape)
            )
        assert o.ome_members["S_gg"].value.shape == shape
        np.testing.assert_allclose(
            o.ome_members["S_qg"].value, o.ome_members["S_gq"].value
        )
        assert o.ome_members["S_Hg"].value.shape == shape
        assert o.ome_members["S_gH"].value.shape == shape
