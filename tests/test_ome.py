# -*- coding: utf-8 -*-
# Test eko.matching_conditions.OperatorMatrixElement
import numpy as np

from eko import interpolation, mellin
from eko.anomalous_dimensions.harmonics import (
    harmonic_S1,
    harmonic_S2,
    harmonic_S3,
    harmonic_S4,
    harmonic_S5,
)
from eko.basis_rotation import singlet_labels
from eko.evolution_operator.grid import OperatorGrid
from eko.interpolation import InterpolatorDispatcher
from eko.matching_conditions.operator_matrix_element import (
    A_non_singlet,
    A_singlet,
    OperatorMatrixElement,
    build_ome,
    quad_ker,
    run_op_integration,
    get_s3x,
    get_s4x,
    get_smx,
)
from eko.strong_coupling import StrongCoupling
from eko.thresholds import ThresholdsAtlas
from eko.matching_conditions.n3lo import s_functions as sf


def test_HarmonicsCache():
    N = np.random.rand() + 1.0j * np.random.rand()
    Sm1 = sf.harmonic_Sm1(N)
    Sm2 = sf.harmonic_Sm2(N)
    S1 = harmonic_S1(N)
    S2 = harmonic_S2(N)
    S3 = harmonic_S3(N)
    S4 = harmonic_S4(N)
    sx = np.array([S1, S2, S3, S4, harmonic_S5(N)])
    smx_test = np.array(
        [
            Sm1,
            Sm2,
            sf.harmonic_Sm3(N),
            sf.harmonic_Sm4(N),
            sf.harmonic_Sm5(N),
        ]
    )
    np.testing.assert_allclose(get_smx(N), smx_test)
    s3x_test = np.array(
        [
            sf.harmonic_S21(N, S1, S2),
            sf.harmonic_S2m1(N, S2, Sm1, Sm2),
            sf.harmonic_Sm21(N, Sm1),
            sf.harmonic_Sm2m1(N, S1, S2, Sm2),
        ]
    )
    np.testing.assert_allclose(get_s3x(N, sx, smx_test), s3x_test)
    Sm31 = sf.harmonic_Sm31(N, Sm1, Sm2)
    s4x_test = np.array(
        [
            sf.harmonic_S31(N, S2, S4),
            sf.harmonic_S211(N, S1, S2, S3),
            sf.harmonic_Sm22(N, Sm31),
            sf.harmonic_Sm211(N, Sm1),
            Sm31,
        ]
    )
    np.testing.assert_allclose(get_s4x(N, sx, smx_test), s4x_test)


def test_build_ome_as():
    # test that if as = 0 ome is and identity
    N = 2
    L = 0.0
    a_s = 0.0
    sx = np.random.rand(19) + 1j * np.random.rand(19)
    nf = 3
    for o in [0, 1, 2, 3]:
        if o == 3:
            N = complex(2.123)
        aNS = A_non_singlet(o, N, sx, nf, L)
        aS = A_singlet(o, N, sx, nf, L)

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
    sx = np.array([1, 1, 1], np.complex_)
    nf = 4
    aNSi = A_non_singlet(1, N, sx, nf, L)
    aSi = A_singlet(1, N, sx, nf, L)
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
    zeros = np.zeros((2, 2))
    monkeypatch.setattr(
        "eko.matching_conditions.operator_matrix_element.A_non_singlet",
        lambda *args: np.array([zeros, zeros, zeros]),
    )
    zeros = np.zeros((3, 3))
    monkeypatch.setattr(
        "eko.matching_conditions.operator_matrix_element.A_singlet",
        lambda *args: np.array([zeros, zeros, zeros]),
    )
    for is_log in [True, False]:
        res_ns = quad_ker(
            u=0,
            order=3,
            mode="NS_qq",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=3,
            L=0.0,
        )
        np.testing.assert_allclose(res_ns, 1.0)
        res_s = quad_ker(
            u=0,
            order=3,
            mode="S_qq",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=3,
            L=0.0,
        )
        np.testing.assert_allclose(res_s, 1.0)
        res_s = quad_ker(
            u=0,
            order=3,
            mode="S_qg",
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=3,
            L=0.0,
        )
        np.testing.assert_allclose(res_s, 0.0)

    # test expanded intrisic inverse kernels
    labels = ["NS_qq", *singlet_labels]
    for label in labels:
        res_ns = quad_ker(
            u=0,
            order=3,
            mode=label,
            is_log=True,
            logx=0.0,
            areas=np.zeros(3),
            backward_method="expanded",
            a_s=0.0,
            nf=3,
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
            order=3,
            mode=label,
            is_log=True,
            logx=0.0,
            areas=np.zeros(3),
            backward_method="exact",
            a_s=0.0,
            nf=3,
            L=0.0,
        )
        if label[-1] == label[-2]:
            np.testing.assert_allclose(res_ns, 1.0)
        else:
            np.testing.assert_allclose(res_ns, 0.0)

    monkeypatch.setattr(interpolation, "log_evaluate_Nx", lambda *args: 0)
    res_ns = quad_ker(
        u=0,
        order=3,
        mode="NS_qq",
        is_log=True,
        logx=0.0,
        areas=np.array([0.01, 0.1, 1.0]),
        backward_method=None,
        a_s=0.0,
        nf=3,
        L=0.0,
    )
    np.testing.assert_allclose(res_ns, 0.0)


def test_run_integration():
    # setup objs
    theory_card = {
        "alphas": 0.35,
        "PTO": 2,
        "ModEv": "TRN",
        "fact_to_ren_scale_ratio": 1.0,
        "Qref": np.sqrt(2),
        "nfref": None,
        "Q0": np.sqrt(2),
        "NfFF": 3,
        "IC": 1,
        "IB": 0,
        "mc": 1.0,
        "mb": 4.75,
        "mt": 173.0,
        "kcThr": 0.0,
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
        "backward_inversion": "",
    }
    g = OperatorGrid.from_dict(
        theory_card,
        operators_card,
        ThresholdsAtlas.from_dict(theory_card),
        StrongCoupling.from_dict(theory_card),
        InterpolatorDispatcher.from_dict(operators_card),
    )
    o = OperatorMatrixElement(g.config, g.managers, is_backward=False)
    log_grid = np.log(o.int_disp.xgrid_raw)
    res = run_op_integration(
        log_grid=(len(log_grid) - 1, log_grid[-1]),
        int_disp=o.int_disp,
        labels=["NS_qq"],
        is_log=True,
        grid_size=len(log_grid),
        a_s=0.333,
        order=theory_card["PTO"],
        L=0,
        nf=4,
        backward_method="",
    )

    # here the last point is a zero, by default
    np.testing.assert_allclose(res[0]["NS_qq"], (0.0, 0.0))

    # test that copy ome does not change anything
    o.copy_ome()
    np.testing.assert_allclose(0.0, o.ome_members["S_qq"].value)


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

    def test_compute(self):
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
        o.compute(self.theory_card["mb"] ** 2, nf=4, L=0)

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
