# -*- coding: utf-8 -*-
# Test eko.matching_conditions.OperatorMatrixElement
import copy

import numpy as np

from eko import basis_rotation as br
from eko import interpolation, mellin
from eko.couplings import Couplings
from eko.evolution_operator.grid import OperatorGrid
from eko.harmonics import compute_cache
from eko.interpolation import InterpolatorDispatcher, XGrid
from eko.matching_conditions.operator_matrix_element import (
    A_non_singlet,
    A_singlet,
    OperatorMatrixElement,
    build_ome,
    quad_ker,
)
from eko.thresholds import ThresholdsAtlas

max_weight_dict = {1: 2, 2: 3, 3: 5}


def test_build_ome_as():
    # test that if as = 0 ome is and identity
    N = complex(2.123)
    L = 0.0
    a_s = 0.0
    nf = 3
    is_msbar = False
    for o in [1, 2, 3]:
        sx_singlet = compute_cache(N, max_weight_dict[o], True)
        sx_ns = sx_singlet
        if o == 3:
            sx_ns = compute_cache(N, max_weight_dict[o], False)

        aNS = A_non_singlet((o, 0), N, sx_ns, nf, L)
        aS = A_singlet((o, 0), N, sx_singlet, nf, L, is_msbar, sx_ns)

        for a in [aNS, aS]:
            for method in ["", "expanded", "exact"]:
                dim = len(a[0])
                if o != 1:
                    assert len(a) == o

                ome = build_ome(a, (o, 0), a_s, method)
                assert ome.shape == (dim, dim)
                assert ome.all() == np.eye(dim).all()


def test_build_ome_nlo():
    # test that the matching is not an identity when L=0 and intrinsic
    N = 2
    L = 0.0
    a_s = 20
    is_msbar = False
    sx = [[1], [1], [1]]
    nf = 4
    aNSi = A_non_singlet((1, 0), N, sx, nf, L)
    aSi = A_singlet((1, 0), N, sx, nf, L, is_msbar)
    for a in [aNSi, aSi]:
        for method in ["", "expanded", "exact"]:
            dim = len(a[0])
            # hh
            assert a[0, -1, -1] != 0.0
            # qh
            assert a[0, -2, -1] == 0.0
            ome = build_ome(a, (1, 0), a_s, method)
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
            order=(3, 0),
            mode0=200,
            mode1=200,
            is_log=is_log,
            logx=0.123,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=3,
            L=0.0,
            is_msbar=False,
        )
        np.testing.assert_allclose(res_ns, 1.0)
        res_s = quad_ker(
            u=0,
            order=(3, 0),
            mode0=100,
            mode1=100,
            is_log=is_log,
            logx=0.123,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=3,
            L=0.0,
            is_msbar=False,
        )
        np.testing.assert_allclose(res_s, 1.0)
        res_s = quad_ker(
            u=0,
            order=(3, 0),
            mode0=100,
            mode1=21,
            is_log=is_log,
            logx=0.0,
            areas=np.zeros(3),
            backward_method=None,
            a_s=0.0,
            nf=3,
            L=0.0,
            is_msbar=False,
        )
        np.testing.assert_allclose(res_s, 0.0)

    # test expanded intrisic inverse kernels
    labels = [(200, 200), *br.singlet_labels]
    for label in labels:
        res_ns = quad_ker(
            u=0,
            order=(3, 0),
            mode0=label[0],
            mode1=label[1],
            is_log=True,
            logx=0.123,
            areas=np.zeros(3),
            backward_method="expanded",
            a_s=0.0,
            nf=3,
            L=0.0,
            is_msbar=False,
        )
        if label[-1] == label[-2]:
            np.testing.assert_allclose(res_ns, 1.0)
        else:
            np.testing.assert_allclose(res_ns, 0.0)

    # test exact intrinsic inverse kernel
    labels.extend(
        [
            (br.matching_hplus_pid, 100),
            (br.matching_hplus_pid, 21),
            (br.matching_hplus_pid, br.matching_hplus_pid),
            (100, br.matching_hplus_pid),
            (21, br.matching_hplus_pid),
            (200, br.matching_hminus_pid),
            (br.matching_hminus_pid, br.matching_hminus_pid),
            (br.matching_hminus_pid, 200),
        ]
    )
    for label in labels:
        res_ns = quad_ker(
            u=0,
            order=(3, 0),
            mode0=label[0],
            mode1=label[1],
            is_log=True,
            logx=0.123,
            areas=np.zeros(3),
            backward_method="exact",
            a_s=0.0,
            nf=3,
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
        order=(3, 0),
        mode0=200,
        mode1=200,
        is_log=True,
        logx=0.0,
        areas=np.array([0.01, 0.1, 1.0]),
        backward_method=None,
        a_s=0.0,
        nf=3,
        L=0.0,
        is_msbar=False,
    )
    np.testing.assert_allclose(res_ns, 0.0)


# def test_run_integration():
#     # setup objs
#     theory_card = {
#         "alphas": 0.35,
#         "PTO": 2,
#         "ModEv": "TRN",
#         "fact_to_ren_scale_ratio": 1.0,
#         "Qref": np.sqrt(2),
#         "nfref": None,
#         "Q0": np.sqrt(2),
#         "nf0": 4,
#         "NfFF": 3,
#         "IC": 1,
#         "IB": 0,
#         "mc": 1.0,
#         "mb": 4.75,
#         "mt": 173.0,
#         "kcThr": 0.0,
#         "kbThr": np.inf,
#         "ktThr": np.inf,
#         "MaxNfPdf": 6,
#         "MaxNfAs": 6,
#         "HQ": "POLE",
#         "ModSV": None,
#     }

#     operators_card = {
#         "Q2grid": [1, 10],
#         "interpolation_xgrid": [0.1, 1.0],
#         "interpolation_polynomial_degree": 1,
#         "interpolation_is_log": True,
#         "debug_skip_singlet": True,
#         "debug_skip_non_singlet": False,
#         "ev_op_max_order": 1,
#         "ev_op_iterations": 1,
#         "backward_inversion": "",
#     }
#     g = OperatorGrid.from_dict(
#         theory_card,
#         operators_card,
#         ThresholdsAtlas.from_dict(theory_card),
#         StrongCoupling.from_dict(theory_card),
#         InterpolatorDispatcher.from_dict(operators_card),
#     )
#     o = OperatorMatrixElement(g.config, g.managers, is_backward=False)
#     log_grid = np.log(o.int_disp.xgrid_raw)
#     res = run_op_integration(
#         log_grid=(len(log_grid) - 1, log_grid[-1]),
#         int_disp=o.int_disp,
#         labels=[(200, 200)],
#         is_log=True,
#         grid_size=len(log_grid),
#         a_s=0.333,
#         order=theory_card["PTO"],
#         L=0,
#         nf=4,
#         backward_method="",
#         is_msbar=False,
#     )

#     # here the last point is a zero, by default
#     np.testing.assert_allclose(res[0][(200, 200)], (0.0, 0.0))

#     # test that copy ome does not change anything
#     o.copy_ome()
#     np.testing.assert_allclose(0.0, o.op_members[(100, 100)].value)


class TestOperatorMatrixElement:
    # setup objs
    theory_card = {
        "alphas": 0.35,
        "alphaem": 0.00781,
        "order": (4, 0),
        "ModEv": "TRN",
        "fact_to_ren_scale_ratio": 1.0,
        "Qref": np.sqrt(2),
        "nfref": None,
        "Q0": np.sqrt(2),
        "nf0": 3,
        "NfFF": 3,
        "IC": 1,
        "IB": 0,
        "mc": 1.51,
        "mb": 4.75,
        "mt": 173.0,
        "kcThr": 1,
        "kbThr": 1,
        "ktThr": np.inf,
        "MaxNfPdf": 6,
        "MaxNfAs": 6,
        "HQ": "POLE",
        "ModSV": None,
    }
    operators_card = {
        "Q2grid": [20],
        "xgrid": [0.1, 1.0],
        "configs": {
            "interpolation_polynomial_degree": 1,
            "interpolation_is_log": True,
            "ev_op_max_order": 1,
            "ev_op_iterations": 1,
            "backward_inversion": "exact",
            "n_integration_cores": 1,
        },
        "debug": {"skip_singlet": True, "skip_non_singlet": False},
    }

    def test_labels(self):
        for skip_singlet in [True, False]:
            for skip_ns in [True, False]:
                operators_card = {
                    "Q2grid": [1, 10],
                    "xgrid": [0.1, 1.0],
                    "configs": {
                        "interpolation_polynomial_degree": 1,
                        "interpolation_is_log": True,
                        "ev_op_max_order": (2, 0),
                        "ev_op_iterations": 1,
                        "backward_inversion": "exact",
                        "n_integration_cores": 1,
                    },
                    "debug": {
                        "skip_singlet": skip_singlet,
                        "skip_non_singlet": skip_ns,
                    },
                }
                g = OperatorGrid.from_dict(
                    self.theory_card,
                    operators_card,
                    ThresholdsAtlas.from_dict(self.theory_card),
                    Couplings.from_dict(self.theory_card),
                    InterpolatorDispatcher(
                        XGrid(
                            operators_card["xgrid"],
                            log=operators_card["configs"]["interpolation_is_log"],
                        ),
                        operators_card["configs"]["interpolation_polynomial_degree"],
                    ),
                )
                o = OperatorMatrixElement(
                    g.config,
                    g.managers,
                    is_backward=True,
                    q2=None,
                    nf=None,
                    L=None,
                    is_msbar=False,
                )
                labels = o.labels
                test_labels = [
                    (200, 200),
                    (br.matching_hminus_pid, br.matching_hminus_pid),
                ]
                for l in test_labels:
                    if skip_ns:
                        assert l not in labels
                    else:
                        assert l in labels
                test_labels = [
                    (21, 21),
                    (21, 100),
                    (21, br.matching_hplus_pid),
                    (100, 21),
                    (100, 100),
                    (100, br.matching_hplus_pid),
                    (br.matching_hplus_pid, 21),
                    (br.matching_hplus_pid, 100),
                    (br.matching_hplus_pid, br.matching_hplus_pid),
                ]
                for l in test_labels:
                    if skip_singlet:
                        assert l not in labels
                    else:
                        assert l in labels

    def test_compute_n3lo(self):
        g = OperatorGrid.from_dict(
            self.theory_card,
            self.operators_card,
            ThresholdsAtlas.from_dict(self.theory_card),
            Couplings.from_dict(self.theory_card),
            InterpolatorDispatcher(
                XGrid(
                    self.operators_card["xgrid"],
                    log=self.operators_card["configs"]["interpolation_is_log"],
                ),
                self.operators_card["configs"]["interpolation_polynomial_degree"],
            ),
        )
        o = OperatorMatrixElement(
            g.config,
            g.managers,
            is_backward=True,
            q2=self.theory_card["mb"] ** 2,
            nf=4,
            L=0,
            is_msbar=False,
        )
        o.compute()

        dim = o.op_members[(200, 200)].value.shape
        np.testing.assert_allclose(
            o.op_members[(200, br.matching_hminus_pid)].value, np.zeros(dim)
        )
        np.testing.assert_allclose(
            o.op_members[(br.matching_hminus_pid, 200)].value, np.zeros(dim)
        )

        for label in [(200, 200), (br.matching_hminus_pid, br.matching_hminus_pid)]:
            mat = o.op_members[label].value
            np.testing.assert_allclose(mat, np.triu(mat))

    def test_compute_lo(self):
        self.theory_card.update({"order": (1, 0)})
        self.operators_card.update(
            dict(debug={"skip_singlet": False, "skip_non_singlet": False})
        )
        g = OperatorGrid.from_dict(
            self.theory_card,
            self.operators_card,
            ThresholdsAtlas.from_dict(self.theory_card),
            Couplings.from_dict(self.theory_card),
            InterpolatorDispatcher(
                XGrid(
                    self.operators_card["xgrid"],
                    log=self.operators_card["configs"]["interpolation_is_log"],
                ),
                self.operators_card["configs"]["interpolation_polynomial_degree"],
            ),
        )
        o = OperatorMatrixElement(
            g.config,
            g.managers,
            is_backward=False,
            q2=self.theory_card["mb"] ** 2,
            nf=4,
            L=0,
            is_msbar=False,
        )
        o.compute()

        dim = o.op_members[(200, 200)].value.shape
        for indices in [(100, br.matching_hplus_pid), (200, br.matching_hminus_pid)]:
            np.testing.assert_allclose(
                o.op_members[(indices[0], indices[0])].value, np.eye(dim[0]), atol=1e-8
            )
            np.testing.assert_allclose(
                o.op_members[(indices[1], indices[1])].value, np.eye(dim[0]), atol=1e-8
            )
            np.testing.assert_allclose(
                o.op_members[(indices[0], indices[1])].value, np.zeros(dim)
            )
            np.testing.assert_allclose(
                o.op_members[(indices[1], indices[0])].value, np.zeros(dim)
            )
        np.testing.assert_allclose(
            o.op_members[(21, 21)].value, np.eye(dim[0]), atol=1e-8
        )
        np.testing.assert_allclose(
            o.op_members[100, 21].value, o.op_members[(21, 100)].value
        )
        np.testing.assert_allclose(
            o.op_members[(br.matching_hplus_pid, 21)].value,
            o.op_members[(21, br.matching_hplus_pid)].value,
        )

    def test_compute_nlo(self):
        operators_card = {
            "Q2grid": [20],
            "xgrid": [0.001, 0.01, 0.1, 1.0],
            "configs": {
                "interpolation_polynomial_degree": 1,
                "interpolation_is_log": True,
                "ev_op_max_order": (2, 0),
                "ev_op_iterations": 1,
                "backward_inversion": "exact",
                "n_integration_cores": 1,
            },
            "debug": {
                "skip_singlet": False,
                "skip_non_singlet": False,
            },
        }
        t = copy.deepcopy(self.theory_card)
        t["order"] = (1, 0)
        g = OperatorGrid.from_dict(
            t,
            operators_card,
            ThresholdsAtlas.from_dict(t),
            Couplings.from_dict(t),
            InterpolatorDispatcher(
                XGrid(
                    operators_card["xgrid"],
                    log=operators_card["configs"]["interpolation_is_log"],
                ),
                operators_card["configs"]["interpolation_polynomial_degree"],
            ),
        )
        o = OperatorMatrixElement(
            g.config,
            g.managers,
            is_backward=False,
            q2=t["mb"] ** 2,
            nf=4,
            L=0,
            is_msbar=False,
        )
        o.compute()

        dim = len(operators_card["xgrid"])
        shape = (dim, dim)
        for indices in [(100, br.matching_hplus_pid), (200, br.matching_hminus_pid)]:
            assert o.op_members[(indices[0], indices[0])].value.shape == shape
            assert o.op_members[(indices[1], indices[1])].value.shape == shape
            assert o.op_members[(indices[0], indices[1])].value.shape == shape
            assert o.op_members[(indices[1], indices[0])].value.shape == shape
            np.testing.assert_allclose(
                o.op_members[(indices[0], indices[1])].value, np.zeros(shape)
            )
            np.testing.assert_allclose(
                o.op_members[(indices[1], indices[0])].value, np.zeros(shape)
            )
        assert o.op_members[(21, 21)].value.shape == shape
        np.testing.assert_allclose(
            o.op_members[(100, 21)].value, o.op_members[(21, 100)].value
        )
        assert o.op_members[(br.matching_hplus_pid, 21)].value.shape == shape
        assert o.op_members[(21, br.matching_hplus_pid)].value.shape == shape
