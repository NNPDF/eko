# Test ekore.matching_conditions.OperatorMatrixElement
import pathlib

import numpy as np
import pytest

from eko import basis_rotation as br
from eko import interpolation, mellin
from eko import scale_variations as sv
from eko.evolution_operator.operator_matrix_element import (
    OperatorMatrixElement,
    build_ome,
    quad_ker,
)
from eko.io.runcards import OperatorCard, TheoryCard
from eko.io.types import InversionMethod
from eko.runner import legacy
from ekore.harmonics import compute_cache
from ekore.operator_matrix_elements.unpolarized.space_like import (
    A_non_singlet,
    A_singlet,
)

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
            for method in [None, InversionMethod.EXPANDED, InversionMethod.EXACT]:
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
        for method in [None, InversionMethod.EXPANDED, InversionMethod.EXACT]:
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


def test_quad_ker_errors():
    for p, t in [(True, False), (True, True)]:
        for mode0, mode1 in [
            (21, br.matching_hplus_pid),
            (200, br.matching_hminus_pid),
        ]:
            with pytest.raises(NotImplementedError):
                quad_ker(
                    u=0.3,
                    order=(1, 0),
                    mode0=mode0,
                    mode1=mode1,
                    is_log=True,
                    logx=0.123,
                    areas=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    backward_method=None,
                    a_s=0.0,
                    nf=3,
                    L=0.0,
                    sv_mode=sv.Modes.expanded,
                    Lsv=0.0,
                    is_msbar=False,
                    is_polarized=p,
                    is_time_like=t,
                )


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
        "ekore.operator_matrix_elements.unpolarized.space_like.A_non_singlet",
        lambda *args: np.array([zeros, zeros, zeros]),
    )
    zeros = np.zeros((3, 3))
    monkeypatch.setattr(
        "ekore.operator_matrix_elements.unpolarized.space_like.A_singlet",
        lambda *args: np.array([zeros, zeros, zeros]),
    )
    for is_log in [True, False]:
        for order, p, t in [((3, 0), False, False), ((2, 0), False, True)]:
            res_ns = quad_ker(
                u=0,
                order=order,
                mode0=200,
                mode1=200,
                is_log=is_log,
                logx=0.123,
                areas=np.zeros(3),
                backward_method=None,
                a_s=0.0,
                nf=3,
                L=0.0,
                sv_mode=sv.Modes.expanded,
                Lsv=0.0,
                is_msbar=False,
                is_polarized=p,
                is_time_like=t,
            )
            np.testing.assert_allclose(res_ns, 1.0)
            res_s = quad_ker(
                u=0,
                order=order,
                mode0=100,
                mode1=100,
                is_log=is_log,
                logx=0.123,
                areas=np.zeros(3),
                backward_method=None,
                a_s=0.0,
                nf=3,
                L=0.0,
                sv_mode=sv.Modes.expanded,
                Lsv=0.0,
                is_msbar=False,
                is_polarized=p,
                is_time_like=t,
            )
            np.testing.assert_allclose(res_s, 1.0)
            res_s = quad_ker(
                u=0,
                order=order,
                mode0=100,
                mode1=21,
                is_log=is_log,
                logx=0.0,
                areas=np.zeros(3),
                backward_method=None,
                a_s=0.0,
                nf=3,
                L=0.0,
                sv_mode=sv.Modes.expanded,
                Lsv=0.0,
                is_msbar=False,
                is_polarized=p,
                is_time_like=t,
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
            backward_method=InversionMethod.EXPANDED,
            a_s=0.0,
            nf=3,
            L=0.0,
            sv_mode=sv.Modes.expanded,
            Lsv=0.0,
            is_msbar=False,
            is_polarized=False,
            is_time_like=False,
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
            backward_method=InversionMethod.EXACT,
            a_s=0.0,
            nf=3,
            L=0.0,
            sv_mode=sv.Modes.expanded,
            Lsv=0.0,
            is_msbar=False,
            is_polarized=False,
            is_time_like=False,
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
        sv_mode=sv.Modes.expanded,
        Lsv=0.0,
        is_msbar=False,
        is_polarized=False,
        is_time_like=False,
    )
    np.testing.assert_allclose(res_ns, 0.0)


# def test_run_integration():
#     # setup objs
#     theory_card = {
#         "alphas": 0.35,
#         "PTO": 2,
#         "ModEv": "TRN",
#         "XIF": 1.0,
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
#         "mugrid": [(1, 3), (10, 5)],
#         "interpolation_xgrid": [0.1, 1.0],
#         "interpolation_polynomial_degree": 1,
#         "interpolation_is_log": True,
#         "debug_skip_singlet": True,
#         "debug_skip_non_singlet": False,
#         "ev_op_max_order": 1,
#         "ev_op_iterations": 1,
#         "backward_inversion": None,
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
#         backward_method=None,
#         is_msbar=False,
#     )

#     # here the last point is a zero, by default
#     np.testing.assert_allclose(res[0][(200, 200)], (0.0, 0.0))

#     # test that copy ome does not change anything
#     o.copy_ome()
#     np.testing.assert_allclose(0.0, o.op_members[(100, 100)].value)


class TestOperatorMatrixElement:
    def test_labels(self, theory_ffns, operator_card, tmp_path: pathlib.Path):
        path = tmp_path / "eko.tar"
        for skip_singlet in [True, False]:
            for skip_ns in [True, False]:
                operator_card.debug.skip_singlet = skip_singlet
                operator_card.debug.skip_non_singlet = skip_ns
                path.unlink(missing_ok=True)
                g = legacy.Runner(theory_ffns(3), operator_card, path=path).op_grid
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

    def test_compute_n3lo(self, theory_ffns, operator_card, tmp_path):
        theory_card: TheoryCard = theory_ffns(5)
        theory_card.heavy.matching_ratios.c = 1.0
        theory_card.heavy.matching_ratios.b = 1.0
        theory_card.order = (4, 0)
        operator_card.debug.skip_singlet = True
        g = legacy.Runner(theory_card, operator_card, path=tmp_path / "eko.tar").op_grid
        o = OperatorMatrixElement(
            g.config,
            g.managers,
            is_backward=True,
            q2=theory_card.heavy.masses.b.value**2,
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
            # TODO: working before d81d78e without k=-1 (before #172)
            np.testing.assert_allclose(mat, np.triu(mat, -1))

    def test_compute_lo(self, theory_ffns, operator_card, tmp_path):
        theory_card = theory_ffns(5)
        theory_card.heavy.matching_ratios.c = 1.0
        theory_card.heavy.matching_ratios.b = 1.0
        theory_card.order = (1, 0)
        operator_card.debug.skip_singlet = False
        operator_card.debug.skip_non_singlet = False
        g = legacy.Runner(theory_card, operator_card, path=tmp_path / "eko.tar").op_grid
        o = OperatorMatrixElement(
            g.config,
            g.managers,
            is_backward=False,
            q2=theory_card.heavy.masses.b.value**2,
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

    def test_compute_nlo(self, theory_ffns, operator_card: OperatorCard, tmp_path):
        theory_card: TheoryCard = theory_ffns(5)
        theory_card.heavy.matching_ratios.c = 1.0
        theory_card.heavy.matching_ratios.b = 1.0
        theory_card.order = (2, 0)
        operator_card.mugrid = [(20.0, 5)]
        operator_card.xgrid = interpolation.XGrid([0.001, 0.01, 0.1, 1.0])
        operator_card.configs.interpolation_polynomial_degree = 1
        operator_card.configs.interpolation_is_log = True
        operator_card.configs.ev_op_max_order = (2, 0)
        operator_card.configs.ev_op_iterations = 1
        operator_card.configs.inversion_method = InversionMethod.EXACT
        operator_card.configs.n_integration_cores = 1
        operator_card.debug.skip_singlet = True
        operator_card.debug.skip_non_singlet = False
        g = legacy.Runner(theory_card, operator_card, path=tmp_path / "eko.tar").op_grid
        o = OperatorMatrixElement(
            g.config,
            g.managers,
            is_backward=False,
            q2=theory_card.heavy.masses.b.value**2,
            nf=4,
            L=0,
            is_msbar=False,
        )
        o.compute()

        dim = len(operator_card.xgrid)
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
