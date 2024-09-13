import warnings

import numpy as np
import pytest

from eko.kernels import EvoMethods
from eko.kernels import singlet as s
from ekore import anomalous_dimensions as ad


def test_zero_lo(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = (3, 0)
    gamma_s = np.random.rand(1, 2, 2) + np.random.rand(1, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_matrix_2D",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in EvoMethods:
        np.testing.assert_allclose(
            s.dispatcher(
                (1, 0), method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
            ),
            np.eye(2),
        )
        np.testing.assert_allclose(
            s.dispatcher(
                (1, 0),
                method,
                np.zeros((1, 2, 2), dtype=complex),
                2,
                1,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            ),
            np.zeros((2, 2)),
        )


def test_zero_nlo_decompose(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = (3, 0)
    gamma_s = np.random.rand(2, 2, 2) + np.random.rand(2, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_matrix_2D",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in [
        EvoMethods.DECOMPOSE_EXPANDED,
        EvoMethods.DECOMPOSE_EXACT,
    ]:
        np.testing.assert_allclose(
            s.dispatcher(
                (2, 0), method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
            ),
            np.eye(2),
        )
        np.testing.assert_allclose(
            s.dispatcher(
                (2, 0),
                method,
                np.zeros((2, 2, 2), dtype=complex),
                2,
                1,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            ),
            np.zeros((2, 2)),
        )


def test_zero_nnlo_decompose(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 3
    ev_op_max_order = (4, 0)
    gamma_s = np.random.rand(3, 2, 2) + np.random.rand(3, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_matrix_2D",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in [
        EvoMethods.DECOMPOSE_EXPANDED,
        EvoMethods.DECOMPOSE_EXACT,
    ]:
        np.testing.assert_allclose(
            s.dispatcher(
                (3, 0), method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
            ),
            np.eye(2),
        )
        np.testing.assert_allclose(
            s.dispatcher(
                (3, 0),
                method,
                np.zeros((3, 2, 2), dtype=complex),
                2,
                1,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            ),
            np.zeros((2, 2)),
        )


def test_zero_n3lo_decompose(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 3
    ev_op_max_order = 3
    gamma_s = np.random.rand(4, 2, 2) + np.random.rand(4, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_matrix_2D",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in [
        EvoMethods.DECOMPOSE_EXPANDED,
        EvoMethods.DECOMPOSE_EXACT,
    ]:
        np.testing.assert_allclose(
            s.dispatcher(
                (4, 0), method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
            ),
            np.eye(2),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            s.dispatcher(
                (4, 0),
                method,
                np.zeros((4, 2, 2), dtype=complex),
                2,
                1,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            ),
            np.zeros((2, 2)),
        )


def test_similarity():
    """All methods should be similar."""
    nf = 3
    a0 = 0.1
    delta_a = 1e-3
    a1 = a0 + delta_a
    ev_op_iterations = 10
    ev_op_max_order = (10, 0)
    gamma_s = np.random.rand(4, 2, 2) + np.random.rand(4, 2, 2) * 1j
    for order in [
        (1, 0),
        (2, 0),
        (3, 0),
    ]:
        ref = s.dispatcher(
            order,
            EvoMethods.DECOMPOSE_EXACT,
            gamma_s,
            a1,
            a0,
            nf,
            ev_op_iterations,
            ev_op_max_order,
        )
        for method in EvoMethods:
            np.testing.assert_allclose(
                s.dispatcher(
                    order,
                    method,
                    gamma_s,
                    a1,
                    a0,
                    nf,
                    ev_op_iterations,
                    ev_op_max_order,
                ),
                ref,
                atol=delta_a,
            )


def test_error():
    with pytest.raises(NotImplementedError):
        s.dispatcher(
            (4, 0), "iterate-exact", np.random.rand(3, 2, 2), 0.2, 0.1, 3, 10, 10
        )


def mk_almost_diag_matrix(n, max_ang=np.pi / 8.0):
    rs = np.random.rand(n) * max_ang
    coss, sins = np.cos(rs), np.sin(rs)
    a = np.array([[coss, sins], [sins, coss]])
    return a.swapaxes(0, 2)


def test_gamma_usage():
    a1 = 0.25
    a0 = 0.3
    nf = 3
    ev_op_iterations = 10
    ev_op_max_order = (10, 0)
    # first check that at order=n only uses the matrices up n
    gamma_s = np.full((4, 2, 2), np.nan)
    for order in range(1, 5):
        gamma_s[order - 1] = mk_almost_diag_matrix(1)
        for method in EvoMethods:
            r = s.dispatcher(
                (order, 0),
                method,
                gamma_s,
                a1,
                a0,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            )
            assert not np.isnan(r).all()
    # second check that at order=n the actual matrix n is used
    for order in range(1, 5):
        gamma_s = mk_almost_diag_matrix(4)
        gamma_s[order - 1] = np.full((2, 2), np.nan)
        for method in EvoMethods:
            if method in [EvoMethods.ITERATE_EXACT, EvoMethods.ITERATE_EXPANDED]:
                # we are actually dividing by the determinant of
                # matrix full of np.nan
                warnings.simplefilter("ignore", RuntimeWarning)
            r = s.dispatcher(
                (order, 0),
                method,
                gamma_s,
                a1,
                a0,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            )
            assert np.isnan(r).any()


def test_singlet_back():
    order = (3, 0)
    gamma_s = np.random.rand(order[0], 2, 2) + np.random.rand(order[0], 2, 2) * 1j
    nf = 4
    a1 = 3.0
    a0 = 4.0
    s10 = s.dispatcher(order, EvoMethods.ITERATE_EXACT, gamma_s, a1, a0, nf, 15, 1)
    np.testing.assert_allclose(
        np.linalg.inv(s10),
        s.dispatcher(order, EvoMethods.ITERATE_EXACT, gamma_s, a0, a1, nf, 15, 1),
    )
