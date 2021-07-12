# -*- coding: utf-8 -*-
import numpy as np
import pytest

from eko import anomalous_dimensions as ad
from eko.kernels import singlet as s

methods = [
    "iterate-expanded",
    "decompose-expanded",
    "perturbative-expanded",
    "truncated",
    "ordered-truncated",
    "iterate-exact",
    "decompose-exact",
    "perturbative-exact",
]


def test_zero_lo(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = 2
    gamma_s = np.random.rand(1, 2, 2) + np.random.rand(1, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_singlet",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in methods:
        try:
            np.testing.assert_allclose(
                s.dispatcher(
                    0, method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
                ),
                np.zeros((2, 2)),
            )
            np.testing.assert_allclose(
                s.dispatcher(
                    0,
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
        except ZeroDivisionError:
            pass


def test_zero_nlo_decompose(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = 2
    gamma_s = np.random.rand(2, 2, 2) + np.random.rand(2, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_singlet",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in [
        "decompose-expanded",
        "decompose-exact",
    ]:
        try:
            np.testing.assert_allclose(
                s.dispatcher(
                    1, method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
                ),
                np.zeros((2, 2)),
            )
            np.testing.assert_allclose(
                s.dispatcher(
                    1,
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
        except ZeroDivisionError:
            pass


def test_zero_nnlo_decompose(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 3
    ev_op_max_order = 3
    gamma_s = np.random.rand(3, 2, 2) + np.random.rand(3, 2, 2) * 1j
    monkeypatch.setattr(
        ad,
        "exp_singlet",
        lambda gamma_S: (
            gamma_S,
            1,
            1,
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
        ),
    )
    for method in [
        # "decompose-expanded",
        "decompose-exact",
    ]:
        try:
            np.testing.assert_allclose(
                s.dispatcher(
                    2, method, gamma_s, 1, 1, nf, ev_op_iterations, ev_op_max_order
                ),
                np.zeros((2, 2)),
            )
            np.testing.assert_allclose(
                s.dispatcher(
                    2,
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
        except ZeroDivisionError:
            pass


def test_similarity():
    """all methods should be similar"""
    nf = 3
    a0 = 0.1
    delta_a = 1e-3
    a1 = a0 + delta_a
    ev_op_iterations = 10
    ev_op_max_order = 10
    gamma_s = np.random.rand(3, 2, 2) + np.random.rand(3, 2, 2) * 1j
    for order in [
        0,
        1,
        2,
    ]:
        ref = s.dispatcher(
            order,
            "decompose-exact",
            gamma_s,
            a1,
            a0,
            nf,
            ev_op_iterations,
            ev_op_max_order,
        )
        for method in methods:
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
        s.dispatcher(2, "AAA", np.random.rand(3, 2, 2), 0.2, 0.1, 3, 10, 10)


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
    ev_op_max_order = 10
    # first check that at order=n only uses the matrices up n
    gamma_s = np.full((3, 2, 2), np.nan)
    for order in range(3):
        gamma_s[order] = mk_almost_diag_matrix(1)
        for method in methods:
            r = s.dispatcher(
                order,
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
    for order in range(3):
        gamma_s = mk_almost_diag_matrix(3)
        gamma_s[order] = np.full((2, 2), np.nan)
        for method in methods:
            r = s.dispatcher(
                order,
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
    gamma_s = np.random.rand(3, 2, 2) + np.random.rand(3, 2, 2) * 1j
    nf = 4
    a1 = 3.0
    a0 = 4.0
    s10 = s.dispatcher(2, "iterate-exact", gamma_s, a1, a0, nf, 1, 1)
    np.testing.assert_allclose(
        np.linalg.inv(s10), s.dispatcher(2, "iterate-exact", gamma_s, a0, a1, nf, 1, 1)
    )
