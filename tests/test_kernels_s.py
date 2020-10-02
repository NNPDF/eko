# -*- coding: utf-8 -*-
import numpy as np

from eko.kernels import singlet as s
from eko import anomalous_dimensions as ad


def test_zero_lo(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = 2
    gamma_s = np.random.rand(1, 2, 2)
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
        "iterate-expanded",
        "decompose-expanded",
        "perturbative-expanded",
        "truncated",
        "ordered-truncated",
        "iterate-exact",
        "decompose-exact",
        "perturbative-exact",
    ]:
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
                np.zeros((1, 2, 2)),
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
    ev_op_max_order = 2
    gamma_s = np.random.rand(2, 2, 2)
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
                np.zeros((2, 2, 2)),
                2,
                1,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            ),
            np.zeros((2, 2)),
        )
