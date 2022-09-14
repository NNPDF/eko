# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest

from eko import anomalous_dimensions as ad
from eko.kernels import QEDvalence as val

methods = [
    # "iterate-expanded",
    # "decompose-expanded",
    # "perturbative-expanded",
    # "truncated",
    # "ordered-truncated",
    "iterate-exact",
    # "decompose-exact",
    # "perturbative-exact",
]


def test_zero(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = (3, 0)
    for qcd in range(1, 3 + 1):
        for qed in range(1, 2 + 1):
            order = (qcd, qed)
            gamma_v = (
                np.random.rand(qcd + 1, qed + 1, 2, 2)
                + np.random.rand(qcd + 1, qed + 1, 2, 2) * 1j
            )
            # monkeypatch.setattr(
            #     ad,
            #     "exp_matrix",
            #     lambda gamma_S: (
            #         gamma_S,
            #         1,
            #         1,
            #         np.array([[1, 0], [0, 0]]),
            #         np.array([[0, 0], [0, 1]]),
            #     ),
            # )
            for method in methods:
                np.testing.assert_allclose(
                    val.dispatcher(
                        order,
                        method,
                        gamma_v,
                        1,
                        1,
                        1,
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(2),
                )
                np.testing.assert_allclose(
                    val.dispatcher(
                        order,
                        method,
                        np.zeros((qcd + 1, qed + 1, 2, 2), dtype=complex),
                        2,
                        1,
                        1,
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(2),
                )


def test_zero_true_gamma(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = (3, 0)
    for qcd in range(1, 3 + 1):
        for qed in range(1, 2 + 1):
            order = (qcd, qed)
            n = np.random.rand()
            gamma_v = ad.gamma_valence_qed(order, n, nf)
            # monkeypatch.setattr(
            #     ad,
            #     "exp_matrix",
            #     lambda gamma_S: (
            #         gamma_S,
            #         1,
            #         1,
            #         np.array([[1, 0], [0, 0]]),
            #         np.array([[0, 0], [0, 1]]),
            #     ),
            # )
            for method in methods:
                np.testing.assert_allclose(
                    val.dispatcher(
                        order,
                        method,
                        gamma_v,
                        1,
                        1,
                        1,
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(2),
                )
                np.testing.assert_allclose(
                    val.dispatcher(
                        order,
                        method,
                        np.zeros((qcd + 1, qed + 1, 2, 2), dtype=complex),
                        2,
                        1,
                        1,
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(2),
                )


def test_error():
    with pytest.raises(NotImplementedError):
        val.dispatcher(
            (3, 2), "AAA", np.random.rand(4, 3, 2, 2), 0.2, 0.1, 0.01, 3, 10, 10
        )