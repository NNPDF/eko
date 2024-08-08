import numpy as np
import pytest

from eko.kernels import EvoMethods
from eko.kernels import singlet_qed as s
from ekore.anomalous_dimensions.unpolarized import space_like as ad

methods = [
    # "iterate-expanded",
    # "decompose-expanded",
    # "perturbative-expanded",
    # "truncated",
    # "ordered-truncated",
    EvoMethods.ITERATE_EXACT,
    # "decompose-exact",
    # "perturbative-exact",
]


def test_zero(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = (3, 0)
    for qcd in range(1, 4 + 1):
        for qed in range(1, 2 + 1):
            order = (qcd, qed)
            gamma_s = (
                np.random.rand(qcd + 1, qed + 1, 4, 4)
                + np.random.rand(qcd + 1, qed + 1, 4, 4) * 1j
            )
            # monkeypatch.setattr(
            #     ad,
            #     "exp_matrix_2D",
            #     lambda gamma_S: (
            #         gamma_S,
            #         1,
            #         1,
            #         np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            #         np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            #         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
            #         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
            #     ),
            # )
            for method in methods:
                np.testing.assert_allclose(
                    s.dispatcher(
                        order,
                        method,
                        gamma_s,
                        [1, 1, 1],
                        np.array([[1, 1], [1, 1]]),
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(4),
                )
                np.testing.assert_allclose(
                    s.dispatcher(
                        order,
                        method,
                        np.zeros((qcd + 1, qed + 1, 4, 4), dtype=complex),
                        [1, 1.5, 2],
                        np.array([[1, 1], [1, 1]]),
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(4),
                )


def test_zero_true_gamma(monkeypatch):
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    ev_op_max_order = (3, 0)
    for qcd in range(1, 4 + 1):
        for qed in range(1, 2 + 1):
            order = (qcd, qed)
            n = np.random.rand()
            gamma_s = ad.gamma_singlet_qed(order, n, nf, (0, 0, 0, 0))
            # monkeypatch.setattr(
            #     ad,
            #     "exp_matrix_2D",
            #     lambda gamma_S: (
            #         gamma_S,
            #         1,
            #         1,
            #         np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            #         np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            #         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
            #         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
            #     ),
            # )
            for method in methods:
                np.testing.assert_allclose(
                    s.dispatcher(
                        order,
                        method,
                        gamma_s,
                        [1, 1, 1],
                        np.array([[1, 1], [1, 1]]),
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(4),
                )
                np.testing.assert_allclose(
                    s.dispatcher(
                        order,
                        method,
                        np.zeros((qcd + 1, qed + 1, 4, 4), dtype=complex),
                        [1.0, 1.5, 2.0],
                        np.array([[1.25, 1], [1.75, 1]]),
                        nf,
                        ev_op_iterations,
                        ev_op_max_order,
                    ),
                    np.eye(4),
                )


def test_error():
    with pytest.raises(NotImplementedError):
        s.dispatcher(
            (3, 2),
            "iterate-exact",
            np.random.rand(4, 3, 2, 2),
            [0.2, 0.1],
            [0.01],
            3,
            10,
            10,
        )
