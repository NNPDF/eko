# -*- coding: utf-8 -*-
import numpy as np

from eko.kernels import non_singlet as ns


def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    gamma_ns = np.array([1, 1])
    for order in [0, 1]:
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
                ns.dispatcher(order, method, gamma_ns, 1, 1, nf, ev_op_iterations), 1.0
            )
            np.testing.assert_allclose(
                ns.dispatcher(order, method, np.zeros(2), 2, 1, nf, ev_op_iterations),
                1.0,
            )
