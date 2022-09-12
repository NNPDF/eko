# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest

from eko import anomalous_dimensions as ad
from eko import beta
from eko.kernels import QEDnon_singlet as ns

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


def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    for qcd in range(1, 3):
        for qed in range(0, 2):
            order = (qcd, qed)
            if order == (0, 0):
                continue
            gamma_ns = (
                np.random.rand(qcd + 1, qed + 1) + np.random.rand(qcd + 1, qed + 1) * 1j
            )
            for method in methods:
                np.testing.assert_allclose(
                    ns.dispatcher(
                        order, method, gamma_ns, 1.0, 1.0, 1.0, nf, ev_op_iterations
                    ),
                    1.0,
                )
                np.testing.assert_allclose(
                    ns.dispatcher(
                        order,
                        method,
                        np.zeros((qcd + 1, qed + 1), dtype=complex),
                        2.0,
                        1.0,
                        1.0,
                        nf,
                        ev_op_iterations,
                    ),
                    1.0,
                )


def test_error():
    with pytest.raises(NotImplementedError):
        ns.dispatcher(
            (4, 2), "iterate-exact", np.random.rand(4, 3), 0.2, 0.1, 0.01, 3, 10
        )
