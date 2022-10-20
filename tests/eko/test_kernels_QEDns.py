# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pytest

from eko import anomalous_dimensions as ad
from eko import basis_rotation as br
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
    for qcd in range(1, 3 + 1):
        for qed in range(0, 2 + 1):
            order = (qcd, qed)
            gamma_ns = (
                np.random.rand(qcd + 1, qed + 1) + np.random.rand(qcd + 1, qed + 1) * 1j
            )
            for method in methods:
                np.testing.assert_allclose(
                    ns.dispatcher(
                        order, method, gamma_ns, 1.0, 1.0, [1.0, 1.0], False, nf, ev_op_iterations
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
                        [1.0, 1.0],
                        False,
                        nf,
                        ev_op_iterations,
                    ),
                    1.0,
                )


def test_zero_true_gamma():
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        for qcd in range(1, 3 + 1):
            for qed in range(0, 2 + 1):
                order = (qcd, qed)
                n = np.random.rand()
                gamma_ns = ad.gamma_ns_qed(order, mode, n, nf)
                for method in methods:
                    np.testing.assert_allclose(
                        ns.dispatcher(
                            order, method, gamma_ns, 1.0, 1.0, [1.0, 1.0], False, nf, ev_op_iterations
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
                            [1.0, 1.0],
                            False,
                            nf,
                            ev_op_iterations,
                        ),
                        1.0,
                    )


def test_ode():
    nf = 3
    ev_op_iterations = 10
    aem_list = [0.01] * ev_op_iterations
    delta_a = -1e-6
    a0 = 0.3
    betaQCD = np.zeros((4, 3), np.complex_)
    for i in range(1, 3 + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for qcd in range(1, 3 + 1):
        for qed in range(1, 2 + 1):
            order = (qcd, qed)
            for a1 in [0.1, 0.2]:
                gamma_ns = (
                    np.random.rand(3 + 1, 2 + 1) + np.random.rand(3 + 1, 2 + 1) * 1j
                )
                gamma_ns[0, 0] = 0.0
                gamma_ns[2, 1] = 0.0
                gamma_ns[3, 1] = 0.0
                gamma_ns[1, 2] = 0.0
                gamma_ns[2, 2] = 0.0
                gamma_ns[3, 2] = 0.0
                gammatot = 0.0
                betatot = 0.0
                for i in range(0, order[0] + 1):
                    for j in range(0, order[1] + 1):
                        gammatot += gamma_ns[i, j] * a1**i * aem_list[0]**j
                        betatot += a1**1 * betaQCD[i, j] * a1**i * aem_list[0]**j

                r = gammatot / betatot
                for method in methods:
                    rhs = r * ns.dispatcher(
                        order, method, gamma_ns, a1, a0, aem_list, False, nf, ev_op_iterations
                    )
                    lhs = (
                        ns.dispatcher(
                            order,
                            method,
                            gamma_ns,
                            a1 + 0.5 * delta_a,
                            a0,
                            aem_list,
                            False,
                            nf,
                            ev_op_iterations,
                        )
                        - ns.dispatcher(
                            order,
                            method,
                            gamma_ns,
                            a1 - 0.5 * delta_a,
                            a0,
                            aem_list,
                            False,
                            nf,
                            ev_op_iterations,
                        )
                    ) / delta_a
                    np.testing.assert_allclose(lhs, rhs, atol=np.abs(delta_a))


def test_ode_true_gamma():
    ev_op_iterations = 10
    aem_list = [0.01] * ev_op_iterations
    nf = 3
    delta_a = -1e-6
    a0 = 0.3
    betaQCD = np.zeros((4, 3), np.complex_)
    for i in range(1, 3 + 1):
        betaQCD[i, 0] = beta.beta_qcd((i + 1, 0), nf)
    betaQCD[1, 1] = beta.beta_qcd((2, 1), nf)
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        for qcd in range(1, 3 + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                for a1 in [0.1, 0.2]:
                    n = 3 + np.random.rand()
                    gamma_ns = ad.gamma_ns_qed(order, mode, n, nf)
                    gammatot = 0.0
                    betatot = 0.0
                    for i in range(0, order[0] + 1):
                        for j in range(0, order[1] + 1):
                            gammatot += gamma_ns[i, j] * a1**i * aem_list[0]**j
                            betatot += a1**1 * betaQCD[i, j] * a1**i * aem_list[0]**j

                    r = gammatot / betatot
                    for method in methods:
                        rhs = r * ns.dispatcher(
                            order, method, gamma_ns, a1, a0, aem_list, False, nf, ev_op_iterations
                        )
                        lhs = (
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                a1 + 0.5 * delta_a,
                                a0,
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                            )
                            - ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                a1 - 0.5 * delta_a,
                                a0,
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                            )
                        ) / delta_a
                        np.testing.assert_allclose(lhs, rhs, atol=np.abs(delta_a))


def test_error():
    with pytest.raises(NotImplementedError):
        ns.dispatcher(
            (4, 2), "iterate-exact", np.random.rand(4, 3), 0.2, 0.1, [0.01], False, 3, 10
        )
