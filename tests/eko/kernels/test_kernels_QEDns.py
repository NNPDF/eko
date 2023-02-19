import warnings

import numpy as np
import pytest

from eko import basis_rotation as br
from eko import beta
from eko.kernels import non_singlet_qed as ns
from ekore.anomalous_dimensions.unpolarized import space_like as ad

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
    running_alpha = [True, False]
    for qcd in range(1, 3 + 1):
        for qed in range(1, 2 + 1):
            order = (qcd, qed)
            gamma_ns = (
                np.random.rand(qcd + 1, qed + 1) + np.random.rand(qcd + 1, qed + 1) * 1j
            )
            for method in methods:
                for running in running_alpha:
                    np.testing.assert_allclose(
                        ns.dispatcher(
                            order,
                            method,
                            gamma_ns,
                            1.0,
                            1.0,
                            [1.0, 1.0],
                            running,
                            nf,
                            ev_op_iterations,
                            1.0,
                            1.0,
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
                            running,
                            nf,
                            ev_op_iterations,
                            1.0,
                            1.0,
                        ),
                        1.0,
                    )


def test_zero_true_gamma():
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    running_alpha = [True, False]
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        for qcd in range(1, 3 + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                n = np.random.rand()
                gamma_ns = ad.gamma_ns_qed(order, mode, n, nf)
                for method in methods:
                    for running in running_alpha:
                        np.testing.assert_allclose(
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                1.0,
                                1.0,
                                [1.0, 1.0],
                                running,
                                nf,
                                ev_op_iterations,
                                1.0,
                                1.0,
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
                                running,
                                nf,
                                ev_op_iterations,
                                1.0,
                                1.0,
                            ),
                            1.0,
                        )


from math import nan

from eko.couplings import Couplings, couplings_mod_ev
from eko.io.types import (
    CouplingEvolutionMethod,
    CouplingsRef,
    EvolutionMethod,
    MatchingScales,
    QuarkMassSchemes,
)

alpharef = (0.118, 0.00781)
masses = [m**2 for m in (2.0, 4.5, 175.0)]
muref = 91.0
couplings = CouplingsRef.from_dict(
    dict(
        alphas=[alpharef[0], muref],
        alphaem=[alpharef[1], nan],
        num_flavs_ref=5,
        max_num_flavs=6,
    )
)
evmod = CouplingEvolutionMethod.EXACT


def test_ode():
    ev_op_iterations = 10
    aem_list = [0.00781] * ev_op_iterations
    nf = 5
    delta_mu2 = 1e-6
    mu2_0 = 5.0**2
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        for qcd in range(1, 3 + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                sc = Couplings(
                    couplings,
                    order,
                    evmod,
                    masses,
                    hqm_scheme=QuarkMassSchemes.POLE,
                    thresholds_ratios=None,
                )
                a0 = sc.a_s(mu2_0)
                for mu2_to in [10**2, 15**2]:
                    dlog_mu2 = np.log(mu2_to + 0.5 * delta_mu2) - np.log(
                        mu2_to - 0.5 * delta_mu2
                    )
                    a1 = sc.a_s(mu2_to)
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
                    for i in range(0, order[0] + 1):
                        for j in range(0, order[1] + 1):
                            gammatot += gamma_ns[i, j] * a1**i * aem_list[0] ** j
                    r = -gammatot
                    for method in methods:
                        rhs = r * ns.dispatcher(
                            order,
                            method,
                            gamma_ns,
                            a1,
                            a0,
                            aem_list,
                            False,
                            nf,
                            ev_op_iterations,
                            mu2_to,
                            mu2_0,
                        )
                        lhs = (
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                sc.a_s(mu2_to + 0.5 * delta_mu2),
                                a0,
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_to + 0.5 * delta_mu2,
                                mu2_0,
                            )
                            - ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                sc.a_s(mu2_to - 0.5 * delta_mu2),
                                a0,
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_to - 0.5 * delta_mu2,
                                mu2_0,
                            )
                        ) / dlog_mu2
                        np.testing.assert_allclose(lhs, rhs, atol=np.abs(1e-3))


def test_ode_true_gamma():
    ev_op_iterations = 10
    aem_list = [0.00781] * ev_op_iterations
    nf = 5
    delta_mu2 = 1e-6
    mu2_0 = 5.0**2
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        for qcd in range(1, 3 + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                sc = Couplings(
                    couplings,
                    order,
                    evmod,
                    masses,
                    hqm_scheme=QuarkMassSchemes.POLE,
                    thresholds_ratios=None,
                )
                a0 = sc.a_s(mu2_0)
                for mu2_to in [10**2, 15**2]:
                    dlog_mu2 = np.log(mu2_to + 0.5 * delta_mu2) - np.log(
                        mu2_to - 0.5 * delta_mu2
                    )
                    a1 = sc.a_s(mu2_to)
                    n = 3 + np.random.rand()
                    gamma_ns = ad.gamma_ns_qed(order, mode, n, nf)
                    gammatot = 0.0
                    for i in range(0, order[0] + 1):
                        for j in range(0, order[1] + 1):
                            gammatot += gamma_ns[i, j] * a1**i * aem_list[0] ** j
                    r = -gammatot
                    for method in methods:
                        rhs = r * ns.dispatcher(
                            order,
                            method,
                            gamma_ns,
                            a1,
                            a0,
                            aem_list,
                            False,
                            nf,
                            ev_op_iterations,
                            mu2_to,
                            mu2_0,
                        )
                        lhs = (
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                sc.a_s(mu2_to + 0.5 * delta_mu2),
                                a0,
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_to + 0.5 * delta_mu2,
                                mu2_0,
                            )
                            - ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                sc.a_s(mu2_to - 0.5 * delta_mu2),
                                a0,
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_to - 0.5 * delta_mu2,
                                mu2_0,
                            )
                        ) / dlog_mu2
                        np.testing.assert_allclose(lhs, rhs, atol=np.abs(1e-3))


def test_error():
    for running in [True, False]:
        with pytest.raises(NotImplementedError):
            ns.dispatcher(
                (4, 2),
                "iterate-exact",
                np.random.rand(4, 3),
                0.2,
                0.1,
                [0.01],
                running,
                3,
                10,
                1.0,
                1.0,
            )
