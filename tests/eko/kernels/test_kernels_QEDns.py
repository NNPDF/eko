import numpy as np
import pytest

from eko import basis_rotation as br
from eko.couplings import Couplings
from eko.kernels import EvoMethods
from eko.kernels import non_singlet_qed as ns
from eko.quantities.couplings import CouplingEvolutionMethod, CouplingsInfo
from eko.quantities.heavy_quarks import QuarkMassScheme
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


def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    ev_op_iterations = 2
    running_alpha = [True, False]
    for qcd in range(1, 4 + 1):
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
                            [1.0, 1.0, 1.0],
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
                            [1.0, 1.5, 2.0],
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
        for qcd in range(1, 4 + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                n = np.random.rand()
                gamma_ns = ad.gamma_ns_qed(order, mode, n, nf, (0, 0, 0, 0, 0, 0, 0))
                for method in methods:
                    for running in running_alpha:
                        np.testing.assert_allclose(
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                [1.0, 1.0, 1.0],
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
                                [1.0, 1.5, 2.0],
                                [1.0, 1.0],
                                running,
                                nf,
                                ev_op_iterations,
                                1.0,
                                1.0,
                            ),
                            1.0,
                        )


alpharef = (0.118, 0.00781)
masses = [m**2 for m in (2.0, 4.5, 175.0)]
muref = 91.0
couplings = CouplingsInfo.from_dict(
    dict(
        alphas=alpharef[0],
        alphaem=alpharef[1],
        ref=(muref, 5),
    )
)
evmod = CouplingEvolutionMethod.EXACT


def test_ode():
    ev_op_iterations = 1
    aem_list = [0.00781] * ev_op_iterations
    nf = 5
    delta_mu2 = 1e-6
    mu2_0 = 5.0**2
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        max_qcd = 4
        for qcd in range(1, max_qcd + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                sc = Couplings(
                    couplings,
                    order,
                    evmod,
                    masses,
                    hqm_scheme=QuarkMassScheme.POLE,
                    thresholds_ratios=[1.0, 1.0, 1.0],
                )
                a0 = sc.a_s(mu2_0)
                for mu2_to in [10**2, 15**2]:
                    dlog_mu2 = np.log(mu2_to + 0.5 * delta_mu2) - np.log(
                        mu2_to - 0.5 * delta_mu2
                    )
                    a1 = sc.a_s(mu2_to)
                    gamma_ns = (
                        np.random.rand(max_qcd + 1, 2 + 1)
                        + np.random.rand(max_qcd + 1, 2 + 1) * 1j
                    )
                    gamma_ns[0, 0] = 0.0
                    gamma_ns[2, 1] = 0.0
                    gamma_ns[3, 1] = 0.0
                    gamma_ns[4, 1] = 0.0
                    gamma_ns[1, 2] = 0.0
                    gamma_ns[2, 2] = 0.0
                    gamma_ns[3, 2] = 0.0
                    gamma_ns[4, 2] = 0.0
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
                            np.geomspace(a0, a1, ev_op_iterations + 1),
                            aem_list,
                            False,
                            nf,
                            ev_op_iterations,
                            mu2_0,
                            mu2_to,
                        )
                        lhs = (
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                np.geomspace(
                                    a0,
                                    sc.a_s(mu2_to + 0.5 * delta_mu2),
                                    ev_op_iterations + 1,
                                ),
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_0,
                                mu2_to + 0.5 * delta_mu2,
                            )
                            - ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                np.geomspace(
                                    a0,
                                    sc.a_s(mu2_to - 0.5 * delta_mu2),
                                    ev_op_iterations + 1,
                                ),
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_0,
                                mu2_to - 0.5 * delta_mu2,
                            )
                        ) / dlog_mu2
                        np.testing.assert_allclose(lhs, rhs, atol=np.abs(2e-4))


def test_ode_true_gamma():
    ev_op_iterations = 1
    aem_list = [0.00781] * ev_op_iterations
    nf = 5
    delta_mu2 = 1e-6
    mu2_0 = 5.0**2
    for mode in br.non_singlet_pids_map.values():
        if mode in [10201, 10101, 10200]:
            continue
        for qcd in range(1, 4 + 1):
            for qed in range(1, 2 + 1):
                order = (qcd, qed)
                sc = Couplings(
                    couplings,
                    order,
                    evmod,
                    masses,
                    hqm_scheme=QuarkMassScheme.POLE,
                    thresholds_ratios=[1.0, 1.0, 1.0],
                )
                a0 = sc.a_s(mu2_0)
                for mu2_to in [10**2, 15**2]:
                    dlog_mu2 = np.log(mu2_to + 0.5 * delta_mu2) - np.log(
                        mu2_to - 0.5 * delta_mu2
                    )
                    a1 = sc.a_s(mu2_to)
                    n = 3 + np.random.rand()
                    gamma_ns = ad.gamma_ns_qed(
                        order, mode, n, nf, (0, 0, 0, 0, 0, 0, 0)
                    )
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
                            np.geomspace(a0, a1, ev_op_iterations + 1),
                            aem_list,
                            False,
                            nf,
                            ev_op_iterations,
                            mu2_0,
                            mu2_to,
                        )
                        lhs = (
                            ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                np.geomspace(
                                    a0,
                                    sc.a_s(mu2_to + 0.5 * delta_mu2),
                                    ev_op_iterations + 1,
                                ),
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_0,
                                mu2_to + 0.5 * delta_mu2,
                            )
                            - ns.dispatcher(
                                order,
                                method,
                                gamma_ns,
                                np.geomspace(
                                    a0,
                                    sc.a_s(mu2_to - 0.5 * delta_mu2),
                                    ev_op_iterations + 1,
                                ),
                                aem_list,
                                False,
                                nf,
                                ev_op_iterations,
                                mu2_0,
                                mu2_to - 0.5 * delta_mu2,
                            )
                        ) / dlog_mu2
                        np.testing.assert_allclose(lhs, rhs, atol=np.abs(2e-4))


def test_error():
    for running in [True, False]:
        with pytest.raises(ValueError):
            ns.dispatcher(
                (5, 2),
                "iterate-exact",
                np.random.rand(4, 3),
                [0.1, 0.2],
                [0.01],
                running,
                3,
                10,
                1.0,
                1.0,
            )
