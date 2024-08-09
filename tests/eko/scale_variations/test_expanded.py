import numpy as np

from eko import basis_rotation as br
from eko.couplings import CouplingEvolutionMethod, Couplings, CouplingsInfo
from eko.kernels import EvoMethods, non_singlet, singlet
from eko.quantities.heavy_quarks import QuarkMassScheme
from eko.scale_variations import Modes, expanded
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns, gamma_singlet

NF = 4
Q02 = 1.65**2
Q12 = 100**2
EV_METHOD = EvoMethods.TRUNCATED


def compute_a_s(q2, order):
    sc = Couplings(
        couplings=CouplingsInfo(
            alphas=0.1181,
            alphaem=0.007496,
            ref=(91.00, 4),
        ),
        order=order,
        method=CouplingEvolutionMethod.EXPANDED,
        masses=np.array([0.0, np.inf, np.inf]),
        hqm_scheme=QuarkMassScheme.POLE,
        thresholds_ratios=np.array([1.0, 1.0, 1.0]),
    )
    # the multiplication for xif2 here it's done explicitly outside
    return sc.a_s(scale_to=q2)


def test_modes():
    assert Modes.expanded.name == "expanded"
    assert Modes.exponentiated.name == "exponentiated"
    assert Modes.unvaried.name == "unvaried"
    assert Modes.expanded.value == 3
    assert Modes.exponentiated.value == 2
    assert Modes.unvaried.value == 1


def test_ns_sv_dispacher():
    """Test to identity"""
    order = (4, 0)
    gamma_ns = np.random.rand(order[0])
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        expanded.non_singlet_variation(gamma_ns, a_s, order, nf, L), 1.0
    )


def test_ns_sv_dispacher_qed():
    """Test to identity"""
    order = (4, 2)
    gamma_ns = np.random.rand(order[0], order[1])
    L = 0
    nf = 5
    a_s = 0.35
    a_em = 0.01
    for alphaem_running in [True, False]:
        np.testing.assert_allclose(
            expanded.non_singlet_variation_qed(
                gamma_ns, a_s, a_em, alphaem_running, order, nf, L
            ),
            1.0,
        )


def test_singlet_sv_dispacher():
    """Test to identity"""
    order = (4, 0)
    gamma_singlet = np.random.rand(order[0], 2, 2)
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        expanded.singlet_variation(gamma_singlet, a_s, order, nf, L, 2), np.eye(2)
    )


def test_singlet_sv_dispacher_qed():
    """Test to identity"""
    order = (4, 2)
    gamma_singlet = np.random.rand(order[0], order[1], 4, 4)
    L = 0
    nf = 5
    a_s = 0.35
    a_em = 0.01
    for alphaem_running in [True, False]:
        np.testing.assert_allclose(
            expanded.singlet_variation_qed(
                gamma_singlet, a_s, a_em, alphaem_running, order, nf, L
            ),
            np.eye(4),
        )


def test_valence_sv_dispacher_qed():
    """Test to identity"""
    order = (4, 2)
    gamma_valence = np.random.rand(order[0], order[1], 2, 2)
    L = 0
    nf = 5
    a_s = 0.35
    a_em = 0.01
    for alphaem_running in [True, False]:
        np.testing.assert_allclose(
            expanded.valence_variation_qed(
                gamma_valence, a_s, a_em, alphaem_running, order, nf, L
            ),
            np.eye(2),
        )


def test_expanded_is_linear():
    r"""Test is linear."""
    for order in [(1, 0), (2, 0), (3, 0), (4, 0)]:
        for n in [2.0, 3.0, 10.0]:
            rel_err_ns = []
            rel_err_s = []
            for L in [0.3, 0.5, 0.7]:
                xif2 = np.exp(L)
                # compute values of alphas
                a0 = compute_a_s(Q02, order)
                a1 = compute_a_s(Q12, order)
                a1_b = compute_a_s(Q12 * xif2, order)
                # Non singlet kernels
                gns = gamma_ns(
                    order,
                    br.non_singlet_pids_map["ns+"],
                    n,
                    NF,
                    n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0),
                )

                # build scheme B solution
                ker_b = non_singlet.dispatcher(
                    order, EV_METHOD, gns, a1, a0, NF, ev_op_iterations=1
                )
                sv_b = non_singlet.dispatcher(
                    order, EV_METHOD, gns, a1_b, a0, NF, ev_op_iterations=1
                )
                sv_b = sv_b * expanded.non_singlet_variation(gns, a1_b, order, NF, L)

                rel_err_ns.append(sv_b / ker_b)

                # Singlet kernels
                gs = gamma_singlet(
                    order, n, NF, n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0)
                )

                # build scheme B solution
                ker_b = singlet.dispatcher(
                    order,
                    EV_METHOD,
                    gs,
                    a1,
                    a0,
                    NF,
                    ev_op_iterations=1,
                    ev_op_max_order=1,
                )
                sv_b = singlet.dispatcher(
                    order,
                    EV_METHOD,
                    gs,
                    a1_b,
                    a0,
                    NF,
                    ev_op_iterations=1,
                    ev_op_max_order=1,
                )
                sv_b = expanded.singlet_variation(gs, a1_b, order, NF, L, 2) @ sv_b
                rel_err_s.append(sv_b @ np.linalg.inv(ker_b))

            # there must be something
            for err in rel_err_ns:
                assert np.abs(err) != 1.0
            for err in np.array(rel_err_s).flatten():
                assert np.abs(err) != 1.0
            # error has to increase
            np.testing.assert_allclose(
                rel_err_ns,
                sorted(rel_err_ns, reverse=True),
                err_msg=f"{order=},{n=},non-singlet",
            )
            np.testing.assert_allclose(
                rel_err_s,
                sorted(rel_err_s, key=np.max, reverse=True),
                err_msg=f"{order=},{n=},singlet",
            )
