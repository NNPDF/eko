import numpy as np

from eko import basis_rotation as br
from eko.beta import beta_qcd_as2, beta_qcd_as3
from eko.couplings import CouplingEvolutionMethod, Couplings, CouplingsInfo
from eko.io.types import ScaleVariationsMethod as svm
from eko.kernels import non_singlet, singlet
from eko.quantities.heavy_quarks import QuarkMassScheme
from eko.scale_variations import Modes, expanded, exponentiated
from ekore.anomalous_dimensions import exp_matrix_2D
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns, gamma_singlet


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


def test_scale_variation_a_vs_b():
    r"""Test ``ModSV=exponentiated`` kernel vs ``ModSV=expanded``.

    We test that the quantity :math:`ker_A / ker_B` for truncated solution,
    is always higher order difference.

    For simplicity we do FFNS nf=4.
    """
    nf = 4
    n = 10
    q02 = 1.65**2
    q12 = 100**2
    ev_method = "truncated"

    ref = CouplingsInfo(
        alphas=0.1181,
        alphaem=0.007496,
        scale=91.00,
        max_num_flavs=6,
        num_flavs_ref=4,
    )

    def compute_a_s(q2, nf, order):
        sc = Couplings(
            couplings=ref,
            order=order,
            method=CouplingEvolutionMethod.EXPANDED,
            masses=np.array([1.51, 1e4, 1e5]) ** 2,
            hqm_scheme=QuarkMassScheme.POLE,
            # thresholds_ratios=np.array([1.0, 1.0, 1.0]) ** 2 * (
            #     xif2 if scvar_method == svm.EXPONENTIATED
            #     else 1.0
            # )
            # Let's do FFNS nf=4 for simplicity
            thresholds_ratios=np.array([1.0, 1.0, 1.0]),
        )
        # the multiplication for xif2 here it's done explicitly above
        return sc.a_s(scale_to=q2, nf_to=nf)

    def scheme_diff(g, L, order, is_singlet):
        """:math:`ker_A / ker_B` for truncated expansion."""

        b0 = beta_qcd_as2(nf)
        if order == (1, 0):
            # series of (1.0 + b0 * L * a0) ** g[0] / b0), L->0
            diff = (
                1
                + a0 * g[0] * L
                + 1 / 2 * a0**2 * g[0] * (-b0 + g[0]) * L**2
                + 1 / 6 * a0**3 * g[0] * (2 * b0**2 - 3 * b0 * g[0] + g[0] ** 2) * L**3
            )
            if is_singlet:
                # series of exp_matrix_2D(np.log(1.0 + b0 * L * a0) * g[0] / b0)[0]
                diff = (
                    np.eye(2)
                    + a0 * g[0] * L
                    + 1 / 2 * a0**2 * g[0] @ (-b0 + g[0]) * L**2
                    + 1
                    / 6
                    * a0**3
                    * g[0]
                    @ (2 * b0**2 - 3 * b0 * g[0] + g[0] @ g[0])
                    * L**3
                )
        # TODO: add higher order expressions
        return diff

    # let's use smaller scale variation to
    # keep the expansions under control
    for xif2 in [0.7**2, 1.4**2]:
        L = np.log(xif2)
        # for order in [(2, 0), (3, 0), (4, 0)]:
        for order in [(1, 0)]:
            # compute values of alphas
            a0 = compute_a_s(q02, nf, order)
            a0_b = a0
            a1_b = compute_a_s(q12 * xif2, nf, order)

            a0_a = compute_a_s(q02 * xif2, nf, order)
            # for FFNS these 2 will coincide
            a1_a = a1_b

            # Non singlet kernels
            gns = gamma_ns(
                order,
                br.non_singlet_pids_map["ns+"],
                n,
                nf,
                n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0),
            )

            # build scheme B solution
            ker_b = non_singlet.dispatcher(
                order, ev_method, gns, a1_b, a0_b, nf, ev_op_iterations=1
            )
            ker_b = ker_b * expanded.non_singlet_variation(gns, a1_b, order, nf, L)

            # build scheme A solution
            gns_a = exponentiated.gamma_variation(gns.copy(), order, nf, L)
            ker_a = non_singlet.dispatcher(
                order, ev_method, gns_a, a1_a, a0_a, nf, ev_op_iterations=1
            )

            ns_diff = scheme_diff(gns, L, order, False)
            np.testing.assert_allclose(
                ker_a / ker_b,
                ns_diff,
                err_msg=f"L={L},order={order},non-singlet",
                rtol=2e-5,
            )

            # Singlet kernels
            gs = gamma_singlet(order, n, nf, n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0))

            # build scheme B solution
            ker_b = singlet.dispatcher(
                order,
                ev_method,
                gs,
                a1_b,
                a0_b,
                nf,
                ev_op_iterations=1,
                ev_op_max_order=1,
            )
            ker_b = expanded.singlet_variation(gs, a1_b, order, nf, L, 2) @ ker_b

            # build scheme A solution
            gs_a = exponentiated.gamma_variation(gs.copy(), order, nf, L)
            ker_a = singlet.dispatcher(
                order,
                ev_method,
                gs_a,
                a1_a,
                a0_a,
                nf,
                ev_op_iterations=1,
                ev_op_max_order=1,
            )

            s_diff = scheme_diff(gs, L, order, True)
            np.testing.assert_allclose(
                np.diag(ker_a @ np.linalg.inv(ker_b)),
                np.diag(s_diff),
                err_msg=f"L={L},order={order},singlet",
                rtol=2e-3,
            )
