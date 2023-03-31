import numpy as np

from eko import basis_rotation as br
from eko.beta import beta_qcd_as2, beta_qcd_as3
from eko.kernels import non_singlet, singlet
from eko.scale_variations import Modes, expanded, exponentiated
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
    r"""
    Test ``ModSV=exponentiated`` kernel vs ``ModSV=expanded``.
    We test that the quantity :math:`(ker_A - ker_B)/ker_{unv}`

    Note this is NOT the real difference between scheme expanded
    and exponentiated since here we don't take into account the
    shifts in path length and :math:`\alpha_s` values.
    The real difference is always higher order.
    """
    nf = 5
    n = 10
    a1 = 0.118 / (4 * np.pi)
    a0 = 0.2 / (4 * np.pi)
    method = "truncated"

    def scheme_diff(g, k, pto, is_singlet):
        """
        :math:`(ker_A - ker_B)/ker_{unv}` for truncated expansion
        Effects due to non commutativity are neglected thus,
        the accuracy of singlet quantities is slightly worse.
        """
        if pto[0] >= 2:
            diff = -g[0] * k * a0  # - 2 * a1 * k * g[0]
        # if pto[0] >= 3:
        #     b0 = beta_qcd_as2(nf)
        #     g02 = g[0] @ g[0] if is_singlet else g[0] ** 2
        #     diff += (
        #         -2 * a1**2 * g[1] * k
        #         + a0**2 * g[1] * k
        #         + a1**2 * b0 * g[0] * k**2
        #         - 0.5 * a0**2 * b0 * g[0] * k**2
        #         - a1 * a0 * g02 * k**2
        #         + 0.5 * a0**2 * g02 * k**2
        #         + a1**2 * g02 * k**2
        #     )
        # if pto[0] >= 4:
        #     b1 = beta_qcd_as3(nf)
        #     g0g1 = g[0] @ g[1] if is_singlet else g[0] * g[1]
        #     g03 = g02 @ g[0] if is_singlet else g02 * g[0]
        #     diff += (
        #         a0**3 * g[2] * k
        #         - 2 * a1**3 * g[2] * k
        #         - 1 / 2 * a0**3 * b1 * g[0] * k**2
        #         + a1**3 * b1 * g[0] * k**2
        #         - a0**3 * b0 * g[1] * k**2
        #         + 2 * a1**3 * b0 * g[1] * k**2
        #         + a0**3 * g0g1 * k**2
        #         - a0**2 * a1 * g0g1 * k**2
        #         - a0 * a1**2 * g0g1 * k**2
        #         + 2 * a1**3 * g0g1 * k**2
        #         + 1 / 3 * a0**3 * b0**2 * g[0] * k**3
        #         - 2 / 3 * a1**3 * b0**2 * g[0] * k**3
        #         - 1 / 2 * a0**3 * b0 * g02 * k**3
        #         + 1 / 2 * a0**2 * a1 * b0 * g02 * k**3
        #         + 1 / 2 * a0 * a1**2 * b0 * g02 * k**3
        #         - a1**3 * b0 * g02 * k**3
        #         + 1 / 6 * a0**3 * g03 * k**3
        #         - 1 / 2 * a0**2 * a1 * g03 * k**3
        #         + 1 / 2 * a0 * a1**2 * g03 * k**3
        #         - 1 / 3 * a1**3 * g03 * k**3
        #     )
        return diff

    for L in [np.log(0.5), np.log(2)]:
        # for order in [(2, 0), (3, 0), (4, 0)]:
        for order in [(2, 0)]:
            # Non singlet kernels
            gns = gamma_ns(order, br.non_singlet_pids_map["ns+"], n, nf)
            ker = non_singlet.dispatcher(
                order, method, gns, a1, a0, nf, ev_op_iterations=1
            )
            gns_a = exponentiated.gamma_variation(gns.copy(), order, nf, L)
            ker_a = non_singlet.dispatcher(
                order, method, gns_a, a1, a0, nf, ev_op_iterations=1
            )
            ker_b = ker * expanded.non_singlet_variation(gns, a1, order, nf, L)
            ns_diff = scheme_diff(gns, L, order, False)
            np.testing.assert_allclose(
                (ker_a - ker_b) / ker,
                ns_diff,
                atol=1e-3,
                err_msg=f"L={L},order={order},non-singlet",
            )

            # Singlet kernels
            gs = gamma_singlet(order, n, nf)
            ker = singlet.dispatcher(
                order, method, gs, a1, a0, nf, ev_op_iterations=1, ev_op_max_order=1
            )
            gs_a = exponentiated.gamma_variation(gs.copy(), order, nf, L)
            ker_a = singlet.dispatcher(
                order, method, gs_a, a1, a0, nf, ev_op_iterations=1, ev_op_max_order=1
            )
            ker_b = ker @ expanded.singlet_variation(gs, a1, order, nf, L, 2)
            s_diff = scheme_diff(gs, L, order, True)
            np.testing.assert_allclose(
                (ker_a - ker_b) @ np.linalg.inv(ker),
                s_diff,
                atol=5e-3,
                err_msg=f"L={L},order={order},singlet",
            )
