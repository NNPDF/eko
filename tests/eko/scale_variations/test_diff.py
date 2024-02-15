"""Test ``ModSV=exponentiated`` kernel vs ``ModSV=expanded``.

We test that the quantity :math:`ker_A / ker_B` for truncated solution,
is always higher order difference.

For simplicity we do FFNS nf=4.
"""

import numpy as np

from eko import basis_rotation as br
from eko.beta import beta_qcd_as2, beta_qcd_as3
from eko.couplings import CouplingEvolutionMethod, Couplings, CouplingsInfo
from eko.kernels import non_singlet, singlet
from eko.quantities.heavy_quarks import QuarkMassScheme
from eko.scale_variations import expanded, exponentiated
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns, gamma_singlet

NF = 4
Q02 = 1.65**2
Q12 = 100**2
EV_METHOD = "truncated"


def compute_a_s(q2, order):
    sc = Couplings(
        couplings=CouplingsInfo(
            alphas=0.1181,
            alphaem=0.007496,
            scale=91.00,
            max_num_flavs=4,
            num_flavs_ref=4,
        ),
        order=order,
        method=CouplingEvolutionMethod.EXPANDED,
        masses=np.array([0.0, np.inf, np.inf]),
        hqm_scheme=QuarkMassScheme.POLE,
        thresholds_ratios=np.array([1.0, 1.0, 1.0]),
    )
    # the multiplication for xif2 here it's done explicitly outside
    return sc.a_s(scale_to=q2)


def scheme_diff_ns(g, a0, a1, L, order):
    """:math:`ker_A / ker_B` for truncated non-singlet expansion."""

    b0 = beta_qcd_as2(NF)
    b1 = beta_qcd_as3(NF)
    if order == (1, 0):
        # series of (1.0 + b0 * L * a0) ** (g[0] / b0), L->0
        diff = 1 + a0 * g[0] * L + 1 / 2 * a0**2 * g[0] * (-b0 + g[0]) * L**2
    elif order == (2, 0):
        # this term is formally 1 + as^2
        diff = (
            1
            - (a1**2 * g[0] * L * (-b1 * g[0] + b0 * (g[1] + b0 * g[0] * L))) / b0**2
            + (
                (a0 * a1 * g[0] * L)
                * (-2 * b1 * g[0] + b0 * (3 * b0 * g[0] * L + g[1] * 2))
            )
            / b0**2
            + (a0**2 * L / (2 * b0**5))
            * (
                +3 * b0**6 * g[0] * L
                + b0**5 * (-3 * g[0] * g[0] * L + g[1] * 2)
                + (2 * b0**3) * (+b1 * g[0] * (g[0]))
                + b0**4 * (-2 * g[0] * g[1])
            )
        )
    return diff


def scheme_diff_s(g, a0, a1, L, order):
    """:math:`ker_A / ker_B` for truncated singlet expansion."""

    b0 = beta_qcd_as2(NF)
    b1 = beta_qcd_as3(NF)
    if order == (1, 0):
        # series of exp(log(1.0 + b0 * L * a0) * g[0] / b0)[0], L->0
        diff = np.eye(2) + a0 * g[0] * L + 1 / 2 * a0**2 * g[0] @ (-b0 + g[0]) * L**2
    elif order == (2, 0):
        # this term is formally 1 + as^2
        diff = (
            np.eye(2)
            - (a1**2 * g[0] * L @ (-b1 * g[0] + b0 * (g[1] + b0 * g[0] * L))) / b0**2
            + (
                (a0 * a1 * g[0] * L)
                @ (-2 * b1 * g[0] + b0 * (3 * b0 * g[0] * L + g[1] * 2))
            )
            / b0**2
            + (a0**2 * L / (2 * b0**5))
            * (
                +3 * b0**6 * g[0] * L
                + b0**5 * (-3 * g[0] @ g[0] * L + g[1] * 2)
                + (2 * b0**3) * (+b1 * g[0] @ (g[0]))
                + b0**4 * (-2 * g[0] @ g[1])
            )
        )
    return diff


def test_scale_variation_a_vs_b():
    r"""Test ``ModSV=exponentiated`` kernel vs ``ModSV=expanded``."""

    # let's use smaller scale variation to
    # keep the expansions under control
    for xif2 in [0.9, 1.1]:
        L = np.log(xif2)
        for order in [(1, 0), (2, 0)]:
            # compute values of alphas
            a0 = compute_a_s(Q02, order)
            a1 = compute_a_s(Q12, order)
            a0_b = a0
            a1_b = compute_a_s(Q12 * xif2, order)
            a0_a = compute_a_s(Q02 * xif2, order)
            a1_a = a1_b  # for FFNS these 2 will coincide
            for n in [2.0, 3.0, 10.0]:
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
                    order, EV_METHOD, gns, a1_b, a0_b, NF, ev_op_iterations=1
                )
                ker_b = ker_b * expanded.non_singlet_variation(gns, a1_b, order, NF, L)

                # build scheme A solution
                gns_a = exponentiated.gamma_variation(gns.copy(), order, NF, L)
                ker_a = non_singlet.dispatcher(
                    order, EV_METHOD, gns_a, a1_a, a0_a, NF, ev_op_iterations=1
                )

                ns_diff = scheme_diff_ns(gns, a0, a1, L, order)
                np.testing.assert_allclose(
                    ker_a / ker_b,
                    ns_diff,
                    err_msg=f"{L=},{order=},{n=},non-singlet",
                    rtol=2e-5 if order == (1, 0) else 3e-3,
                )

                # Singlet kernels
                gs = gamma_singlet(
                    order, n, NF, n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0)
                )

                # build scheme B solution
                ker_b = singlet.dispatcher(
                    order,
                    EV_METHOD,
                    gs,
                    a1_b,
                    a0_b,
                    NF,
                    ev_op_iterations=1,
                    ev_op_max_order=1,
                )
                ker_b = expanded.singlet_variation(gs, a1_b, order, NF, L, 2) @ ker_b

                # build scheme A solution
                gs_a = exponentiated.gamma_variation(gs.copy(), order, NF, L)
                ker_a = singlet.dispatcher(
                    order,
                    EV_METHOD,
                    gs_a,
                    a1_a,
                    a0_a,
                    NF,
                    ev_op_iterations=1,
                    ev_op_max_order=1,
                )

                s_diff = scheme_diff_s(gs, a0, a1, L, order)
                np.testing.assert_allclose(
                    np.diag(ker_a @ np.linalg.inv(ker_b)),
                    np.diag(s_diff),
                    err_msg=f"{L=},{order=},{n=},singlet",
                    rtol=2e-4 if order == (1, 0) else 9e-3,
                )
