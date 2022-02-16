# -*- coding: utf-8 -*-

import numpy as np

from eko.anomalous_dimensions import gamma_ns, gamma_singlet
from eko.kernels import non_singlet, singlet
from eko.scale_variations import a, b


def test_ns_sv_dispacher():
    """Test to identity"""
    order = 2
    gamma_ns = np.random.rand(order + 1)
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        b.non_singlet_variation(gamma_ns, a_s, order, nf, L), 1.0
    )


def test_singlet_sv_dispacher():
    """Test to identity"""
    order = 2
    gamma_singlet = np.random.rand(order + 1, 2, 2)
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        b.singlet_variation(gamma_singlet, a_s, order, nf, L), np.eye(2)
    )


def test_scale_variation_a_vs_b():
    """Test sv_scheme A kernel vs sv_scheme B"""
    nf = 5
    n = 10
    a1 = 0.118 / (4 * np.pi)
    a0 = 0.30 / (4 * np.pi)
    method = "truncated"
    # one precision test and one with
    # a more physical setting
    for L in [np.log(2), np.log(1 + 1e-2)]:
        ns_rtol = L / 2
        sing_rtol = L
        for order in [0, 1, 2]:
            # Non singlet kernels
            gns = gamma_ns(order, 4, n, nf)
            ker = non_singlet.dispatcher(
                order, method, gns, a1, a0, nf, ev_op_iterations=1
            )
            gns_a = a.gamma_variation(gns, order, nf, L)
            ker_a = non_singlet.dispatcher(
                order, method, gns_a, a1, a0, nf, ev_op_iterations=1
            )
            ker_b = ker * b.non_singlet_variation(gns, a1, order, nf, L)
            np.testing.assert_allclose(ker_a, ker_b, rtol=ns_rtol)

            # Singlet kernels
            gs = gamma_singlet(order, n, nf)
            ker = singlet.dispatcher(
                order, method, gs, a1, a0, nf, ev_op_iterations=1, ev_op_max_order=1
            )
            gs_a = a.gamma_variation(gs, order, nf, L)
            ker_a = singlet.dispatcher(
                order, method, gs_a, a1, a0, nf, ev_op_iterations=1, ev_op_max_order=1
            )
            ker_b = ker @ b.singlet_variation(gs, a1, order, nf, L)
            np.testing.assert_allclose(ker_a, ker_b, rtol=sing_rtol)
