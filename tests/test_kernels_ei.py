# -*- coding: utf-8 -*-
import numpy as np

from eko.kernels import evolution_integrals as ei
from eko import beta


def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    for fnc in [ei.j00, ei.j01_exact, ei.j01_expanded, ei.j11_exact, ei.j11_expanded]:
        np.testing.assert_allclose(fnc(1, 1, nf), 0)


def test_der_lo():
    """LO derivative"""
    nf = 3
    a0 = 5
    a1 = 3
    delta_a = -1e-6
    rhs = 1.0 / (beta.beta(0, nf) * a1)
    lhs = (
        ei.j00(a1 + 0.5 * delta_a, a0, nf) - ei.j00(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs)


def test_der_nlo_exp():
    """expanded NLO derivative"""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 01
    rhs = 1.0 / (beta.beta(0, nf) * a1 + beta.beta(1, nf) * a1 ** 2)
    lhs = (
        ei.j01_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j01_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs((beta.b(1, nf) * a1) ** 2))
    # 11
    rhs = 1.0 / (beta.beta(0, nf) + beta.beta(1, nf) * a1)
    lhs = (
        ei.j11_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j11_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(beta.b(1, nf) * a1))


def test_der_nlo_exa():
    """exact NLO derivative"""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 01
    rhs = 1.0 / (beta.beta(0, nf) * a1 + beta.beta(1, nf) * a1 ** 2)
    lhs = (
        ei.j01_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j01_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 11
    rhs = 1.0 / (beta.beta(0, nf) + beta.beta(1, nf) * a1)
    lhs = (
        ei.j11_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j11_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
