# -*- coding: utf-8 -*-
import numpy as np

from eko import beta
from eko.kernels import evolution_integrals as ei


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
    rhs = 1.0 / (beta.beta_qcd((0, 0), nf) * a1)
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
    rhs = 1.0 / (beta.beta_qcd((0, 0), nf) * a1 + beta.beta_qcd((1, 0), nf) * a1**2)
    lhs = (
        ei.j01_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j01_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(
        rhs, lhs, atol=np.abs((beta.b_qcd((1, 0), nf) * a1) ** 2)
    )
    # 11
    rhs = 1.0 / (beta.beta_qcd((0, 0), nf) + beta.beta_qcd((1, 0), nf) * a1)
    lhs = (
        ei.j11_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j11_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(beta.b_qcd((1, 0), nf) * a1))


def test_der_nlo_exa():
    """exact NLO derivative"""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 01
    rhs = 1.0 / (beta.beta_qcd((0, 0), nf) * a1 + beta.beta_qcd((1, 0), nf) * a1**2)
    lhs = (
        ei.j01_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j01_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 11
    rhs = 1.0 / (beta.beta_qcd((0, 0), nf) + beta.beta_qcd((1, 0), nf) * a1)
    lhs = (
        ei.j11_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j11_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)


def test_der_nnlo_exp():
    """expanded NNLO derivative"""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6

    # Integrals are expanded to the order 0( a_s^3 ) so they can match the derivative to a_s^2
    # The corresponding prefactor  prorpotional to a_s^2 are included in the tollerance.

    # 02
    rhs = 1.0 / (
        beta.beta_qcd((0, 0), nf) * a1
        + beta.beta_qcd((1, 0), nf) * a1**2
        + beta.beta_qcd((2, 0), nf) * a1**3
    )
    lhs = (
        ei.j02_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j02_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    toll = (
        (
            -beta.b_qcd((1, 0), nf) ** 3
            + 2 * beta.b_qcd((2, 0), nf) * beta.b_qcd((1, 0), nf)
        )
        / beta.beta_qcd((0, 0), nf)
        * a1**2
    )
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(toll))
    # 12
    rhs = 1.0 / (
        beta.beta_qcd((0, 0), nf)
        + beta.beta_qcd((1, 0), nf) * a1
        + beta.beta_qcd((2, 0), nf) * a1**2
    )
    lhs = (
        ei.j12_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j12_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    toll = (
        (beta.b_qcd((1, 0), nf) ** 2 - beta.b_qcd((2, 0), nf))
        / beta.beta_qcd((0, 0), nf)
        * a1**2
    )
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(toll))
    # 22
    rhs = a1 / (
        beta.beta_qcd((0, 0), nf)
        + beta.beta_qcd((1, 0), nf) * a1
        + beta.beta_qcd((2, 0), nf) * a1**2
    )
    lhs = (
        ei.j22_expanded(a1 + 0.5 * delta_a, a0, nf)
        - ei.j22_expanded(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(
        rhs,
        lhs,
        atol=np.abs(beta.b_qcd((1, 0), nf) / beta.beta_qcd((0, 0), nf) * a1**2),
    )


def test_der_nnlo_exa():
    """exact NNLO derivative"""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 02
    rhs = 1.0 / (
        beta.beta_qcd((0, 0), nf) * a1
        + beta.beta_qcd((1, 0), nf) * a1**2
        + beta.beta_qcd((2, 0), nf) * a1**3
    )
    lhs = (
        ei.j02_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j02_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 12
    rhs = 1.0 / (
        beta.beta_qcd((0, 0), nf)
        + beta.beta_qcd((1, 0), nf) * a1
        + beta.beta_qcd((2, 0), nf) * a1**2
    )
    lhs = (
        ei.j12_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j12_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 12
    rhs = a1 / (
        beta.beta_qcd((0, 0), nf)
        + beta.beta_qcd((1, 0), nf) * a1
        + beta.beta_qcd((2, 0), nf) * a1**2
    )
    lhs = (
        ei.j22_exact(a1 + 0.5 * delta_a, a0, nf)
        - ei.j22_exact(a1 - 0.5 * delta_a, a0, nf)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
