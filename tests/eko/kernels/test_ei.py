import numpy as np

from eko import beta
from eko.kernels import evolution_integrals as ei


def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf)
    b_vec = [beta.beta_qcd((2 + i, 0), nf) / beta0 for i in range(0, 2 + 1)]
    for fnc in [
        ei.j13_exact,
        ei.j13_expanded,
        ei.j23_exact,
        ei.j14_exact,
        ei.j14_expanded,
        ei.j24_exact,
        ei.j24_expanded,
        ei.j34_exact,
    ]:
        np.testing.assert_allclose(fnc(1, 1, beta0, b_vec), 0)
    for fnc in [
        ei.j12,
        ei.j23_expanded,
        ei.j34_expanded,
    ]:
        np.testing.assert_allclose(fnc(1, 1, beta0), 0)


def test_zero_qed():
    """No evolution results in exp(0)"""
    aem = 0.00058
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf) + aem * beta.beta_qcd((2, 1), nf)
    b_vec = [beta.beta_qcd((2 + i, 0), nf) / beta0 for i in range(0, 2 + 1)]
    for fnc in [
        ei.j23_exact,
        ei.j13_exact,
        ei.j34_exact,
        ei.j24_exact,
        ei.j14_exact,
    ]:
        np.testing.assert_allclose(fnc(1, 1, beta0, b_vec), 0)
    for fnc in [
        ei.j12,
        ei.j23_expanded,
        ei.j34_expanded,
    ]:
        np.testing.assert_allclose(fnc(1, 1, beta0), 0)


def test_der_lo():
    """LO derivative."""
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf)
    a0 = 5
    a1 = 3
    delta_a = -1e-6
    rhs = 1.0 / (beta.beta_qcd((2, 0), nf) * a1)
    lhs = (
        ei.j12(a1 + 0.5 * delta_a, a0, beta0) - ei.j12(a1 - 0.5 * delta_a, a0, beta0)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs)


def test_der_nlo_exp():
    """Expanded NLO derivative."""
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf)
    b_vec = [beta.beta_qcd((2 + i, 0), nf) / beta0 for i in range(0, 2 + 1)]
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 01
    rhs = 1.0 / (beta.beta_qcd((2, 0), nf) * a1 + beta.beta_qcd((3, 0), nf) * a1**2)
    lhs = (
        ei.j13_expanded(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j13_expanded(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    np.testing.assert_allclose(
        rhs, lhs, atol=np.abs((beta.b_qcd((3, 0), nf) * a1) ** 2)
    )
    # 11
    rhs = 1.0 / (beta.beta_qcd((2, 0), nf) + beta.beta_qcd((3, 0), nf) * a1)
    lhs = (
        ei.j23_expanded(a1 + 0.5 * delta_a, a0, beta0)
        - ei.j23_expanded(a1 - 0.5 * delta_a, a0, beta0)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(beta.b_qcd((3, 0), nf) * a1))


def test_der_nlo_exa():
    """Exact NLO derivative."""
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf)
    b_vec = [beta.beta_qcd((2 + i, 0), nf) / beta0 for i in range(0, 2 + 1)]
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 01
    rhs = 1.0 / (beta.beta_qcd((2, 0), nf) * a1 + beta.beta_qcd((3, 0), nf) * a1**2)
    lhs = (
        ei.j13_exact(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j13_exact(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 11
    rhs = 1.0 / (beta.beta_qcd((2, 0), nf) + beta.beta_qcd((3, 0), nf) * a1)
    lhs = (
        ei.j23_exact(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j23_exact(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)


def test_der_nnlo_exp():
    """Expanded NNLO derivative."""
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf)
    b_vec = [beta.beta_qcd((2 + i, 0), nf) / beta0 for i in range(0, 2 + 1)]
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6

    # Integrals are expanded to the order 0( a_s^3 ) so they can match the derivative to a_s^2
    # The corresponding prefactor  proportional to a_s^2 are included in the tolerance.

    # 02
    rhs = 1.0 / (
        beta.beta_qcd((2, 0), nf) * a1
        + beta.beta_qcd((3, 0), nf) * a1**2
        + beta.beta_qcd((4, 0), nf) * a1**3
    )
    lhs = (
        ei.j14_expanded(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j14_expanded(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    toll = (
        (
            -(beta.b_qcd((3, 0), nf) ** 3)
            + 2 * beta.b_qcd((4, 0), nf) * beta.b_qcd((3, 0), nf)
        )
        / beta.beta_qcd((2, 0), nf)
        * a1**2
    )
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(toll))
    # 12
    rhs = 1.0 / (
        beta.beta_qcd((2, 0), nf)
        + beta.beta_qcd((3, 0), nf) * a1
        + beta.beta_qcd((4, 0), nf) * a1**2
    )
    lhs = (
        ei.j24_expanded(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j24_expanded(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    toll = (
        (beta.b_qcd((3, 0), nf) ** 2 - beta.b_qcd((4, 0), nf))
        / beta.beta_qcd((2, 0), nf)
        * a1**2
    )
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(toll))
    # 22
    rhs = a1 / (
        beta.beta_qcd((2, 0), nf)
        + beta.beta_qcd((3, 0), nf) * a1
        + beta.beta_qcd((4, 0), nf) * a1**2
    )
    lhs = (
        ei.j34_expanded(a1 + 0.5 * delta_a, a0, beta0)
        - ei.j34_expanded(a1 - 0.5 * delta_a, a0, beta0)
    ) / delta_a
    np.testing.assert_allclose(
        rhs,
        lhs,
        atol=np.abs(beta.b_qcd((3, 0), nf) / beta.beta_qcd((2, 0), nf) * a1**2),
    )


def test_der_nnlo_exa():
    """Exact NNLO derivative."""
    nf = 3
    beta0 = beta.beta_qcd((2, 0), nf)
    b_vec = [beta.beta_qcd((2 + i, 0), nf) / beta0 for i in range(0, 2 + 1)]
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6
    # 02
    rhs = 1.0 / (
        beta.beta_qcd((2, 0), nf) * a1
        + beta.beta_qcd((3, 0), nf) * a1**2
        + beta.beta_qcd((4, 0), nf) * a1**3
    )
    lhs = (
        ei.j14_exact(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j14_exact(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 12
    rhs = 1.0 / (
        beta.beta_qcd((2, 0), nf)
        + beta.beta_qcd((3, 0), nf) * a1
        + beta.beta_qcd((4, 0), nf) * a1**2
    )
    lhs = (
        ei.j24_exact(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j24_exact(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
    # 12
    rhs = a1 / (
        beta.beta_qcd((2, 0), nf)
        + beta.beta_qcd((3, 0), nf) * a1
        + beta.beta_qcd((4, 0), nf) * a1**2
    )
    lhs = (
        ei.j34_exact(a1 + 0.5 * delta_a, a0, beta0, b_vec)
        - ei.j34_exact(a1 - 0.5 * delta_a, a0, beta0, b_vec)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs, atol=np.abs(delta_a))  # in fact O(delta_a^2)
