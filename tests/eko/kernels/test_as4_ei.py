import numpy as np

from eko import beta
from eko.kernels import as4_evolution_integrals as as4_ei
from eko.kernels.evolution_integrals import j12


def test_zero():
    """No evolution results in exp(0)"""
    beta0 = np.random.random()
    b_list = np.random.random(3)
    roots = np.random.random(3)
    for fnc in [
        as4_ei.j13_exact,
        as4_ei.j23_exact,
        as4_ei.j33_exact,
    ]:
        np.testing.assert_allclose(fnc(1, 1, beta0, b_list, roots), 0)
    for fnc in [as4_ei.j13_expanded, as4_ei.j23_expanded]:
        np.testing.assert_allclose(fnc(1, 1, beta0, b_list), 0)
    np.testing.assert_allclose(as4_ei.j33_expanded(1, 1, beta0), 0)


def test_roots():
    b_list = [12.34, 0.56, 7.68]
    roots = as4_ei.roots(b_list)

    def poly(x):
        return 1 + b_list[0] * x + b_list[1] * x**2 + b_list[2] * x**3

    np.testing.assert_allclose([poly(x) for x in roots], np.zeros(3), atol=1e-14)


def test_derivative():
    b_list = [5.45, 1.23, 678]
    x = 0.124
    delta_x = -1e-6

    def poly(x):
        return 1 + b_list[0] * x + b_list[1] * x**2 + b_list[2] * x**3

    lhs = (poly(x + 0.5 * delta_x) - poly(x - 0.5 * delta_x)) / delta_x
    rhs = as4_ei.derivative(x, b_list)
    np.testing.assert_allclose(lhs, rhs)


def test_der_n3lo_exa():
    """Exact N3LO derivative."""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6

    beta0 = beta.beta_qcd((2, 0), nf)
    b1 = beta.beta_qcd((3, 0), nf)
    b2 = beta.beta_qcd((4, 0), nf)
    b3 = beta.beta_qcd((5, 0), nf)
    b_list = [b1, b2, b3]

    den = beta0 * (1 + b1 * a1 + b2 * a1**2 + b3 * a1**3)
    roots = as4_ei.roots(b_list)

    # 33
    rhs = a1**2 / den
    j33p = as4_ei.j33_exact(a1 + 0.5 * delta_a, a0, beta0, b_list, roots)
    j33m = as4_ei.j33_exact(a1 - 0.5 * delta_a, a0, beta0, b_list, roots)
    lhs = (j33p - j33m) / delta_a
    np.testing.assert_allclose(rhs, lhs)
    # 23
    rhs = a1 / den
    j23p = as4_ei.j23_exact(a1 + 0.5 * delta_a, a0, beta0, b_list, roots)
    j23m = as4_ei.j23_exact(a1 - 0.5 * delta_a, a0, beta0, b_list, roots)
    lhs = (j23p - j23m) / delta_a
    np.testing.assert_allclose(rhs, lhs)
    # 13
    rhs = 1.0 / den
    j13p = as4_ei.j13_exact(a1 + 0.5 * delta_a, a0, beta0, b_list, roots)
    j13m = as4_ei.j13_exact(a1 - 0.5 * delta_a, a0, beta0, b_list, roots)
    lhs = (j13p - j13m) / delta_a
    np.testing.assert_allclose(rhs, lhs)

    # 03
    rhs = 1.0 / (a1 * den)
    j00p = j12(a1 + 0.5 * delta_a, a0, beta0)
    j00m = j12(a1 - 0.5 * delta_a, a0, beta0)
    lhs = (
        as4_ei.j03_exact(j00p, j13p, j23p, j33p, b_list)
        - as4_ei.j03_exact(j00m, j13m, j23m, j33m, b_list)
    ) / delta_a
    np.testing.assert_allclose(rhs, lhs)


def test_der_n3lo_exp():
    """Expanded N3LO derivative."""
    nf = 3
    a0 = 0.3
    a1 = 0.1
    delta_a = -1e-6

    beta0 = beta.beta_qcd((2, 0), nf)
    b1 = beta.beta_qcd((3, 0), nf)
    b2 = beta.beta_qcd((4, 0), nf)
    b3 = beta.beta_qcd((5, 0), nf)
    b_list = [b1, b2, b3]

    den = beta0 * (1 + b1 * a1 + b2 * a1**2 + b3 * a1**3)

    # 33
    rhs = a1**2 / den
    j33p = as4_ei.j33_expanded(a1 + 0.5 * delta_a, a0, beta0)
    j33m = as4_ei.j33_expanded(a1 - 0.5 * delta_a, a0, beta0)
    lhs = (j33p - j33m) / delta_a
    toll = np.abs(-(b1 * a1**3) / beta0)
    np.testing.assert_allclose(rhs, lhs, atol=toll)
    # 23
    rhs = a1 / den
    j23p = as4_ei.j23_expanded(a1 + 0.5 * delta_a, a0, beta0, b_list)
    j23m = as4_ei.j23_expanded(a1 - 0.5 * delta_a, a0, beta0, b_list)
    lhs = (j23p - j23m) / delta_a
    toll = np.abs(((b1**2 - b2) * a1**3) / beta0)
    np.testing.assert_allclose(rhs, lhs, atol=toll)
    # 13
    rhs = 1.0 / den
    j13p = as4_ei.j13_expanded(a1 + 0.5 * delta_a, a0, beta0, b_list)
    j13m = as4_ei.j13_expanded(a1 - 0.5 * delta_a, a0, beta0, b_list)
    lhs = (j13p - j13m) / delta_a
    toll = np.abs(((-(b1**3) + 2 * b1 * b2 - b3) * a1**3) / beta0)
    np.testing.assert_allclose(rhs, lhs, atol=toll)

    # 03
    rhs = 1.0 / (a1 * den)
    j00p = j12(a1 + 0.5 * delta_a, a0, beta0)
    j00m = j12(a1 - 0.5 * delta_a, a0, beta0)
    lhs = (
        as4_ei.j03_expanded(j00p, j13p, j23p, j33p, b_list)
        - as4_ei.j03_expanded(j00m, j13m, j23m, j33m, b_list)
    ) / delta_a
    toll = np.abs(((b1 * 4 - 3 * b1 * b2 + b2**2 + 2 * b1 * b3) * a1**3) / beta0)
    np.testing.assert_allclose(rhs, lhs, atol=toll)
