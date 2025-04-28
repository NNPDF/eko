import warnings

import numpy as np
import pytest

import ekore.anomalous_dimensions.unpolarized.space_like as ad
from eko import beta
from eko.kernels import EvoMethods
from eko.kernels import non_singlet as ns


def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    gamma_ns = np.array([1 + 0.0j, 1 + 0j, 1 + 0j, 1 + 0j])
    for order in [1, 2, 3, 4]:
        for method in EvoMethods:
            np.testing.assert_allclose(
                ns.dispatcher((order, 0), method, gamma_ns, 1.0, 1.0, nf),
                1.0,
            )
            np.testing.assert_allclose(
                ns.dispatcher(
                    (order, 0),
                    method,
                    np.zeros(order + 1, dtype=complex),
                    2.0,
                    1.0,
                    nf,
                ),
                1.0,
            )


def test_ode_lo():
    nf = 3
    gamma_ns = np.random.rand(1) + 0j
    delta_a = -1e-6
    a0 = 0.3
    for a1 in [0.1, 0.2]:
        r = a1 * gamma_ns / (beta.beta_qcd((2, 0), nf) * a1**2)
        for method in EvoMethods:
            rhs = r * ns.dispatcher((1, 0), method, gamma_ns, a1, a0, nf)
            lhs = (
                ns.dispatcher(
                    (1, 0),
                    method,
                    gamma_ns,
                    a1 + 0.5 * delta_a,
                    a0,
                    nf,
                )
                - ns.dispatcher(
                    (1, 0),
                    method,
                    gamma_ns,
                    a1 - 0.5 * delta_a,
                    a0,
                    nf,
                )
            ) / delta_a
            np.testing.assert_allclose(lhs, rhs, atol=np.abs(delta_a))


def test_ode_nlo():
    nf = 3
    gamma_ns = np.random.rand(2) + 0j
    delta_a = -1e-6
    a0 = 0.3
    for a1 in [0.1, 0.2]:
        r = (a1 * gamma_ns[0] + a1**2 * gamma_ns[1]) / (
            beta.beta_qcd((2, 0), nf) * a1**2 + beta.beta_qcd((3, 0), nf) * a1**3
        )
        for method in [EvoMethods.ITERATE_EXACT]:
            rhs = r * ns.dispatcher((2, 0), method, gamma_ns, a1, a0, nf)
            lhs = (
                ns.dispatcher(
                    (2, 0),
                    method,
                    gamma_ns,
                    a1 + 0.5 * delta_a,
                    a0,
                    nf,
                )
                - ns.dispatcher(
                    (2, 0),
                    method,
                    gamma_ns,
                    a1 - 0.5 * delta_a,
                    a0,
                    nf,
                )
            ) / delta_a
            np.testing.assert_allclose(lhs, rhs, atol=np.abs(delta_a))


def test_ode_nnlo():
    nf = 3
    gamma_ns = np.random.rand(3) + 0j
    delta_a = -1e-6
    a0 = 0.3
    for a1 in [0.1, 0.2]:
        r = (gamma_ns[0] + a1 * gamma_ns[1] + a1**2 * gamma_ns[2]) / (
            beta.beta_qcd((2, 0), nf) * a1
            + beta.beta_qcd((3, 0), nf) * a1**2
            + beta.beta_qcd((4, 0), nf) * a1**3
        )
        for method in [EvoMethods.ITERATE_EXACT]:
            rhs = r * ns.dispatcher((3, 0), method, gamma_ns, a1, a0, nf)
            lhs = (
                ns.dispatcher((3, 0), method, gamma_ns, a1 + 0.5 * delta_a, a0, nf)
                - ns.dispatcher((3, 0), method, gamma_ns, a1 - 0.5 * delta_a, a0, nf)
            ) / delta_a
            np.testing.assert_allclose(lhs, rhs)


def test_ode_n3lo():
    nf = 3
    gamma_ns = np.random.rand(4) + 0j
    delta_a = -1e-6
    a0 = 0.3
    for a1 in [0.1, 0.2]:
        r = (
            gamma_ns[0] + a1 * gamma_ns[1] + a1**2 * gamma_ns[2] + a1**3 * gamma_ns[3]
        ) / (
            beta.beta_qcd((2, 0), nf) * a1
            + beta.beta_qcd((3, 0), nf) * a1**2
            + beta.beta_qcd((4, 0), nf) * a1**3
            + beta.beta_qcd((5, 0), nf) * a1**4
        )
        for method in [EvoMethods.ITERATE_EXACT]:
            rhs = r * ns.dispatcher((4, 0), method, gamma_ns, a1, a0, nf)
            lhs = (
                ns.dispatcher((4, 0), method, gamma_ns, a1 + 0.5 * delta_a, a0, nf)
                - ns.dispatcher((4, 0), method, gamma_ns, a1 - 0.5 * delta_a, a0, nf)
            ) / delta_a
            np.testing.assert_allclose(lhs, rhs)


def test_error(monkeypatch):
    monkeypatch.setattr("eko.beta.beta_qcd", lambda *_args: 1.0)
    with pytest.raises(NotImplementedError, match="order is not implemented"):
        ns.dispatcher(
            (5, 0), EvoMethods.ITERATE_EXACT, np.random.rand(3) + 0j, 0.2, 0.1, 3
        )
    with pytest.raises(NotImplementedError):
        ad.gamma_ns((2, 0), 10202, 1, (0, 0, 0, 0, 0, 0, 0), 3)


def test_gamma_usage():
    a1 = 0.1
    a0 = 0.3
    nf = 3
    # first check that at order=n only uses the matrices up n
    gamma_ns = np.full(4, np.nan)
    for order in range(1, 5):
        gamma_ns[order - 1] = np.random.rand()
        for method in EvoMethods:
            r = ns.dispatcher((order, 0), method, gamma_ns, a1, a0, nf)
            assert not np.isnan(r)
    # second check that at order=n the actual matrix n is used
    for order in range(1, 5):
        gamma_ns = np.random.rand(order)
        gamma_ns[order - 1] = np.nan
        for method in EvoMethods:
            if method is EvoMethods.ORDERED_TRUNCATED:
                # we are actually dividing by a np.nan,
                # since the sum of U vec is nan
                warnings.simplefilter("ignore", RuntimeWarning)
            r = ns.dispatcher((order, 0), method, gamma_ns, a1, a0, nf)
            assert np.isnan(r)
