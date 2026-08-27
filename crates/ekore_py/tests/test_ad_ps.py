import numpy as np
import pytest
from ekore_rs import Cache, ad_ps
from ekore_rs.constants import PID_NSM_U, PID_NSP
from numpy.testing import assert_allclose

CF = 4.0 / 3.0
CA = 3.0
TR = 1.0 / 2.0


@pytest.mark.parametrize("order_qcd, expected_len", [(0, 0), (1, 1), (2, 2)])
def test_gamma_ns_qcd_shape(order_qcd, expected_len):
    c = Cache(0.0 + 0.0j)
    g = ad_ps.gamma_ns_qcd(order_qcd, PID_NSP, c, nf=3)
    assert g.shape == (expected_len,)
    assert g.dtype == np.complex128


@pytest.mark.parametrize(
    "order_qcd, expected_shape", [(0, (0, 2, 2)), (1, (1, 2, 2)), (2, (2, 2, 2))]
)
def test_gamma_singlet_qcd_shape(order_qcd, expected_shape):
    c = Cache(0.0 + 0.0j)
    g = ad_ps.gamma_singlet_qcd(order_qcd, c, nf=3)
    assert g.shape == expected_shape
    assert g.dtype == np.complex128


def test_order_qcd_out_of_range_raises():
    c = Cache(0.0 + 0.0j)
    with pytest.raises(ValueError):
        ad_ps.gamma_ns_qcd(3, PID_NSP, c, nf=3)
    with pytest.raises(ValueError):
        ad_ps.gamma_singlet_qcd(3, c, nf=3)


def test_gamma_ns_qcd():
    nf = 3
    c = Cache(1.0 + 0.0j)

    r = ad_ps.gamma_ns_qcd(2, PID_NSP, c, nf)

    assert_allclose(r[0], 0.0, atol=1e-14)
    assert_allclose(r, np.zeros_like(r), atol=2e-6)


def test_gamma_singlet_qcd():
    nf = 5
    c = Cache(2.0 + 0.0j)

    g = ad_ps.gamma_singlet_qcd(2, c, nf)

    # LO
    assert_allclose(g[0, 0, 0] + g[0, 1, 0], 4.0 * CF / 3.0, atol=1e-12)
    assert_allclose(g[0, 0, 1] + g[0, 1, 1], 3.0 + nf / 3.0, atol=1e-12)

    # NLO
    expected_nlo = (
        4.0
        * nf
        * (
            0.574074074 * CF
            - 2.0 * CA * (-7.0 / 18.0 + 1.0 / 6.0 * (5.0 - np.pi**2 / 3.0))
        )
        * TR
    )
    assert_allclose(-g[1, 0, 1], expected_nlo, atol=1e-9)


def test_guards_unknown_mode_raises():
    nf = 4
    c = Cache(1.234 + 0.0j)

    with pytest.raises(ValueError):
        ad_ps.gamma_ns_qcd(2, PID_NSM_U, c, nf)
