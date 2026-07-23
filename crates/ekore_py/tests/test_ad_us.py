import numpy as np
import pytest
from ekore_py import Cache, ad_us
from ekore_py.constants import (
    PID_NSM,
    PID_NSM_D,
    PID_NSM_U,
    PID_NSP,
    PID_NSP_D,
    PID_NSP_U,
    PID_NSV,
)
from numpy.testing import assert_allclose

MAX_ORDER_QCD = 4
MAX_ORDER_QED = 2


@pytest.mark.parametrize(
    "order_qcd, expected_len", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
)
def test_gamma_ns_qcd_shape(order_qcd, expected_len):
    c = Cache(0.0 + 0.0j)
    var = (0, 0, 0)
    g = ad_us.gamma_ns_qcd(order_qcd, PID_NSP, c, nf=3, n3lo_variation=var)
    assert g.shape == (expected_len,)
    assert g.dtype == np.complex128


@pytest.mark.parametrize(
    "order_qcd, expected_shape",
    [(0, (0, 2, 2)), (1, (1, 2, 2)), (2, (2, 2, 2)), (3, (3, 2, 2)), (4, (4, 2, 2))],
)
def test_gamma_singlet_qcd_shape(order_qcd, expected_shape):
    c = Cache(0.0 + 0.0j)
    var = (0, 0, 0, 0)
    g = ad_us.gamma_singlet_qcd(order_qcd, c, nf=3, n3lo_variation=var)
    assert g.shape == expected_shape
    assert g.dtype == np.complex128


@pytest.mark.parametrize(
    "order_qcd, order_qed, expected_shape",
    [(3, 2, (4, 3)), (1, 1, (2, 2))],
)
def test_gamma_ns_qed_shape(order_qcd, order_qed, expected_shape):
    c = Cache(0.0 + 0.0j)
    var = (0, 0, 0)
    g = ad_us.gamma_ns_qed(order_qcd, order_qed, PID_NSP_U, c, nf=3, n3lo_variation=var)
    assert g.shape == expected_shape
    assert g.dtype == np.complex128


@pytest.mark.parametrize(
    "order_qcd, order_qed, expected_shape",
    [(3, 2, (4, 3, 4, 4))],
)
def test_gamma_singlet_qed_shape(order_qcd, order_qed, expected_shape):
    c = Cache(0.0 + 0.0j)
    var = (0, 0, 0, 0, 0, 0, 0)
    g = ad_us.gamma_singlet_qed(order_qcd, order_qed, c, nf=3, n3lo_variation=var)
    assert g.shape == expected_shape
    assert g.dtype == np.complex128


@pytest.mark.parametrize(
    "order_qcd, order_qed, expected_shape",
    [(3, 2, (4, 3, 2, 2))],
)
def test_gamma_valence_qed_shape(order_qcd, order_qed, expected_shape):
    c = Cache(0.0 + 0.0j)
    var = (0, 0, 0)
    g = ad_us.gamma_valence_qed(order_qcd, order_qed, c, nf=3, n3lo_variation=var)
    assert g.shape == expected_shape
    assert g.dtype == np.complex128


def test_order_out_of_range_raises():
    c = Cache(0.0 + 0.0j)
    with pytest.raises(ValueError):
        ad_us.gamma_ns_qcd(
            MAX_ORDER_QCD + 1, PID_NSP, c, nf=3, n3lo_variation=(0, 0, 0)
        )
    with pytest.raises(ValueError):
        ad_us.gamma_singlet_qcd(MAX_ORDER_QCD + 1, c, nf=3, n3lo_variation=(0, 0, 0, 0))
    with pytest.raises(ValueError):
        ad_us.gamma_ns_qed(
            MAX_ORDER_QCD + 1, 1, PID_NSP_U, c, nf=4, n3lo_variation=(0, 0, 0)
        )
    with pytest.raises(ValueError):
        ad_us.gamma_ns_qed(
            1, MAX_ORDER_QED + 1, PID_NSP_U, c, nf=4, n3lo_variation=(0, 0, 0)
        )
    with pytest.raises(ValueError):
        ad_us.gamma_valence_qed(MAX_ORDER_QCD + 1, 1, c, nf=4, n3lo_variation=(0, 0, 0))
    with pytest.raises(ValueError):
        ad_us.gamma_singlet_qed(
            1, MAX_ORDER_QED + 1, c, nf=4, n3lo_variation=(0, 0, 0, 0, 0, 0, 0)
        )


def test_unknown_ns_mode_raises():
    c = Cache(1.234 + 0.0j)
    with pytest.raises(ValueError):
        ad_us.gamma_ns_qed(2, 0, 10106, c, nf=4, n3lo_variation=(0, 0, 0))


def test_gamma_ns_qcd():
    nf3, nf5 = 3, 5
    var = (0, 0, 0)
    c = Cache(1.0 + 0.0j)
    nsm_refs = [0.06776363, 0.064837, 0.07069]
    nss_refs = [-0.01100459, -0.00779938, -0.0142098]

    r3 = ad_us.gamma_ns_qcd(3, PID_NSP, c, nf3, var)
    assert_allclose(r3[0], 0.0, atol=1e-14)

    r2 = ad_us.gamma_ns_qcd(2, PID_NSM, c, nf3, var)
    assert_allclose(r2, np.zeros_like(r2), atol=2e-6)

    r3 = ad_us.gamma_ns_qcd(3, PID_NSM, c, nf3, var)
    assert_allclose(r3, np.zeros_like(r3), atol=2e-4)

    r3 = ad_us.gamma_ns_qcd(3, PID_NSV, c, nf3, var)
    assert_allclose(r3, np.zeros_like(r3), atol=8e-4)

    for v in range(3):
        var_nsm = (0, v, 0)
        r4 = ad_us.gamma_ns_qcd(4, PID_NSM, c, nf5, var_nsm)
        assert_allclose(r4[3], nsm_refs[v], atol=6e-6)

        var_nsv = (0, 0, v)
        r4 = ad_us.gamma_ns_qcd(4, PID_NSV, c, nf5, var_nsv)
        assert_allclose(r4[3], nsm_refs[v] + nss_refs[v], atol=1e-5)

    r4 = ad_us.gamma_ns_qcd(4, PID_NSP, c, nf3, var)
    assert np.any(np.abs(r4) ** 2 > 1e-12), "expected a non-trivial N3LO entry"


def test_gamma_singlet_qcd():
    nf = 5
    c = Cache(2.0 + 0.0j)
    quark_refs = [0.053441, 0.225674, -0.118792]
    gluon_refs = [-0.0300842, 0.283004, -0.343172]

    for imod in range(3):
        var = (imod, imod, imod, imod)
        g4 = ad_us.gamma_singlet_qcd(4, c, nf, var)

        qq, qg = g4[3, 0, 0], g4[3, 0, 1]
        gq, gg = g4[3, 1, 0], g4[3, 1, 1]

        assert_allclose(qq + gq, quark_refs[imod], atol=2e-6)
        assert_allclose(qg + gg, gluon_refs[imod], atol=2e-5)


def test_gamma_ns_qed():
    nf = 3
    var = (0, 0, 0)
    c = Cache(1.0 + 0.0j)
    nsm_pids = [PID_NSM_U, PID_NSM_D]

    for pid in nsm_pids:
        r1 = ad_us.gamma_ns_qed(1, 1, pid, c, nf, var)
        assert_allclose(r1, np.zeros_like(r1), atol=1e-5)

    r1 = ad_us.gamma_ns_qed(1, 1, PID_NSP_U, c, nf, var)
    assert_allclose(r1[0, 0], 0.0, atol=1e-15)
    assert_allclose(r1[0, 1], 0.0, atol=1e-5)

    r1 = ad_us.gamma_ns_qed(1, 1, PID_NSP_D, c, nf, var)
    assert_allclose(r1[0, 0], 0.0, atol=1e-15)
    assert_allclose(r1[0, 1], 0.0, atol=1e-5)

    for pid in nsm_pids:
        r2 = ad_us.gamma_ns_qed(1, 2, pid, c, nf, var)
        assert_allclose(r2, np.zeros_like(r2), atol=1e-5)

    for pid in nsm_pids:
        r2 = ad_us.gamma_ns_qed(2, 1, pid, c, nf, var)
        assert_allclose(r2, np.zeros_like(r2), atol=1e-5)

    for pid in nsm_pids:
        r3 = ad_us.gamma_ns_qed(3, 1, pid, c, nf, var)
        assert_allclose(r3, np.zeros_like(r3), atol=1e-3)


def test_gamma_valence_qed():
    nf = 3
    var = (0, 0, 0)
    c = Cache(2.0 + 0.0j)

    g = ad_us.gamma_valence_qed(3, 2, c, nf, var)

    assert_allclose(g[0, 0], np.zeros((2, 2)), atol=1e-15)

    assert_allclose(g[3, 0, 0, 0], 459.646893789751, atol=1e-5)
    assert_allclose(g[3, 0, 0, 1], 0.0, atol=1e-5)
    assert_allclose(g[3, 0, 1, 0], 0.0, atol=1e-5)
    assert_allclose(g[3, 0, 1, 1], 437.60340375, atol=1e-5)


def test_gamma_singlet_qed():
    nf = 3
    var = (0, 0, 0, 0, 0, 0, 0)
    c = Cache(2.0 + 0.0j)

    g = ad_us.gamma_singlet_qed(3, 2, c, nf, var)

    assert_allclose(g[0, 0], np.zeros((4, 4)), atol=1e-15)

    ref = np.array(
        [
            [3.857918949669738, 0.0, -290.42193908689745, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-3.859554320251334, 0.0, 290.4252052962147, 0.0],
            [0.0, 0.0, 0.0, 448.0752570151872],
        ]
    )
    assert_allclose(g[3, 0].real, ref, atol=1e-5)
    assert_allclose(g[3, 0].imag, np.zeros((4, 4)), atol=1e-5)
