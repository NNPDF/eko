import numpy as np
import pytest
from ekore_py import Cache, ome_us
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    "order, expected_shape", [(0, (0, 3, 3)), (1, (1, 3, 3)), (2, (2, 3, 3))]
)
def test_a_singlet_shape(order, expected_shape):
    c = Cache(0.0 + 0.0j)
    a = ome_us.A_singlet(order, c, nf=3, L=0.0)
    assert a.shape == expected_shape
    assert a.dtype == np.complex128


@pytest.mark.parametrize(
    "order, expected_shape", [(0, (0, 2, 2)), (1, (1, 2, 2)), (2, (2, 2, 2))]
)
def test_a_non_singlet_shape(order, expected_shape):
    c = Cache(0.0 + 0.0j)
    a = ome_us.A_non_singlet(order, c, nf=3, L=0.0)
    assert a.shape == expected_shape
    assert a.dtype == np.complex128


def test_a_non_singlet():
    nf = 5
    L = 0.0
    c = Cache(1.0 + 0.0j)

    a = ome_us.A_non_singlet(2, c, nf, L)

    # LO
    assert_allclose(a[0], np.zeros((2, 2)), atol=1e-14)
    # NNLO
    assert_allclose(a[1], np.zeros((2, 2)), atol=1e-14)


def test_a_singlet():
    nf = 5
    L = 100.0
    c = Cache(2.0 + 0.0j)

    a = ome_us.A_singlet(2, c, nf, L)

    # LO
    assert_allclose(a[0, 0, 2] + a[0, 1, 2] + a[0, 2, 2], 0.0, atol=1e-10)
    assert_allclose(a[0, 0, 0] + a[0, 1, 0] + a[0, 2, 0], 0.0, atol=1e-12)

    # NNLO
    assert_allclose(a[1, 0, 0] + a[1, 1, 0] + a[1, 2, 0], 0.0, atol=2e-6)
    assert_allclose(a[1, 0, 1] + a[1, 1, 1] + a[1, 2, 1], 0.0, atol=1e-11)


def test_guards_invalid_order_raises():
    nf = 4
    L = 0.0
    c = Cache(1.234 + 0.0j)

    with pytest.raises(ValueError):
        ome_us.A_singlet(3, c, nf, L)
    with pytest.raises(ValueError):
        ome_us.A_non_singlet(3, c, nf, L)
