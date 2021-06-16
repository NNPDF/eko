# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.member import OpMember, OperatorBase, MemberName


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return OpMember(ma, mae), ma, mae


class TestOpMember:
    shape = (2, 2)

    def _mkOM(self):
        return mkOM(self.shape)

    def test_add(self):
        a, ma, mae = self._mkOM()
        b, mb, mbe = self._mkOM()

        # plain sum and diff
        for c, d in [(a + b, a - b), (b + a, -(b - a))]:
            assert_almost_equal(c.value, ma + mb)
            assert_almost_equal(c.error, mae + mbe)
            assert_almost_equal(d.value, ma - mb)
            assert_almost_equal(d.error, mae + mbe)

        # add to 0
        for c in [0 + a, a + 0, a - 0, 0 - a]:
            assert_almost_equal(c.value, ma)
            assert_almost_equal(c.error, mae)

        # errors:
        # wrong other
        with pytest.raises(ValueError):
            _ = 1 + OpMember(mb, mbe)
        with pytest.raises(ValueError):
            _ = OpMember(mb, mbe) + 1
        with pytest.raises(NotImplementedError):
            _ = [] + OpMember(mb, mbe)
        with pytest.raises(NotImplementedError):
            _ = OpMember(mb, mbe) + []

    def test_neg(self):
        a, ma, mae = self._mkOM()
        b = -a
        assert_almost_equal(b.value, -ma)
        assert_almost_equal(b.error, mae)
        # is inverse?
        c = a + b
        assert_almost_equal(c.value, np.zeros(self.shape))

    def test_add_relatives(self):
        a, ma, mae = self._mkOM()
        b, mb, mbe = self._mkOM()
        # abstract sum
        c = sum([a, b])
        assert_almost_equal(c.value, ma + mb)
        assert_almost_equal(c.error, mae + mbe)
        # beware of the error computation here below ...
        # +=
        a += b
        assert_almost_equal(a.value, ma + mb)
        assert_almost_equal(a.error, mae + mbe)

        # -=
        a -= b
        assert_almost_equal(a.value, ma)
        assert_almost_equal(a.error, mae + 2.0 * mbe)

    def test_eq(self):
        a, ma, _mae = self._mkOM()
        b, _mb, mbe = self._mkOM()
        c = OpMember(ma, mbe)

        assert a != b
        assert a == c

    def test_matmul(self):
        a, ma, mae = self._mkOM()
        b, mb, mbe = self._mkOM()
        # plain product
        c = a @ b
        assert_almost_equal(c.value, ma @ mb)
        assert_almost_equal(c.error, np.abs(mae @ mb) + np.abs(ma @ mbe))
        d = b @ a
        assert_almost_equal(d.value, mb @ ma)
        assert_almost_equal(d.error, np.abs(mbe @ ma) + np.abs(mb @ mae))
        assert c != d

    def test_mul(self):
        a, ma, mae = self._mkOM()
        b = np.pi
        # plain product
        c, d = a * b, b * a
        assert_almost_equal(c.value, ma * b)
        assert_almost_equal(c.error, mae * b)
        assert_almost_equal(d.value, ma * b)
        assert_almost_equal(d.error, mae * b)

        # errors
        # div is useless
        with pytest.raises(TypeError):
            _ = c / d
        # wrong other
        with pytest.raises(NotImplementedError):
            _ = c * []
        with pytest.raises(NotImplementedError):
            _ = [] * c

    def test_copy(self):
        a, _, _ = self._mkOM()
        b = a.copy()
        assert_almost_equal(a.value, b.value)
        assert_almost_equal(a.error, b.error)


class TestOperatorBase:
    def test_getitem(self):
        opm = {MemberName("S.S"): 1}
        ob = OperatorBase(opm, 0.0)
        assert ob[MemberName("S.S")] == ob["S.S"]
