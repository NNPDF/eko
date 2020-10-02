# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.operator.member import OpMember


def mkOM(name, shape):
    ma, mae = np.random.rand(2, *shape)
    return OpMember(ma, mae, name), ma, mae


class TestOpMember:
    shape = (2, 2)

    def _mkOM(self, name):
        return mkOM(name, self.shape)

    def test_name(self):
        ma, mae = np.random.rand(2, *self.shape)
        for i in ["S", "g"]:
            for t in ["S", "g"]:
                n = f"{t}.{i}"
                a = OpMember(ma, mae, n)
                # splitting
                assert a.input == i
                assert a.target == t
                # __str__
                assert str(a) == n
        # check wrongs
        for n in [".", "a.", ".b", "ab"]:
            with pytest.raises(ValueError):
                a = OpMember(ma, mae, n)
                _ = a.input
                _ = a.target

    def test_add(self):
        a, ma, mae = self._mkOM("S.S")
        b, mb, mbe = self._mkOM("S.S")

        # plain sum and diff
        for c, d in [(a + b, a - b), (b + a, -(b - a))]:
            assert_almost_equal(c.value, ma + mb)
            assert_almost_equal(c.error, mae + mbe)
            assert c.name == "S.S"
            assert_almost_equal(d.value, ma - mb)
            assert_almost_equal(d.error, mae + mbe)
            assert d.name == "S.S"

        # add to 0
        for c in [0 + a, a + 0, a - 0, 0 - a]:
            assert_almost_equal(c.value, ma)
            assert_almost_equal(c.error, mae)
            assert c.name == "S.S"

        # errors:
        # non-matching name
        with pytest.raises(ValueError):
            _ = a + OpMember(mb, mbe, "S.g")
        # non-phyiscal
        with pytest.raises(ValueError):
            _ = a + OpMember(mb, mbe, "NS_v")
        # wrong other
        with pytest.raises(ValueError):
            _ = 1 + OpMember(mb, mbe, "S.g")
        with pytest.raises(ValueError):
            _ = OpMember(mb, mbe, "S.g") + 1
        with pytest.raises(NotImplementedError):
            _ = [] + OpMember(mb, mbe, "S.g")
        with pytest.raises(NotImplementedError):
            _ = OpMember(mb, mbe, "S.g") + []

    def test_neg(self):
        a, ma, mae = self._mkOM("S.S")
        b = -a
        assert_almost_equal(b.value, -ma)
        assert_almost_equal(b.error, mae)
        assert b.name == a.name
        # is inverse?
        c = a + b
        assert_almost_equal(c.value, np.zeros(self.shape))

    def test_add_relatives(self):
        a, ma, mae = self._mkOM("S.S")
        b, mb, mbe = self._mkOM("S.S")
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
        a, ma, _mae = self._mkOM("S.S")
        b, _mb, mbe = self._mkOM("S.S")
        c = OpMember(ma, mbe, "S.S")

        assert a != b
        assert a == c

    def test_mul(self):
        a, ma, mae = self._mkOM("S.g")
        b, mb, mbe = self._mkOM("g.S")
        # plain product
        c = a * b
        assert c.name == "S.S"
        assert_almost_equal(c.value, ma @ mb)
        assert_almost_equal(c.error, np.abs(mae @ mb) + np.abs(ma @ mbe))
        d = b * a
        assert d.name == "g.g"
        assert_almost_equal(d.value, mb @ ma)
        assert_almost_equal(d.error, np.abs(mbe @ ma) + np.abs(mb @ mae))
        assert c != d

        # non-physical case
        e, me, mee = self._mkOM("NS_v")
        f, mf, mfe = self._mkOM("NS_v")
        g = e * f
        assert g.name == "NS_v.NS_v"
        assert_almost_equal(g.value, me @ mf)
        assert_almost_equal(g.error, np.abs(mee @ mf) + np.abs(me @ mfe))

        # errors
        # div is useless
        with pytest.raises(TypeError):
            _ = c / d
        # non-phyiscal
        with pytest.raises(ValueError):
            _ = a * OpMember(mb, mbe, "NS_v")
        # wrong name
        with pytest.raises(ValueError):
            _ = a * OpMember(mb, mbe, "S.S")
        # wrong other
        with pytest.raises(NotImplementedError):
            _ = c * []
        with pytest.raises(NotImplementedError):
            _ = [] * c

    def test_apply_pdf(self):
        a, ma, mae = self._mkOM("S.S")
        pdf = np.random.rand(self.shape[0])
        prod = a.apply_pdf(pdf)
        assert len(prod) == 2
        assert_almost_equal(prod[0], ma @ pdf)
        assert_almost_equal(prod[1], mae @ pdf)

    def test_join_1(self):
        # two steps
        a, _, _ = self._mkOM("V.V")
        b, _, _ = self._mkOM("V.V")
        steps = [{a.name: a}, {b.name: b}]
        paths = [["V.V", "V.V"]]
        j2 = OpMember.join(steps, paths)
        assert j2.name == "V.V"
        assert j2 == a * b
        # three steps
        c, _, _ = self._mkOM("V.V")
        steps = [{a.name: a}, {b.name: b}, {c.name: c}]
        paths = [["V.V", "V.V", "V.V"]]
        j3 = OpMember.join(steps, paths)
        assert j3.name == "V.V"
        assert j3 == (a * b) * c
        assert j3 == a * (b * c)
        assert j3 == j2 * c

    def test_join_2(self):
        # two non-trivial steps
        ss0, _, _ = self._mkOM("S.S")
        sg0, _, _ = self._mkOM("S.g")
        ss1, _, _ = self._mkOM("S.S")
        gs1, _, _ = self._mkOM("g.S")
        steps = [{ss0.name: ss0, sg0.name: sg0}, {ss1.name: ss1, gs1.name: gs1}]
        paths = [["S.S", "S.S"], ["S.g", "g.S"]]
        j2 = OpMember.join(steps, paths)
        assert j2.name == "S.S"
        assert j2 == ss0 * ss1 + sg0 * gs1
