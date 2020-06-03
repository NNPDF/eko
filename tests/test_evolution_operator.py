# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.evolution_operator import OperatorMember

class TestOperatorMember:
    shape = (2,2)
    def test_name(self):
        ma,mae = np.random.rand(2,*self.shape)
        for i in ["S","g"]:
            for t in ["S","g"]:
                n = f"{t}.{i}"
                a = OperatorMember(ma,mae,n)
                # splitting
                assert a.input == i
                assert a.target == t
                # __str__
                assert str(a) == n
        # check wrongs
        for n in [".","a.",".b","ab"]:
            with pytest.raises(ValueError):
                o = OperatorMember(ma,mae,n)
                _t = o.target
                _i = o.input

    def test_add(self):
        ma,mb,mae,mbe = np.random.rand(4,*self.shape)
        a = OperatorMember(ma,mae,"S.S")
        b = OperatorMember(mb,mbe,"S.S")

        # plain sum and diff
        for c,d in [(a + b,a - b),(b+a, -(b-a))]:
            assert_almost_equal(c.value, ma+mb)
            assert_almost_equal(c.error, mae+mbe)
            assert c.name == "S.S"
            assert_almost_equal(d.value, ma-mb)
            assert_almost_equal(d.error, mae+mbe)
            assert d.name == "S.S"

        # add to 0
        for c in [0 + a, a + 0, a - 0, 0 - a]:
            assert_almost_equal(c.value, ma)
            assert_almost_equal(c.error, mae)
            assert c.name == "S.S"

        # errors:
        # non-matching name
        with pytest.raises(ValueError):
            _ = a + OperatorMember(mb,mbe,"S.g")
        # wrong other
        with pytest.raises(ValueError):
            _ = 1 + OperatorMember(mb,mbe,"S.g")
        with pytest.raises(ValueError):
            _ = OperatorMember(mb,mbe,"S.g") + 1
        with pytest.raises(NotImplementedError):
            _ = [] + OperatorMember(mb,mbe,"S.g")
        with pytest.raises(NotImplementedError):
            _ = OperatorMember(mb,mbe,"S.g") + []

    def test_neg(self):
        ma,mae = np.random.rand(2,*self.shape)
        a = OperatorMember(ma,mae,"S.S")
        b = -a
        assert_almost_equal(b.value, -ma)
        assert_almost_equal(b.error, mae)
        assert b.name == a.name
        # is inverse?
        c = a + b
        assert_almost_equal(c.value, np.zeros(self.shape))

    def test_add_relatives(self):
        ma,mb,mae,mbe = np.random.rand(4,*self.shape)
        a = OperatorMember(ma,mae,"S.S")
        b = OperatorMember(mb,mbe,"S.S")
        # abstract sum
        c = sum([a,b])
        assert_almost_equal(c.value, ma+mb)
        assert_almost_equal(c.error, mae+mbe)
        # beware of the error computation here below ...
        # +=
        a += b
        assert_almost_equal(a.value, ma+mb)
        assert_almost_equal(a.error, mae+mbe)

        # -=
        a -= b
        assert_almost_equal(a.value, ma)
        assert_almost_equal(a.error, mae+2.*mbe)

    def test_eq(self):
        ma,mb,mae,mbe = np.random.rand(4,*self.shape)
        a = OperatorMember(ma,mae,"S.S")
        b = OperatorMember(mb,mbe,"S.S")
        c = OperatorMember(ma,mbe,"S.S")

        assert a != b
        assert a == c

    def test_mul(self):
        ma,mb,mae,mbe = np.random.rand(4,*self.shape)
        a = OperatorMember(ma,mae,"S.g")
        b = OperatorMember(mb,mbe,"g.S")
        # plain product
        c = a*b
        assert_almost_equal(c.value, ma@mb)
        assert_almost_equal(c.error, np.abs(mae@mb) + np.abs(ma@mbe))
        d = b*a
        assert_almost_equal(d.value, mb@ma)
        assert_almost_equal(d.error, np.abs(mbe@ma) + np.abs(mb@mae))
        assert c != d

        # errors
        # div is useless
        with pytest.raises(TypeError):
            _ = c/d
        # wrong name
        with pytest.raises(ValueError):
            _ = a *  OperatorMember(mb,mbe,"S.S")
        # wrong other
        with pytest.raises(NotImplementedError):
            _ = c*[]
        with pytest.raises(NotImplementedError):
            _ = []*c

    def test_call(self):
        ma,mae = np.random.rand(2,*self.shape)
        a = OperatorMember(ma,mae,"S.g")
        pdf = np.random.rand(self.shape[0])
        prod = a(pdf)
        assert len(prod) == 2
        assert_almost_equal(prod[0], ma@pdf)
        assert_almost_equal(prod[1], mae@pdf)

    def test_join_1(self):
        # two steps
        ma,mb,mae,mbe = np.random.rand(4,*self.shape)
        a = OperatorMember(ma,mae,"V.V")
        b = OperatorMember(mb,mbe,"V.V")
        steps = [{a.name:a},{b.name:b}]
        paths = [["V.V","V.V"]]
        j2 = OperatorMember.join(steps,paths, "V.V")
        assert j2.name == "V.V"
        assert j2 == a*b
        # three steps
        mc,mce = np.random.rand(2,*self.shape)
        c = OperatorMember(mc,mce,"V.V")
        steps = [{a.name:a},{b.name:b},{c.name:c}]
        paths = [["V.V","V.V","V.V"]]
        j3 = OperatorMember.join(steps,paths, "V.V")
        assert j3.name == "V.V"
        assert j3 == (a*b)*c
        assert j3 == a*(b*c)
        assert j3 == j2*c
