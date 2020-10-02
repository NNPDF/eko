# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.operator.member import OpMember
from eko.operator.physical import PhysicalOperator


def mkOM(name, shape):
    ma, mae = np.random.rand(2, *shape)
    return OpMember(ma, mae, name), ma, mae


class TestPhysicalOperator:
    shape = (2, 2)

    def _mkOM(self, name):
        return mkOM(name, self.shape)

    def _dict(self, ops):
        return {op.name: op for op in ops}

    def test_call(self):
        VV, mVV, mVVe = self._mkOM("V.V")
        V3V3, _, _ = self._mkOM("V3.V3")
        Tg, mTg, mTge = self._mkOM("T24.g")
        TS, mTS, mTSe = self._mkOM("T24.S")
        a = PhysicalOperator(self._dict([VV, V3V3, Tg, TS]), 1)
        # create pdfs
        V, V3, S, g = np.random.rand(4, self.shape[0])
        # check
        target, errs = a.apply_pdf({"V": V, "g": g, "S": S})
        # without providing V3 it will not be computed
        assert list(target.keys()) == ["V", "T24"]
        assert_almost_equal(target["V"], mVV @ V)
        assert_almost_equal(errs["V"], mVVe @ V)
        assert_almost_equal(target["T24"], mTS @ S + mTg @ g)
        assert_almost_equal(errs["T24"], mTSe @ S + mTge @ g)
        # whithout gluon we cannot compute T24
        target, errs = a.apply_pdf({"V": V, "V3": V3, "S": S})
        assert list(target.keys()) == ["V", "V3"]

    def test_get_raw_operators(self):
        VV, _, _ = self._mkOM("V.V")
        V3V3, _, _ = self._mkOM("V3.V3")
        a = PhysicalOperator(self._dict([VV, V3V3]), 1)
        raw = a.get_raw_operators()
        for op_or_err in ["operators", "operator_errors"]:
            for op in raw[op_or_err].values():
                # check list(list(float))
                assert isinstance(op, list)
                for inner in op:
                    assert isinstance(inner, list)

    def test_mul(self):
        VVl, _, _ = self._mkOM("V.V")
        V3V3l, _, _ = self._mkOM("V3.V3")
        T3Sl, _, _ = self._mkOM("T3.S")
        T3gl, _, _ = self._mkOM("T3.g")
        SSl, _, _ = self._mkOM("S.S")
        gSl, _, _ = self._mkOM("g.S")
        a = PhysicalOperator(self._dict([VVl, V3V3l, T3gl, T3Sl, SSl, gSl]), 1)
        VVh, _, _ = self._mkOM("V.V")
        V8V8h, _, _ = self._mkOM("V8.V8")
        T3T3h, _, _ = self._mkOM("T3.T3")
        SSh, _, _ = self._mkOM("S.S")
        Sgh, _, _ = self._mkOM("S.g")
        b = PhysicalOperator(self._dict([VVh, V8V8h, T3T3h, SSh, Sgh]), 2)
        c = b * a
        assert c.q2_final == b.q2_final
        # only V and T3 can be computed
        assert list(c.op_members.keys()) == ["V.V", "T3.g", "T3.S", "S.S"]
        assert c.op_members["V.V"] == VVh * VVl
        assert c.op_members["T3.S"] == T3T3h * T3Sl
        assert c.op_members["T3.g"] == T3T3h * T3gl
        assert c.op_members["S.S"] == SSh * SSl + Sgh * gSl
        # errors
        with pytest.raises(ValueError):
            _ = a * {}
