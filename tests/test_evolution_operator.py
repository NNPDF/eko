# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.evolution_operator import (
    OperatorMember,
    PhysicalOperator,
    Operator,
    _get_kernel_integrands,
)
from eko import thresholds


def mkOM(name, shape):
    ma, mae = np.random.rand(2, *shape)
    return OperatorMember(ma, mae, name), ma, mae


class TestOperatorMember:
    shape = (2, 2)

    def _mkOM(self, name):
        return mkOM(name, self.shape)

    def test_name(self):
        ma, mae = np.random.rand(2, *self.shape)
        for i in ["S", "g"]:
            for t in ["S", "g"]:
                n = f"{t}.{i}"
                a = OperatorMember(ma, mae, n)
                # splitting
                assert a.input == i
                assert a.target == t
                # __str__
                assert str(a) == n
        # check wrongs
        for n in [".", "a.", ".b", "ab"]:
            with pytest.raises(ValueError):
                a = OperatorMember(ma, mae, n)
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
            _ = a + OperatorMember(mb, mbe, "S.g")
        # non-phyiscal
        with pytest.raises(ValueError):
            _ = a + OperatorMember(mb, mbe, "NS_v")
        # wrong other
        with pytest.raises(ValueError):
            _ = 1 + OperatorMember(mb, mbe, "S.g")
        with pytest.raises(ValueError):
            _ = OperatorMember(mb, mbe, "S.g") + 1
        with pytest.raises(NotImplementedError):
            _ = [] + OperatorMember(mb, mbe, "S.g")
        with pytest.raises(NotImplementedError):
            _ = OperatorMember(mb, mbe, "S.g") + []

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
        c = OperatorMember(ma, mbe, "S.S")

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
            _ = a * OperatorMember(mb, mbe, "NS_v")
        # wrong name
        with pytest.raises(ValueError):
            _ = a * OperatorMember(mb, mbe, "S.S")
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
        j2 = OperatorMember.join(steps, paths)
        assert j2.name == "V.V"
        assert j2 == a * b
        # three steps
        c, _, _ = self._mkOM("V.V")
        steps = [{a.name: a}, {b.name: b}, {c.name: c}]
        paths = [["V.V", "V.V", "V.V"]]
        j3 = OperatorMember.join(steps, paths)
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
        j2 = OperatorMember.join(steps, paths)
        assert j2.name == "S.S"
        assert j2 == ss0 * ss1 + sg0 * gs1


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
        target, errs = a({"V": V, "g": g, "S": S})
        # without providing V3 it will not be computed
        assert list(target.keys()) == ["V", "T24"]
        assert_almost_equal(target["V"], mVV @ V)
        assert_almost_equal(errs["V"], mVVe @ V)
        assert_almost_equal(target["T24"], mTS @ S + mTg @ g)
        assert_almost_equal(errs["T24"], mTSe @ S + mTge @ g)
        # whithout gluon we cannot compute T24
        target, errs = a({"V": V, "V3": V3, "S": S})
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


# int_0.5^1 dz z^k = (1 - 0.5**(k+1))/(k+1)
def get_ker(k):
    def ker(z, args, k=k):
        lnx = args[0]
        return lnx * z ** k

    return ker


def get_res(k, xg, cut=0):
    return [[x * ((1 - cut) ** (k + 1) - 0.5 ** (k + 1)) / (k + 1)] for x in np.log(xg)]


def test_get_kernel_integrands():
    xg = [np.exp(-1), 0.9]

    ints_s, ints_ns = _get_kernel_integrands(
        [[get_ker(2), get_ker(3), get_ker(4), get_ker(5)]],
        [get_ker(1)],
        0.5,
        1.0,
        xg,
        cut=0,
    )
    op_ns = ints_ns()
    assert "NS_v" in op_ns
    assert "NS_p" in op_ns
    assert "NS_m" in op_ns
    assert op_ns["NS_v"].value.shape == (len(xg), 1)
    assert_almost_equal(op_ns["NS_v"].value, get_res(1, xg))
    ops_s = ints_s()
    assert_almost_equal(ops_s["S_qq"].value, get_res(2, xg))
    assert_almost_equal(ops_s["S_qg"].value, get_res(3, xg))
    assert_almost_equal(ops_s["S_gq"].value, get_res(4, xg))
    assert_almost_equal(ops_s["S_gg"].value, get_res(5, xg))


class TestOperator:
    def test_meta(self):
        xg = [0.5, 1.0]
        meta = dict(nf=3, q2ref=1, q2=2)
        op = Operator(0.5, 1.0, xg, [], [], meta)
        assert op.nf == meta["nf"]
        assert op.q2ref == meta["q2ref"]
        assert op.q2 == meta["q2"]
        assert_almost_equal(op.xgrid, xg)

    def test_compose(self):
        xg = [np.exp(-1), 0.9]
        meta = dict(nf=3, q2ref=1, q2=2)
        op1 = Operator(
            0.5,
            1.0,
            xg,
            [get_ker(1)],
            [[get_ker(2), get_ker(3), get_ker(4), get_ker(5)]],
            meta,
            0,
        )
        # FFNS
        t = thresholds.ThresholdsConfig(1, "FFNS", nf=3)
        instruction_set = t.get_composition_path(3, 0)
        ph = op1.compose([], instruction_set, 2)
        assert isinstance(ph, PhysicalOperator)
        # V.V is NS_v
        assert_almost_equal(ph.op_members["V.V"].value, get_res(1, xg))
