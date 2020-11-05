# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.operator.member import OpMember
from eko.operator.physical import PhysicalOperator
from eko.operator import flavors


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return OpMember(ma, mae), ma, mae


class TestPhysicalOperator:
    shape = (2, 2)

    def _mkOM(self, name):
        return mkOM(self.shape)

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
        raw = a.to_raw()
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


def mk_op_members(shape=(2, 2)):
    m = np.random.rand(7, *shape)
    om = {}
    for j, l in enumerate(flavors.full_labels):
        om[l] = m[j]
    return om


def get_ad_to_evol_map(nf, is_vfns, intrinsic_range=None):
    oms = mk_op_members()
    m = PhysicalOperator.ad_to_evol_map(oms, nf, 1, is_vfns, intrinsic_range)
    return sorted(map(str, m.op_members.keys()))


def test_ad_to_evol_map_ffns():
    triv_ops = ("S.S", "S.g", "g.S", "g.g", "V.V", "V3.V3", "T3.T3", "V8.V8", "T8.T8")
    # FFNS3
    assert sorted(triv_ops) == get_ad_to_evol_map(3, False)
    # FFNS3 + IC
    assert sorted([*triv_ops, "c+.c+", "c-.c-"]) == get_ad_to_evol_map(3, False, [4])
    # FFNS3 + IC + IB
    assert sorted(
        [*triv_ops, "c+.c+", "c-.c-", "b+.b+", "b-.b-"]
    ) == get_ad_to_evol_map(3, False, [4, 5])
    # FFNS4 + IC + IB
    with pytest.raises(ValueError):
        get_ad_to_evol_map(4, False, [4, 5])
    # FFNS4 + IB
    assert sorted(
        [*triv_ops, "V15.V15", "T15.T15", "b+.b+", "b-.b-"]
    ) == get_ad_to_evol_map(4, False, [5])
    # FFNS6
    ks = get_ad_to_evol_map(6, False)
    assert len(ks) == 4 + 1 + 2 * 5


def test_ad_to_evol_map_vfns():
    triv_ops = ("S.S", "S.g", "g.S", "g.g", "V.V", "V3.V3", "T3.T3", "V8.V8", "T8.T8")
    # VFNS, nf=3 patch
    assert sorted([*triv_ops, "V15.V", "T15.S", "T15.g"]) == get_ad_to_evol_map(3, True)
    # VFNS, nf=3 patch + IC
    assert sorted(
        [*triv_ops, "V15.V", "T15.S", "T15.g", "T15.c+", "V15.c-"]
    ) == get_ad_to_evol_map(3, True, [4])
    # VFNS, nf=3 patch + IC + IB
    assert sorted(
        [*triv_ops, "V15.V", "T15.S", "T15.g", "T15.c+", "V15.c-", "b+.b+", "b-.b-"]
    ) == get_ad_to_evol_map(3, True, [4, 5])
    # VFNS, nf=4 patch + IC + IB
    ks = sorted(
        [*triv_ops, "V15.V15", "T15.T15", "V24.V", "T24.S", "T24.g", "T24.b+", "V24.b-"]
    )
    assert ks == get_ad_to_evol_map(4, True, [4, 5])
    # VFNS, nf=4 patch + IB
    assert ks == get_ad_to_evol_map(4, True, [5])
