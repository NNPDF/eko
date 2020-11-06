# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from eko.operator.member import OpMember
from eko.operator.physical import PhysicalOperator
from eko.operator import flavors


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return OpMember(ma, mae)


class TestPhysicalOperator:
    shape = (2, 2)

    def _mkOM(self, n):
        return [mkOM(self.shape) for j in range(n)]

    def _mkNames(self, ns):
        return [flavors.MemberName(n) for n in ns]

    def test_matmul(self):
        VVl, V3V3l, T3Sl, T3gl, SSl, gSl = self._mkOM(6)
        a = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("V.V", "V3.V3", "T3.S", "T3.g", "S.S", "g.S")),
                    (VVl, V3V3l, T3Sl, T3gl, SSl, gSl),
                )
            ),
            1,
        )
        VVh, V8V8h, T3T3h, SSh, Sgh = self._mkOM(5)
        b = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("V.V", "V8.V8", "T3.T3", "S.S", "S.g")),
                    (VVh, V8V8h, T3T3h, SSh, Sgh),
                )
            ),
            2,
        )
        c = b @ a
        assert c.q2_final == b.q2_final
        # only V and T3 can be computed
        assert sorted([str(k) for k in c.op_members.keys()]) == sorted(
            ["V.V", "T3.g", "T3.S", "S.S"]
        )
        assert c.op_members[flavors.MemberName("V.V")] == VVh @ VVl
        assert c.op_members[flavors.MemberName("T3.S")] == T3T3h @ T3Sl
        assert c.op_members[flavors.MemberName("T3.g")] == T3T3h @ T3gl
        assert c.op_members[flavors.MemberName("S.S")] == SSh @ SSl + Sgh @ gSl
        # errors
        with pytest.raises(ValueError):
            _ = a @ {}


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
