# -*- coding: utf-8 -*-

import numpy as np
import pytest

from eko import member
from eko.operator.physical import PhysicalOperator
from eko.operator import flavors


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return member.OpMember(ma, mae)


class TestPhysicalOperator:
    shape = (2, 2)

    def _mkOM(self, n):
        return [mkOM(self.shape) for j in range(n)]

    def _mkNames(self, ns):
        return [member.MemberName(n) for n in ns]

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
        assert c.op_members[member.MemberName("V.V")] == VVh @ VVl
        assert c.op_members[member.MemberName("T3.S")] == T3T3h @ T3Sl
        assert c.op_members[member.MemberName("T3.g")] == T3T3h @ T3gl
        assert c.op_members[member.MemberName("S.S")] == SSh @ SSl + Sgh @ gSl
        # errors
        with pytest.raises(ValueError):
            _ = a @ {}

    def test_to_flavor_basis_tensor_ss(self):
        (SS,) = self._mkOM(1)
        a = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("S.S",)),
                    (SS,),
                )
            ),
            1,
        )
        vt, _ = a.to_flavor_basis_tensor()
        np.testing.assert_allclose(vt[6, :, 6, :], vt[6, :, 5, :])
        np.testing.assert_allclose(vt[6, :, 6, :], vt[5, :, 6, :])
        np.testing.assert_allclose(vt[6, :, 6, :], SS.value[:, :] / (2 * 3))
        np.testing.assert_allclose(vt[1, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 1, :], 0)
        np.testing.assert_allclose(vt[7, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 7, :], 0)

    def test_to_flavor_basis_tensor_gg(self):
        (gg,) = self._mkOM(1)
        a = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("g.g",)),
                    (gg,),
                )
            ),
            1,
        )
        vt, _ = a.to_flavor_basis_tensor()
        np.testing.assert_allclose(vt[6, :, 6, :], 0)
        np.testing.assert_allclose(vt[7, :, 7, :], gg.value[:, :])
        np.testing.assert_allclose(vt[1, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 1, :], 0)
        np.testing.assert_allclose(vt[7, :, :7, :], 0)
        np.testing.assert_allclose(vt[8:, :, 7, :], 0)


def mk_op_members(shape=(2, 2)):
    m = np.random.rand(len(member.full_labels), *shape)
    e = np.random.rand(len(member.full_labels), *shape)
    om = {}
    for j, l in enumerate(member.full_labels):
        om[l] = member.OpMember(m[j], e[j])
    return om


def get_ad_to_evol_map(nf, intrinsic_range=None):
    oms = mk_op_members()
    m = PhysicalOperator.ad_to_evol_map(oms, nf, 1, intrinsic_range)
    return sorted(map(str, m.op_members.keys()))


def test_ad_to_evol_map():
    triv_ops = ("S.S", "S.g", "g.S", "g.g", "V.V", "V3.V3", "T3.T3", "V8.V8", "T8.T8")
    # nf=3
    assert sorted([*triv_ops, "V15.V", "T15.S", "T15.g"]) == get_ad_to_evol_map(3)
    # nf=3 + IC
    assert sorted(
        [*triv_ops, "V15.V", "T15.S", "T15.g", "T15.c+", "V15.c-"]
    ) == get_ad_to_evol_map(3, [4])
    # nf=3 + IC + IB
    assert sorted(
        [*triv_ops, "V15.V", "T15.S", "T15.g", "T15.c+", "V15.c-", "b+.b+", "b-.b-"]
    ) == get_ad_to_evol_map(3, [4, 5])
    # nf=4 + IC + IB
    ks = sorted(
        [*triv_ops, "V15.V15", "T15.T15", "V24.V", "T24.S", "T24.g", "T24.b+", "V24.b-"]
    )
    assert ks == get_ad_to_evol_map(4, [4, 5])
    # nf=4 + IB
    assert ks == get_ad_to_evol_map(4, [5])
    # nf=6
    assert sorted(
        [*triv_ops, "T15.T15", "V15.V15", "T24.T24", "V24.V24", "T35.T35", "V35.V35"]
    ) == get_ad_to_evol_map(6)
