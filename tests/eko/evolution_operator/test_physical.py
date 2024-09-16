import numpy as np
import pytest

from eko import basis_rotation as br
from eko import member
from eko.evolution_operator.matching_condition import MatchingCondition
from eko.evolution_operator.physical import PhysicalOperator


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return member.OpMember(ma, mae)


class TestPhysicalOperator:
    shape = (2, 2)

    def _mkOM(self, n):
        return [mkOM(self.shape) for j in range(n)]

    def _mkNames(self, ns):
        return [member.MemberName(n) for n in ns]

    # def test_operation(self):
    #     VV = self._mkOM(1)
    #     a = PhysicalOperator(
    #         dict(
    #             zip(
    #                 self._mkNames(("V.V",)),
    #                 (VV),
    #             )
    #         ),
    #         1,
    #     )
    #     assert "__matmul__" == PhysicalOperator.operation(a)
    #     b = member.ScalarOperator(dict(zip(self._mkNames(("V.V",)), (1.0,))), 1)
    #     assert "__mul__" == PhysicalOperator.operation(b)

    def test_matmul_scalar(self):
        VV = self._mkOM(1)
        a = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("V.V",)),
                    (VV),
                )
            ),
            1,
        )
        n = 2.0
        b = member.ScalarOperator(dict(zip(self._mkNames(("V.V",)), (n,))), 1)
        c = a @ b
        assert c.q2_final == b.q2_final
        assert VV[0] * n == c.op_members[member.MemberName("V.V")]
        d = b @ a
        assert d.q2_final == b.q2_final
        assert VV[0] * n == d.op_members[member.MemberName("V.V")]
        e = b @ b
        assert e.q2_final == b.q2_final
        assert n * n == e.op_members[member.MemberName("V.V")]

    def test_matmul(self):
        VVl, V3V3l, T3T3l, SSl, gSl = self._mkOM(5)
        a = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("V.V", "V3.V3", "T3.T3", "S.S", "g.S")),
                    (VVl, V3V3l, T3T3l, SSl, gSl),
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
        # V, T3 and S can be computed
        assert sorted(str(k) for k in c.op_members.keys()) == sorted(
            ["V.V", "T3.T3", "S.S"]
        )
        assert c.op_members[member.MemberName("V.V")] == VVh @ VVl
        assert c.op_members[member.MemberName("T3.T3")] == T3T3h @ T3T3l
        assert c.op_members[member.MemberName("S.S")] == SSh @ SSl + Sgh @ gSl
        T3S, T3g = self._mkOM(2)
        mc = MatchingCondition(
            dict(
                zip(
                    self._mkNames(("T3.S", "T3.g")),
                    (T3S, T3g),
                )
            ),
            1,
        )
        ap = a = PhysicalOperator(
            dict(
                zip(
                    self._mkNames(("V.V", "V3.V3", "S.S", "g.S")),
                    (VVl, V3V3l, SSl, gSl),
                )
            ),
            1,
        )
        # check matching conditions
        d = b @ mc @ ap
        assert sorted(str(k) for k in d.op_members.keys()) == sorted(["T3.S"])
        assert (
            d.op_members[member.MemberName("T3.S")]
            == T3T3h @ T3S @ SSl + T3T3h @ T3g @ gSl
        )
        dd = b @ (mc @ ap)
        assert sorted(str(k) for k in dd.op_members.keys()) == sorted(["T3.S"])
        assert (
            d.op_members[member.MemberName("T3.S")]
            == dd.op_members[member.MemberName("T3.S")]
        )
        # errors
        with pytest.raises(ValueError):
            _ = a @ {}
        with pytest.raises(TypeError):
            _ = {} @ a

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
        vt, _ = a.to_flavor_basis_tensor(False)
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
        vt, _ = a.to_flavor_basis_tensor(False)
        np.testing.assert_allclose(vt[6, :, 6, :], 0)
        np.testing.assert_allclose(vt[7, :, 7, :], gg.value[:, :])
        np.testing.assert_allclose(vt[1, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 1, :], 0)
        np.testing.assert_allclose(vt[7, :, :7, :], 0)
        np.testing.assert_allclose(vt[8:, :, 7, :], 0)

    shape = (4, 4)

    def test_to_flavor_basis_tensor_ss_qed(self):
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
        vt, _ = a.to_flavor_basis_tensor(qed=True)
        np.testing.assert_allclose(vt[6, :, 6, :], vt[6, :, 5, :])
        np.testing.assert_allclose(vt[6, :, 6, :], vt[5, :, 6, :])
        np.testing.assert_allclose(vt[6, :, 6, :], SS.value[:, :] / (2 * 3))
        np.testing.assert_allclose(vt[1, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 1, :], 0)
        np.testing.assert_allclose(vt[7, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 7, :], 0)

    def test_to_flavor_basis_tensor_gg_qed(self):
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
        vt, _ = a.to_flavor_basis_tensor(False)
        np.testing.assert_allclose(vt[6, :, 6, :], 0)
        np.testing.assert_allclose(vt[7, :, 7, :], gg.value[:, :])
        np.testing.assert_allclose(vt[1, :, :, :], 0)
        np.testing.assert_allclose(vt[:, :, 1, :], 0)
        np.testing.assert_allclose(vt[7, :, :7, :], 0)
        np.testing.assert_allclose(vt[8:, :, 7, :], 0)


def mk_op_members(shape=(2, 2), qed=False):
    if not qed:
        full_labels = br.full_labels
    else:
        full_labels = br.full_unified_labels
    m = np.random.rand(len(full_labels), *shape)
    e = np.random.rand(len(full_labels), *shape)
    om = {}
    for j, lab in enumerate(full_labels):
        om[lab] = member.OpMember(m[j], e[j])
    return om


def get_ad_to_evol_map(nf, qed=False):
    oms = mk_op_members(qed=qed)
    m = PhysicalOperator.ad_to_evol_map(oms, nf, 1, qed)
    return sorted(map(str, m.op_members.keys()))


def test_ad_to_evol_map():
    triv_ops = ("S.S", "S.g", "g.S", "g.g", "V.V", "V3.V3", "T3.T3", "V8.V8", "T8.T8")
    # nf=3
    assert sorted(
        [*triv_ops, "c+.c+", "c-.c-", "b+.b+", "b-.b-", "t+.t+", "t-.t-"]
    ) == get_ad_to_evol_map(3)
    # nf=4
    ks = sorted([*triv_ops, "V15.V15", "T15.T15", "b+.b+", "b-.b-", "t+.t+", "t-.t-"])
    assert ks == get_ad_to_evol_map(4)
    # nf=6
    assert sorted(
        [*triv_ops, "T15.T15", "V15.V15", "T24.T24", "V24.V24", "T35.T35", "V35.V35"]
    ) == get_ad_to_evol_map(6)


def test_ad_to_evol_map_qed():
    triv_ops = (
        "g.g",
        "g.ph",
        "g.S",
        "g.Sdelta",
        "ph.g",
        "ph.ph",
        "ph.S",
        "ph.Sdelta",
        "S.g",
        "S.ph",
        "S.S",
        "S.Sdelta",
        "Sdelta.g",
        "Sdelta.ph",
        "Sdelta.S",
        "Sdelta.Sdelta",
        "V.V",
        "V.Vdelta",
        "Vdelta.V",
        "Vdelta.Vdelta",
        "Vd3.Vd3",
        "Td3.Td3",
    )
    # nf=3
    assert sorted(
        [*triv_ops, "c+.c+", "c-.c-", "b+.b+", "b-.b-", "t+.t+", "t-.t-"]
    ) == get_ad_to_evol_map(3, True)
    # nf=4
    ks = sorted([*triv_ops, "Vu3.Vu3", "Tu3.Tu3", "b+.b+", "b-.b-", "t+.t+", "t-.t-"])
    assert ks == get_ad_to_evol_map(4, True)
    # nf=6
    assert sorted(
        [*triv_ops, "Tu3.Tu3", "Vu3.Vu3", "Td8.Td8", "Vd8.Vd8", "Tu8.Tu8", "Vu8.Vu8"]
    ) == get_ad_to_evol_map(6, qed=True)
