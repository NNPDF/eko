# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal

from eko import member
from eko.matching_conditions import MatchingCondition


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return member.OpMember(ma, mae)


class TestMatchingCondition:
    shape = (2, 2)

    def mkOME(self):
        ns = mkOM(self.shape)
        qq = mkOM(self.shape)
        qg = mkOM(self.shape)
        gq = mkOM(self.shape)
        gg = mkOM(self.shape)
        return dict(NS_qq=ns, S_qq=qq, S_qg=qg, S_gq=gq, S_gg=gg)

    def test_split_ad_to_evol_map(self):
        ome = self.mkOME()
        a = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, 1)
        triv_keys = [
            "V.V",
            "T3.T3",
            "V3.V3",
            "T8.T8",
            "V8.V8",
            "S.S",
            "S.g",
            "g.S",
            "g.g",
        ]
        # nf = 3
        keys3 = [
            "T15.S",
            "T15.g",
            "V15.V",
        ]
        assert sorted([str(k) for k in a.op_members.keys()]) == sorted(
            [*triv_keys, *keys3]
        )
        assert_almost_equal(
            a.op_members[member.MemberName("V.V")].value,
            np.eye(self.shape[0]) + ome["NS_qq"].value,
        )
        # if alpha is zero, nothing non-trivial should happen
        b = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, 0)
        assert sorted([str(k) for k in b.op_members.keys()]) == sorted(
            [*triv_keys, *keys3]
        )
        assert_almost_equal(
            b.op_members[member.MemberName("V.V")].value,
            np.eye(self.shape[0]),
        )
        # nf=3 + IC
        c = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, 0, [4])
        assert sorted([str(k) for k in c.op_members.keys()]) == sorted(
            [*triv_keys, *keys3, "S.c+", "V.c-", "T15.c+", "V15.c-"]
        )
        assert_almost_equal(
            c.op_members[member.MemberName("V.V")].value,
            b.op_members[member.MemberName("V.V")].value,
        )
        assert_almost_equal(
            c.op_members[member.MemberName("T15.c+")].value,
            -3.0 * np.eye(self.shape[0]),
        )
        assert_almost_equal(
            c.op_members[member.MemberName("V15.c-")].value,
            -3.0 * np.eye(self.shape[0]),
        )
        # nf=3 + IB
        d = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, 0, [5])
        assert sorted([str(k) for k in d.op_members.keys()]) == sorted(
            [*triv_keys, *keys3, "b+.b+", "b-.b-"]
        )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.b+")].value,
            np.eye(self.shape[0]),
        )
        # nf=4 + IB
        d = MatchingCondition.split_ad_to_evol_map(ome, 4, 1, 1, [5])
        assert sorted([str(k) for k in d.op_members.keys()]) == sorted(
            [
                *triv_keys,
                "T15.T15",
                "V15.V15",
                "T24.S",
                "T24.g",
                "V24.V",
                "T24.b+",
                "V24.b-",
                "S.b+",
                "V.b-",
            ]
        )
        assert_almost_equal(
            d.op_members[member.MemberName("V.V")].value,
            a.op_members[member.MemberName("V.V")].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("T24.b+")].value,
            -4.0 * np.eye(self.shape[0]),
        )
        assert_almost_equal(
            d.op_members[member.MemberName("V24.b-")].value,
            -4.0 * np.eye(self.shape[0]),
        )
        assert_almost_equal(
            d.op_members[member.MemberName("V24.V")].value,
            np.eye(self.shape[0]) + ome["NS_qq"].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("T24.S")].value,
            np.eye(self.shape[0]) + ome["NS_qq"].value - 4.0 * ome["S_qq"].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("T24.g")].value,
            -4.0 * ome["S_qg"].value,
        )
