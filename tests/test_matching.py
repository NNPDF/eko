# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

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
        return dict(NS=ns, S_qq=qq, S_qg=qg, S_gq=gq, S_gg=gg)

    def test_split_ad_to_evol_map(self):
        ome = self.mkOME()
        a = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, 1)
        keys3 = [
            "V.V",
            "T3.T3",
            "V3.V3",
            "T8.T8",
            "V8.V8",
            "S.S",
            "S.g",
            "g.S",
            "g.g",
            "T15.S",
            "T15.g",
            "V15.V",
        ]
        assert sorted([str(k) for k in a.op_members.keys()]) == sorted(keys3)
        assert_almost_equal(
            a.op_members[member.MemberName("V.V")].value,
            np.eye(self.shape[0]) + ome["NS"].value,
        )
        # if alpha is zero, nothing should happen
        b = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, 0)
        assert sorted([str(k) for k in b.op_members.keys()]) == sorted(keys3)
        assert_almost_equal(
            b.op_members[member.MemberName("V.V")].value,
            np.eye(self.shape[0]),
        )
