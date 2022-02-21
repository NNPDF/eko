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
        ome = {}
        for key in ["qq", "qg", "gq", "gg", "Hq", "Hg"]:
            ome.update({f"S_{key}": mkOM(self.shape)})
            if "g" not in key:
                ome.update({f"NS_{key}": mkOM(self.shape)})
        return ome

    def update_intrinsic_OME(self, ome):
        for key in ["HH", "qH", "gH"]:
            ome.update({f"S_{key}": mkOM(self.shape)})
            if "g" not in key:
                ome.update({f"NS_{key}": mkOM(self.shape)})

    def test_split_ad_to_evol_map(self):
        ome = self.mkOME()
        a = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, [])
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
            "c+.S",
            "c+.g",
            # "c-.V",
        ]
        assert sorted(str(k) for k in a.op_members.keys()) == sorted(
            [*triv_keys, *keys3]
        )
        assert_almost_equal(
            a.op_members[member.MemberName("V.V")].value,
            ome["NS_qq"].value,
        )
        # # if alpha is zero, nothing non-trivial should happen
        b = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, [])
        assert sorted(str(k) for k in b.op_members.keys()) == sorted(
            [*triv_keys, *keys3]
        )
        # assert_almost_equal(
        #     b.op_members[member.MemberName("V.V")].value,
        #     np.eye(self.shape[0]),
        # )
        # nf=3 + IC
        self.update_intrinsic_OME(ome)
        c = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, [4])
        assert sorted(str(k) for k in c.op_members.keys()) == sorted(
            [*triv_keys, *keys3, "S.c+", "g.c+", "c+.c+", "c-.c-"]
        )
        assert_almost_equal(
            c.op_members[member.MemberName("V.V")].value,
            b.op_members[member.MemberName("V.V")].value,
        )
        # nf=3 + IB
        d = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, [5])
        assert sorted(str(k) for k in d.op_members.keys()) == sorted(
            [*triv_keys, *keys3, "b+.b+", "b-.b-"]
        )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.b+")].value,
            np.eye(self.shape[0]),
        )
        # nf=4 + IB
        d = MatchingCondition.split_ad_to_evol_map(ome, 4, 1, [5])
        assert sorted(str(k) for k in d.op_members.keys()) == sorted(
            [
                *triv_keys,
                "T15.T15",
                "V15.V15",
                "S.b+",
                "g.b+",
                # "V.b-",
                "b+.S",
                "b+.g",
                "b+.b+",
                # "b-.V",
                "b-.b-",
            ]
        )
        assert_almost_equal(
            d.op_members[member.MemberName("V.V")].value,
            a.op_members[member.MemberName("V.V")].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.S")].value,
            ome["S_Hq"].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.g")].value,
            ome["S_Hg"].value,
        )
