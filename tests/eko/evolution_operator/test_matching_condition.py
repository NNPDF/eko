import numpy as np
from numpy.testing import assert_almost_equal

from eko import basis_rotation as br
from eko import member
from eko.evolution_operator.matching_condition import MatchingCondition


def mkOM(shape):
    ma, mae = np.random.rand(2, *shape)
    return member.OpMember(ma, mae)


class TestMatchingCondition:
    shape = (2, 2)

    def mkOME(self):
        ome = {}
        for key in [
            *br.singlet_labels,
            (br.matching_hplus_pid, 100),
            (br.matching_hplus_pid, 21),
            (200, 200),
            (br.matching_hminus_pid, 200),
            (br.matching_hplus_pid, br.matching_hplus_pid),
            (br.matching_hminus_pid, br.matching_hminus_pid),
            (200, br.matching_hminus_pid),
            (100, br.matching_hplus_pid),
            (21, br.matching_hplus_pid),
        ]:
            ome.update({key: mkOM(self.shape)})
        return ome

    def test_split_ad_to_evol_map(self):
        ome = self.mkOME()
        a = MatchingCondition.split_ad_to_evol_map(ome, 3, 1)
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
            "S.c+",
            "g.c+",
            "c+.c+",
            "c-.c-",
            "b+.b+",
            "b-.b-",
            "t+.t+",
            "t-.t-",
        ]
        assert sorted(str(k) for k in a.op_members.keys()) == sorted(
            [*triv_keys, *keys3]
        )
        assert_almost_equal(
            a.op_members[member.MemberName("V.V")].value,
            ome[(200, 200)].value,
        )
        # nf=4 + IB
        d = MatchingCondition.split_ad_to_evol_map(ome, 4, 1)
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
                "t+.t+",
                "t-.t-",
            ]
        )
        assert_almost_equal(
            d.op_members[member.MemberName("V.V")].value,
            a.op_members[member.MemberName("V.V")].value,
        )
        # assert_almost_equal(
        #     d.op_members[member.MemberName("b-.V")].value,
        #     ome[(br.matching_hminus_pid, 200)].value,
        # )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.S")].value,
            ome[(br.matching_hplus_pid, 100)].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.g")].value,
            ome[(br.matching_hplus_pid, 21)].value,
        )

    def test_split_ad_to_evol_map_qed(self):
        ome = self.mkOME()
        a = MatchingCondition.split_ad_to_evol_map(ome, 3, 1, qed=True)
        triv_keys = [
            "ph.ph",
            "S.S",
            "S.g",
            "g.S",
            "g.g",
            "Sdelta.Sdelta",
            "V.V",
            "Vdelta.Vdelta",
            "Td3.Td3",
            "Vd3.Vd3",
        ]
        # nf = 3
        keys3 = [
            "c+.S",
            "c+.g",
            "S.c+",
            "g.c+",
            "c+.c+",
            "c-.c-",
            # "c-.V",
            "b+.b+",
            "b-.b-",
            "t+.t+",
            "t-.t-",
        ]
        assert sorted(str(k) for k in a.op_members.keys()) == sorted(
            [*triv_keys, *keys3]
        )
        assert_almost_equal(
            a.op_members[member.MemberName("V.V")].value,
            ome[(200, 200)].value,
        )
        # nf=4 + IB
        d = MatchingCondition.split_ad_to_evol_map(ome, 4, 1, qed=True)
        assert sorted(str(k) for k in d.op_members.keys()) == sorted(
            [
                *triv_keys,
                "Tu3.Tu3",
                "Vu3.Vu3",
                "S.b+",
                "g.b+",
                # "V.b-",
                "b+.S",
                "b+.g",
                "b+.b+",
                # "b-.V",
                "b-.b-",
                "t+.t+",
                "t-.t-",
            ]
        )
        assert_almost_equal(
            d.op_members[member.MemberName("V.V")].value,
            a.op_members[member.MemberName("V.V")].value,
        )
        # assert_almost_equal(
        #     d.op_members[member.MemberName("b-.V")].value,
        #     ome[(br.matching_hminus_pid, 200)].value,
        # )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.S")].value,
            ome[(br.matching_hplus_pid, 100)].value,
        )
        assert_almost_equal(
            d.op_members[member.MemberName("b+.g")].value,
            ome[(br.matching_hplus_pid, 21)].value,
        )
