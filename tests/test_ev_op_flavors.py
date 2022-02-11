# -*- coding: utf-8 -*-
import numpy as np
import pytest

from eko import basis_rotation as br
from eko import member
from eko.evolution_operator import flavors


class TestOpMember:
    def test_hash(self):
        d = {member.MemberName("S.S"): 1}
        assert member.MemberName("S.S") in d
        assert member.MemberName("S.g") not in d

    def test_split(self):
        Sg = member.MemberName("S.g")
        assert Sg.input == "g"
        assert Sg.target == "S"
        # errors
        with pytest.raises(ValueError):
            _ = member.MemberName(".").input
        with pytest.raises(ValueError):
            _ = member.MemberName("bla").input


def test_pids_from_intrinsic_evol():
    def get(d):
        raw = np.zeros(len(br.flavor_basis_pids))
        for pid, w in d.items():
            raw[br.flavor_basis_pids.index(pid)] = w
        return raw

    # g
    g = get({21: 1})
    for norm in [True, False]:
        assert pytest.approx(g) == flavors.pids_from_intrinsic_evol("g", 3, norm)
        assert pytest.approx(g) == flavors.pids_from_intrinsic_evol("g", 4, norm)
    # S(nf=3) = u+ + d+ + s+
    S3 = get({1: 1, -1: 1, 2: 1, -2: 1, 3: 1, -3: 1})
    assert pytest.approx(S3) == flavors.pids_from_intrinsic_evol("S", 3, False)
    assert pytest.approx(S3 / (2 * 3)) == flavors.pids_from_intrinsic_evol("S", 3, True)
    # S(nf=4) = u+ + d+ + s+
    S4 = get({1: 1, -1: 1, 2: 1, -2: 1, 3: 1, -3: 1, 4: 1, -4: 1})
    assert pytest.approx(S4) == flavors.pids_from_intrinsic_evol("S", 4, False)
    # T3 = u+ - d+
    T3 = get({2: 1, -2: 1, 1: -1, -1: -1})
    assert pytest.approx(T3) == flavors.pids_from_intrinsic_evol("T3", 3, False)
    assert pytest.approx(T3) == flavors.pids_from_intrinsic_evol("T3", 4, False)
    # V15(nf=3) = V(nf=3) = u- + d- + s-
    assert pytest.approx(
        flavors.pids_from_intrinsic_evol("V", 3, False)
    ) == flavors.pids_from_intrinsic_evol("V15", 3, False)
    # V15(nf=4) =  u- + d- + s- - 3c- =!= V(nf=4)
    assert pytest.approx(
        flavors.pids_from_intrinsic_evol("V", 4, False)
    ) != flavors.pids_from_intrinsic_evol("V15", 4, False)
    # c+
    cp = get({4: 1, -4: 1})
    assert pytest.approx(cp) == flavors.pids_from_intrinsic_evol("c+", 3, False)
    assert pytest.approx(cp) == flavors.pids_from_intrinsic_evol("c+", 4, False)


def test_get_range():
    assert (3, 3) == flavors.get_range([])
    assert (3, 3) == flavors.get_range([member.MemberName(n) for n in ["S.S", "V3.V3"]])
    assert (3, 4) == flavors.get_range(
        [member.MemberName(n) for n in ["S.S", "V3.V3", "T15.S"]]
    )
    assert (4, 4) == flavors.get_range(
        [member.MemberName(n) for n in ["S.S", "V3.V3", "T15.T15"]]
    )


def test_rotate_pm_to_flavor():
    # g is still there
    assert all(([0] * (1 + 6) + [1] + [0] * 6) == flavors.rotate_pm_to_flavor("g"))
    # now t+ and t- are easiest
    assert all(
        ([0] + [1] + [0] * (2 * 5 + 1) + [1]) == flavors.rotate_pm_to_flavor("t+")
    )
    assert all(
        ([0] + [-1] + [0] * (2 * 5 + 1) + [1]) == flavors.rotate_pm_to_flavor("t-")
    )
    with pytest.raises(ValueError):
        flavors.rotate_pm_to_flavor("cbar")


def test_rotate_matching():
    m = flavors.rotate_matching(4)
    assert len(list(filter(lambda e: "c+" in e, m.keys()))) == 2
    assert len(list(filter(lambda e: "b-" in e, m.keys()))) == 1


def test_rotate_matching_is_inv():
    def replace_names(k):
        for q in range(4, 6 + 1):
            k = k.replace(flavors.quark_names[q - 1] + "+", f"T{q**2-1}").replace(
                flavors.quark_names[q - 1] + "-", f"V{q**2-1}"
            )
        return k

    def load(m):
        mm = np.zeros((len(br.evol_basis), len(br.evol_basis)))
        for k, v in m.items():
            k = replace_names(k)
            kk = k.split(".")
            mm[br.evol_basis.index(kk[0]), br.evol_basis.index(kk[1])] = v
        return mm

    for nf in range(4, 6 + 1):
        m = load(flavors.rotate_matching(nf))
        minv = load(flavors.rotate_matching_inverse(nf))
        np.testing.assert_allclose(m @ minv, np.eye(len(br.evol_basis)), atol=1e-10)

def test_pids_from_iuev():
    for nf in range(3, 6+1):
        labels = br.iuev_labels(nf)
        for lab in labels:
            n = flavors.pids_from_iuev(lab, nf, True)
            for lab2 in labels:
                n2 = flavors.pids_from_iuev(lab2, nf, False)
                if lab == lab2:
                    np.testing.assert_allclose(n @ n2, 1.)
                else:
                    np.testing.assert_allclose(n @ n2, 0., atol=1e-10, err_msg=f"{lab} is not orthogonal to {lab2} in nf={nf}")
    with pytest.raises(ValueError):
        flavors.pids_from_iuev("V3",4,True)
    with pytest.raises(ValueError):
        flavors.pids_from_iuev("T0",7,True)
    with pytest.raises(ValueError):
        flavors.pids_from_iuev("V0",7,True)