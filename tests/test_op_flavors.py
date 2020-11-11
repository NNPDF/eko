# -*- coding: utf-8 -*-
import numpy as np

import pytest

from eko.operator import flavors
from eko import basis_rotation as br


class TestOpMember:
    def test_hash(self):
        d = {flavors.MemberName("S.S"): 1}
        assert flavors.MemberName("S.S") in d
        assert flavors.MemberName("S.g") not in d

    def test_split(self):
        Sg = flavors.MemberName("S.g")
        assert Sg.input == "g"
        assert Sg.target == "S"
        # errors
        with pytest.raises(ValueError):
            _ = flavors.MemberName(".").input
        with pytest.raises(ValueError):
            _ = flavors.MemberName("bla").input


def test_pids_from_intrinsic_evol():
    def get(d):
        raw = np.zeros(len(br.flavor_basis_pids))
        for pid, w in d.items():
            raw[br.flavor_basis_pids.index(pid)] = w
        return raw

    # g
    g = get({21: 1})
    assert pytest.approx(g) == flavors.pids_from_intrinsic_evol("g", 3, False)
    assert pytest.approx(g) == flavors.pids_from_intrinsic_evol("g", 4, False)
    # S(nf=3) = u+ + d+ + s+
    S3 = get({1: 1, -1: 1, 2: 1, -2: 1, 3: 1, -3: 1})
    assert pytest.approx(S3) == flavors.pids_from_intrinsic_evol("S", 3, False)
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
    assert (3, 3) == flavors.get_range(
        [flavors.MemberName(n) for n in ["S.S", "V3.V3"]]
    )
    assert (3, 4) == flavors.get_range(
        [flavors.MemberName(n) for n in ["S.S", "V3.V3", "T15.S"]]
    )
    assert (4, 4) == flavors.get_range(
        [flavors.MemberName(n) for n in ["S.S", "V3.V3", "T15.T15"]]
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
