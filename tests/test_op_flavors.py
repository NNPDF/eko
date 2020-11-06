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

    def test_is_physical(self):
        assert not flavors.MemberName("NS_p").is_physical
        assert flavors.MemberName("S.S").is_physical


def test_pids_from_intrinsic_evol():
    raw = dict(zip(br.flavor_basis_pids, np.zeros(len(br.flavor_basis_pids))))
    # g
    g = raw.copy()
    g.update({21: 1})
    assert pytest.approx(g) == flavors.pids_from_intrinsic_evol("g", 3)
    assert pytest.approx(g) == flavors.pids_from_intrinsic_evol("g", 4)
    # S(nf=3) = u+ + d+ + s+
    S3 = raw.copy()
    S3.update({1: 1, -1: 1, 2: 1, -2: 1, 3: 1, -3: 1})
    assert pytest.approx(S3) == flavors.pids_from_intrinsic_evol("S", 3)
    # S(nf=4) = u+ + d+ + s+
    S4 = raw.copy()
    S4.update({1: 1, -1: 1, 2: 1, -2: 1, 3: 1, -3: 1, 4: 1, -4: 1})
    assert pytest.approx(S4) == flavors.pids_from_intrinsic_evol("S", 4)
    # T3 = u+ - d+
    T3 = raw.copy()
    T3.update({2: 1, -2: 1, 1: -1, -1: -1})
    assert pytest.approx(T3) == flavors.pids_from_intrinsic_evol("T3", 3)
    assert pytest.approx(T3) == flavors.pids_from_intrinsic_evol("T3", 4)
    # V15(nf=3) = V(nf=3) = u- + d- + s-
    assert pytest.approx(
        flavors.pids_from_intrinsic_evol("V", 3)
    ) == flavors.pids_from_intrinsic_evol("V15", 3)
    # V15(nf=4) =  u- + d- + s- - 3c- =!= V(nf=4)
    assert pytest.approx(
        flavors.pids_from_intrinsic_evol("V", 4)
    ) != flavors.pids_from_intrinsic_evol("V15", 4)
    # c+
    cp = raw.copy()
    cp.update({4: 1, -4: 1})
    assert pytest.approx(cp) == flavors.pids_from_intrinsic_evol("c+", 3)
    assert pytest.approx(cp) == flavors.pids_from_intrinsic_evol("c+", 4)


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
