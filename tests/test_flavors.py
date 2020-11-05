# -*- coding: utf-8 -*-
import pytest

from eko.flavours import get_singlet_paths, get_all_flavour_paths, NSV, NSP


def test_get_singlet_paths():
    # trivial solution
    a = get_singlet_paths("q", "q", 1)
    assert a == [["S_qq"]]
    # level 2
    b = get_singlet_paths("q", "g", 2)
    assert b == [["S_qq", "S_qg"], ["S_qg", "S_gg"]]
    # level 4
    c = get_singlet_paths("g", "g", 4)
    assert len(c) == 2 ** 3
    for path in c:
        # check start + end
        assert path[0][-2] == "g"
        assert path[-1][-1] == "g"
        # check concatenation
        for k, el in enumerate(path[:-1]):
            assert el[-1] == path[k + 1][-2]

    # errors
    with pytest.raises(ValueError):
        get_singlet_paths("q", "g", 0)
    with pytest.raises(ValueError):
        get_singlet_paths("q", "S", 1)
    with pytest.raises(ValueError):
        get_singlet_paths("S", "q", 1)


def test_get_all_flavour_paths():
    # check range
    for nf in range(3, 6 + 1):
        ls = get_all_flavour_paths(nf)
        assert len(ls) == 2 * nf + 1
    # inspect nf=4
    ls4 = get_all_flavour_paths(4)
    # check full valence first
    v = ls4[0]
    assert v.name == "V"
    # no threshold
    p0 = v.get_path(3, 0)
    assert p0 == {"V": [[NSV]]}
    with pytest.raises(ValueError):
        v.get_path(3, 1)
    # 1 threshold
    p1 = v.get_path(4, 1)
    assert p1 == {"V": [[NSV, NSV]]}
    # check T15, which comes active only after crossing the first threshold
    t15 = ls4[6]
    assert t15.name == "T15"
    # no threshold
    p0 = t15.get_path(3, 0)
    assert p0 == {"S": [["S_qq"]], "g": [["S_qg"]]}
    # 1 threshold
    p0 = t15.get_path(4, 1)
    assert p0 == {"S": [[NSP, "S_qq"]], "g": [[NSP, "S_qg"]]}
    # check gluon
    g = ls4[-1]
    assert g.name == "g"
    # no threshold
    p0 = g.get_path(3, 0)
    assert p0 == {"S": [["S_gq"]], "g": [["S_gg"]]}
