# -*- coding: utf-8 -*-

from eko.flavours import get_singlet_paths

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
