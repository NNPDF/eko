"""Tests for the threshold class"""
from dataclasses import astuple

import numpy as np

from eko.matchings import Atlas, Segment, flavor_shift, is_downward_path, nf_default
from eko.quantities.heavy_quarks import MatchingScales


class TestPathSegment:
    def test_tuple(self):
        p = Segment(0, 1, 3)
        assert astuple(p) == (0, 1, 3)
        # is hashable?
        d = {}
        d[p] = 1
        assert d[p] == 1

    def test_str(self):
        p = Segment(0, 1, 3)
        s = str(p)
        assert s.index("0") > 0
        assert s.index("1") > 0
        assert s.index("3") > 0


ZERO = (0.0, 0)


class TestAtlas:
    def test_init(self):
        # 3 thr
        tc3 = Atlas(MatchingScales([1, 2, 3]), ZERO)
        assert tc3.walls == [0, 1, 2, 3, np.inf]
        # 2 thr
        tc2 = Atlas(MatchingScales([0, 2, 3]), ZERO)
        assert tc2.walls == [0, 0, 2, 3, np.inf]

    def test_nfref(self):
        # weird but fine
        tc = Atlas(MatchingScales([1, 2, 3]), (1.5, 3))
        p = tc.path((1.5, 4))
        assert len(p) == 2
        assert astuple(p[0]) == (1.5, 1.0, 3)
        assert astuple(p[1]) == (1.0, 1.5, 4)

    def test_str(self):
        walls = MatchingScales([1.23, 9.87, 14.54])
        stc3 = str(Atlas(walls, ZERO))

        for w in walls:
            assert f"{w:.2e}" in stc3

    def test_ffns(self):
        tc3 = Atlas.ffns(3, 0.0)
        assert tc3.walls == [0] + [np.inf] * 4
        tc4 = Atlas.ffns(4, 3.0)
        assert tc4.walls == [0] * 2 + [np.inf] * 3
        assert len(tc4.path((2.0, 4))) == 1

    def test_path_3thr(self):
        tc = Atlas(MatchingScales([1, 2, 3]), (0.5, 3))
        p1 = tc.path((0.7, 3))
        assert len(p1) == 1
        assert astuple(p1[0]) == (0.5, 0.7, 3)

    def test_path_3thr_backward(self):
        tc = Atlas(MatchingScales([1, 2, 3]), (2.5, 5))
        p1 = tc.path((0.7, 3))
        assert len(p1) == 3
        assert astuple(p1[0]) == (2.5, 2.0, 5)
        assert astuple(p1[1]) == (2.0, 1.0, 4)
        assert astuple(p1[2]) == (1.0, 0.7, 3)

    def test_path_3thr_on_threshold(self):
        tc = Atlas(MatchingScales([1, 2, 3]), (0.5, 3))
        # on the right of mc
        p3 = tc.path((1.0, 4))
        assert len(p3) == 2
        assert p3[0].nf == 3
        assert astuple(p3[1]) == (1.0, 1.0, 4)
        # on the left of mc
        p4 = tc.path((1.0, 3))
        assert len(p4) == 1
        assert p4[0].nf == 3

    def test_path_3thr_weird(self):
        tc = Atlas(MatchingScales([1, 2, 3]), (0.5, 3))
        # the whole distance underground
        p6 = tc.path((3.5, 3))
        assert len(p6) == 1
        assert astuple(p6[0]) == (0.5, 3.5, 3)
        mu2_from = 3.5
        mu2_to = 0.7
        #                   0
        #      1 <-----------
        #      ---> 2
        #   3 < -----
        #      |    |    |
        origin = (mu2_from, 3)
        target = (mu2_to, 5)
        p7 = Atlas(MatchingScales([1, 2, 3]), origin).path(target)
        assert [s.nf for s in p7] == list(range(3, 5 + 1))
        assert p7[0].origin == mu2_from
        assert p7[2].target == mu2_to
        #                   0
        #      1 <-----------
        #      ---> 2 -> 3
        #   4 < ---------
        #      |    |    |
        target = (mu2_to, 6)
        p8 = Atlas(MatchingScales([1, 2, 3]), origin).path(target)
        assert [s.nf for s in p8] == list(range(3, 6 + 1))
        assert p8[0].origin == mu2_from
        assert p8[3].target == mu2_to


def test_nf():
    nf4 = Atlas.ffns(4, 0.0)
    for q2 in [1.0, 1e1, 1e2, 1e3, 1e4]:
        assert nf_default(q2, nf4) == 4
    at = Atlas(MatchingScales([1, 2, 3]), (0.5, 3))
    assert nf_default(0.9, at) == 3
    assert nf_default(1.1, at) == 4


def test_downward_path():
    thr_atlas = Atlas(np.power([2, 3, 4], 2).tolist(), (91**2, 3))
    mu2_to = 5**2
    path_3 = thr_atlas.path((mu2_to, 3))
    # path_3 is downward in q2
    is_downward = is_downward_path(path_3)
    assert is_downward is True
    assert flavor_shift(is_downward) == 4
    # path_6 is downward in q2, but forward in nf
    path_6 = thr_atlas.path((mu2_to, 6))
    is_downward = is_downward_path(path_6)
    assert is_downward is False
    assert flavor_shift(is_downward) == 3
