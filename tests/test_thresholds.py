# -*- coding: utf-8 -*-
"""
    Tests for the threshold class
"""
import pytest
import numpy as np

from eko.thresholds import ThresholdsAtlas, Area, PathSegment


class TestPathSegment:
    def test_intersect_fully_inside(self):
        a = Area(0, 1, 3)
        fwd = PathSegment.intersect(0.2, 0.8, a)
        assert fwd.q2_from == 0.2
        assert fwd.q2_to == 0.8
        assert fwd._area == a  # pylint: disable=protected-access
        bwd = PathSegment.intersect(0.8, 0.2, a)
        assert bwd.q2_from == 0.8
        assert bwd.q2_to == 0.2
        assert bwd._area == a  # pylint: disable=protected-access

    def test_intersect_too_low(self):
        a = Area(0, 1, 3)
        fwd = PathSegment.intersect(-0.2, 0.8, a)
        assert fwd.q2_from == 0
        assert fwd.q2_to == 0.8
        assert fwd._area == a  # pylint: disable=protected-access
        bwd = PathSegment.intersect(0.8, -0.2, a)
        assert bwd.q2_from == 0.8
        assert bwd.q2_to == 0.0
        assert bwd._area == a  # pylint: disable=protected-access

    def test_intersect_no_overlap(self):
        a = Area(0, 1, 3)
        fwd = PathSegment.intersect(-0.2, 1.8, a)
        assert fwd.q2_from == 0
        assert fwd.q2_to == 1
        assert fwd._area == a  # pylint: disable=protected-access
        bwd = PathSegment.intersect(1.8, -0.2, a)
        assert bwd.q2_from == 1
        assert bwd.q2_to == 0.0
        assert bwd._area == a  # pylint: disable=protected-access

    def test_print(self):
        a = Area(0, 1, 3)
        assert "nf=3" in str(a)

        p = PathSegment(0, 1, a)
        assert "nf=3" in str(p)

    def test_tuple(self):
        a = Area(0, 1, 3)
        p = PathSegment(0, 1, a)
        assert p.tuple == (0, 1)
        # is hashable?
        d = dict()
        d[p.tuple] = 1
        assert d[p.tuple] == 1


class TestThresholdsConfig:
    def test_init(self):
        # 3 thr
        tc3 = ThresholdsAtlas([1, 2, 3])
        assert tc3.area_walls == [0, 1, 2, 3, np.inf]
        assert tc3.areas[0].nf == 3
        assert tc3.areas[1].nf == 4
        # 2 thr
        tc2 = ThresholdsAtlas([0, 2, 3])
        assert tc2.area_walls == [0, 0, 2, 3, np.inf]
        assert tc2.areas[0].nf == 3
        assert tc2.areas[1].nf == 4

        # errors
        with pytest.raises(ValueError):
            ThresholdsAtlas([1.0, 0.0])

    def test_from_dict(self):
        tc = ThresholdsAtlas.from_dict(
            {
                "mc": 1.0,
                "mb": 4.0,
                "mt": 100.0,
                "kcThr": 1,
                "kbThr": 2.0,
                "ktThr": np.inf,
                "Q0": 1.0,
                "MaxNfPdf": 6,
            }
        )
        assert tc.area_walls[1:-1] == [1.0, 64.0, np.inf]
        assert tc.q2_ref == 1.0

    def test_ffns(self):
        tc3 = ThresholdsAtlas.ffns(3)
        assert tc3.area_walls == [0] + [np.inf]
        tc4 = ThresholdsAtlas.ffns(4)
        assert tc4.area_walls == [0] * 2 + [np.inf]

    def test_path_3thr(self):
        tc = ThresholdsAtlas([1, 2, 3], 0.5)
        p1 = tc.path(0.7)
        assert len(p1) == 1
        assert p1[0].q2_from == 0.5
        assert p1[0].q2_to == 0.7
        assert p1[0].nf == 3

        p2 = tc.path(1.5, 2.5)
        assert len(p2) == 2
        assert p2[0].nf == 5
        assert p2[1].nf == 4

    def test_path_filter(self):
        ta1 = ThresholdsAtlas([0, 2, 3], 0.5)
        assert len(ta1.path(2.5)) == 2
        ta2 = ThresholdsAtlas([1, 2, 3], 0.5)
        assert len(ta2.path(2.5)) == 3

    def test_nf(self):
        nf4 = ThresholdsAtlas.ffns(4)
        for q2 in [1.0, 1e1, 1e2, 1e3, 1e4]:
            assert nf4.nf(q2) == 4
        ta = ThresholdsAtlas([1, 2, 3], 0.5)
        assert ta.nf(0.9) == 3
        assert ta.nf(1.1) == 4
