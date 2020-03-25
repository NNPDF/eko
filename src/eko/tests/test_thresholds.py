"""
    Tests for the threshold class
"""
import numpy as np
from eko.thresholds import Threshold, Area


def test_ffns():
    # Check the setup for FFNS produces the correct thing
    tholder = Threshold(scheme="FFNS", nf=4, q2_ref=4.5)
    assert len(tholder._areas) == 1 # pylint: disable=protected-access
    area_path = tholder.get_path_from_q2_ref(87.4)
    assert len(area_path) == 1
    area = area_path[0]
    assert area.q2_min == 0.0
    assert area.q2_max == np.inf


def test_vnfs():
    # Tests the setup for VFNS
    th_list = [5, 50]
    qfin = 42
    for q2_ref in [2, 8, 61]:
        tholder = Threshold(scheme="VFNS", threshold_list=th_list, q2_ref=q2_ref)
        assert len(tholder._areas) == (len(th_list) + 1) # pylint: disable=protected-access
        area_path = tholder.get_path_from_q2_ref(qfin)
        # The first area should contain q2_ref and the last qfin
        assert area_path[0](q2_ref)
        assert area_path[-1](qfin)


def get_areas(q2_ref=42.0, nf=4):
    q2_change = q2_ref * 0.99
    area_left = Area(0.0, q2_change, q2_ref, nf)
    area_right = Area(q2_change, q2_ref * 2, q2_ref, nf)
    area_far = Area(q2_ref * 4, q2_ref * 5, q2_ref, nf)
    return area_left, area_right, area_far


def test_area_order():
    # Checks that areas work
    area_left, area_right, area_far = get_areas()
    assert area_left < area_right
    assert area_right > area_left
    assert area_far > area_left


def test_area_q2_towards():
    q2_ref = 42
    area_left, area_right, area_far = get_areas(q2_ref)
    assert area_left.q2_towards(q2_ref) == area_left.q2_max
    assert area_far.q2_towards(q2_ref) == area_far.q2_min
    assert area_right.q2_towards(q2_ref) == q2_ref
