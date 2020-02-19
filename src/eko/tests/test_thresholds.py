"""
    Tests for the threshold class
"""
import numpy as np
from eko.thresholds import Threshold, Area


def test_ffns():
    # Check the setup for FFNS produces the correct thing
    tholder = Threshold(scheme="FFNS", nf=4, qref=4.5)
    assert len(tholder._areas) == 1
    area_path = tholder.get_path_from_q0(87.4)
    assert len(area_path) == 1
    area = area_path[0]
    assert area.qmin == 0.0
    assert area.qmax == np.inf


def test_vnfs():
    # Tests the setup for VFNS
    th_list = [5, 50]
    qfin = 42
    for qref in [2, 8, 61]:
        tholder = Threshold(scheme="VFNS", threshold_list=th_list, qref=qref)
        assert len(tholder._areas) == (len(th_list) + 1)
        area_path = tholder.get_path_from_q0(qfin)
        # The first area should contain qref and the last qfin
        assert area_path[0](qref)
        assert area_path[-1](qfin)


def get_areas(qref=42.0, nf=4):
    q_change = qref - qref * 0.01
    area_left = Area(0.0, q_change, qref, nf)
    area_right = Area(q_change, qref * 2, qref, nf)
    area_far = Area(qref * 4, qref * 5, qref, nf)
    return area_left, area_right, area_far


def test_area_order():
    # Checks that areas work
    area_left, area_right, area_far = get_areas()
    assert area_left < area_right
    assert area_right > area_left
    assert area_far > area_left


def test_area_q_towards():
    qref = 42
    area_left, area_right, area_far = get_areas(qref)
    assert area_left.q_towards(qref) == area_left.qmax
    assert area_far.q_towards(qref) == area_far.qmin
    assert area_right.q_towards(qref) == qref
