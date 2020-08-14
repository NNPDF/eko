# -*- coding: utf-8 -*-
"""
    Tests for the threshold class
"""
import pytest
import numpy as np

from eko.thresholds import ThresholdsConfig, Area


class TestThresholdsConfig:
    def test_init(self):
        # at the moment it is fine to have ZM-VFNS with NfFF set
        _tc = ThresholdsConfig(2, "ZM-VFNS", threshold_list=[1, 2], nf=2)
        # errors
        with pytest.raises(ValueError):
            ThresholdsConfig(2, "FFNS")
        with pytest.raises(ValueError):
            ThresholdsConfig(2, "FFNS", nf=2, threshold_list=[1.0, 2.0])
        with pytest.raises(NotImplementedError):
            ThresholdsConfig(
                2, "VFNS"
            )  # I enforced ZM-VFNS -> "explicit is better then implicit"
        with pytest.raises(ValueError):
            ThresholdsConfig(2, "ZM-VFNS")

    def test_from_dict(self):
        # here it is ok to have *all* keys set
        tc_vfns = ThresholdsConfig.from_dict(
            {"FNS": "ZM-VFNS", "NfFF": 3, "Q0": np.sqrt(2), "mc": 2, "mb": 4, "mt": 175}
        )
        assert tc_vfns.scheme == "ZM-VFNS"
        tc_ffns = ThresholdsConfig.from_dict(
            {"FNS": "FFNS", "NfFF": 3, "Q0": np.sqrt(2), "mc": 2, "mb": 4, "mt": 175}
        )
        assert tc_ffns.scheme == "FFNS"

    def test_nf_range_nf_ref(self):
        # FFNS
        for nf in [3, 4, 5, 6]:
            tc_ffns = ThresholdsConfig(2, "FFNS", nf=nf)
            assert list(tc_ffns.nf_range()) == [nf]
            assert tc_ffns.nf_ref == nf
        # VFNS - 1 threshold
        for k, q2_ref in enumerate([2, 4]):
            tc_vfns1 = ThresholdsConfig(q2_ref, "ZM-VFNS", threshold_list=[3])
            assert list(tc_vfns1.nf_range()) == [3, 4]
            assert (
                tc_vfns1.nf_ref == 3 + k
            )  # = 3 if below and 4 if above the one threshold

    def test_ffns(self):
        # Check the setup for FFNS produces the correct thing
        tholder = ThresholdsConfig(4.5, "FFNS", nf=4)
        assert len(tholder._areas) == 1  # pylint: disable=protected-access
        area_path = tholder.get_path_from_q2_ref(87.4)
        assert len(area_path) == 1
        area = area_path[0]
        assert area.q2_min == 0.0
        assert area.q2_max == np.inf

    def test_vnfs(self):
        # Tests the setup for VFNS
        th_list = [5, 50]
        q2fin = 42
        for q2_ref in [2, 8, 61]:
            tholder = ThresholdsConfig(q2_ref, "ZM-VFNS", threshold_list=th_list)
            assert (
                len(tholder._areas) == len(th_list) + 1 # pylint: disable=protected-access
            )
            area_path = tholder.get_path_from_q2_ref(q2fin)
            # The first area should contain q2_ref and the last q2fin
            assert area_path[0](q2_ref)
            assert area_path[-1](q2fin)


class TestAreas:
    def _get_areas(self, q2_ref=42.0, nf=4):
        q2_change = q2_ref * 0.99
        area_left = Area(0.0, q2_change, q2_ref, nf)
        area_right = Area(q2_change, q2_ref * 2, q2_ref, nf)
        area_far = Area(q2_ref * 4, q2_ref * 5, q2_ref, nf)
        return area_left, area_right, area_far

    def test_order(self):
        # Checks that areas work
        area_left, area_right, area_far = self._get_areas()
        assert area_left < area_right
        assert area_right > area_left
        assert area_far > area_left

    def test_q2_towards(self):
        q2_ref = 42
        area_left, area_right, area_far = self._get_areas(q2_ref)
        assert area_left.q2_towards(q2_ref) == area_left.q2_max
        assert area_far.q2_towards(q2_ref) == area_far.q2_min
        assert area_right.q2_towards(q2_ref) == q2_ref
