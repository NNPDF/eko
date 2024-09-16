from math import nan

import numpy as np
import pytest

from eko.quantities import heavy_quarks as hq


def test_HeavyQuarks():
    with pytest.raises(ValueError):
        hq.MatchingRatios([1, 2, 3, 4])
    r = hq.MatchingRatios([0.5, 2.0, 3.0])
    assert len(r) == 3
    assert r.c == 0.5
    assert r.b == 2.0
    assert r.t == 3.0
    r.c = 0.7
    assert len(r) == 3
    assert r.c == 0.7
    assert r.b == 2.0
    assert r.t == 3.0
    r.b = 2.7
    assert len(r) == 3
    assert r.c == 0.7
    assert r.b == 2.7
    assert r.t == 3.0
    r.t = 3.7
    assert len(r) == 3
    assert r.c == 0.7
    assert r.b == 2.7
    assert r.t == 3.7


def test_HeavyInfo():
    i = hq.HeavyInfo(
        masses=hq.HeavyQuarkMasses(
            [
                hq.QuarkMassRef([2.0, nan]),
                hq.QuarkMassRef([5.0, nan]),
                hq.QuarkMassRef([100.0, nan]),
            ]
        ),
        masses_scheme=hq.QuarkMassScheme.POLE,
        matching_ratios=hq.MatchingRatios([1.5, 2.0, 3.0]),
    )
    np.testing.assert_allclose(i.squared_ratios, [2.25, 4.0, 9.0])
