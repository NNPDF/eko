from math import nan

import numpy as np
import pytest

from eko.quantities import heavy_quarks as hq


def test_HeavyQuarks():
    with pytest.raises(ValueError):
        hq.MatchingRatios([1, 2, 3, 4])
    l = hq.MatchingRatios([0.5, 2.0, 3.0])
    assert len(l) == 3
    assert l.c == 0.5
    assert l.b == 2.0
    assert l.t == 3.0
    l.c = 0.7
    assert len(l) == 3
    assert l.c == 0.7
    assert l.b == 2.0
    assert l.t == 3.0
    l.b = 2.7
    assert len(l) == 3
    assert l.c == 0.7
    assert l.b == 2.7
    assert l.t == 3.0
    l.t = 3.7
    assert len(l) == 3
    assert l.c == 0.7
    assert l.b == 2.7
    assert l.t == 3.7


def test_HeavyInfo():
    i = hq.HeavyInfo(
        num_flavs_init=4,
        num_flavs_max_pdf=6,
        intrinsic_flavors=[4, 5],
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
