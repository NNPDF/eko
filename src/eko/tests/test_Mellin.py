# -*- coding: utf-8 -*-
# Test Mellin stuff
from numpy.testing import assert_almost_equal

import numpy as np
from eko.Mellin import (
    inverse_Mellin_transform,
    get_path_Talbot,
    get_path_line,
    get_path_edge,
)


def test_inverse_Mellin_transform():
    """test inverse_Mellin_transform"""
    f = lambda N: 1.0 / (N + 1.0)
    g = lambda x: x
    p, j = get_path_Talbot()
    for x in [0.1, 0.3, 0.5, 0.7]:
        e = g(x)
        a = inverse_Mellin_transform(f, p, j, x, 1e-2)
        assert_almost_equal(e, a[0])
        assert_almost_equal(0.0, a[1])


def test_get_path_dev():
    """test derivatives to path"""
    epss = [1e-2, 1e-3, 1e-4, 1e-5]
    for pj in [
        get_path_Talbot(),
        get_path_Talbot(2),
        get_path_line(1),
        get_path_line(2, 2),
        get_path_edge(2),
        get_path_edge(2, 2),
    ]:
        (p, j) = pj
        for t0 in [0.2, 0.4, 0.6, 0.8]:  # avoid 0.5 due to Talbot+edge
            # compute numeric derivative + extrapolation
            num = []
            for eps in epss:
                a = (p(t0 + eps) - p(t0 - eps)) / (2.0 * eps)
                num = np.append(num, a)
            f = np.polyfit(epss, num, 2)
            ex = j(t0)
            assert_almost_equal(ex, f[2], decimal=4)
