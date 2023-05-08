"""Testing values obtained from |MELA| functions."""

import numpy as np

import ekore.anomalous_dimensions.unpolarized.space_like.as1 as sl_ad_as1
import ekore.anomalous_dimensions.unpolarized.time_like.as1 as ad_as1
from ekore.harmonics import cache as c

NF = 5


def test_qq():
    cache = c.reset()
    np.testing.assert_almost_equal(ad_as1.gamma_qq(1, cache), 0)
    np.testing.assert_almost_equal(
        ad_as1.gamma_qq(1, cache), sl_ad_as1.gamma_ns(1, cache)
    )


def test_qg():
    np.testing.assert_almost_equal(ad_as1.gamma_qg(2), -1 / 3)


def test_gq():
    np.testing.assert_almost_equal(ad_as1.gamma_gq(3, NF), -140 / 9)


def test_gg():
    cache = c.reset()
    np.testing.assert_almost_equal(ad_as1.gamma_gg(2, NF, cache), 10 / 3)
    np.testing.assert_almost_equal(
        ad_as1.gamma_gg(2, NF, cache), sl_ad_as1.gamma_gg(2, cache, NF)
    )
