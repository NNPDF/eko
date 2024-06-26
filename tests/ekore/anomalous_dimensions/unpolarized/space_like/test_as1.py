# Test LO splitting functions
import numpy as np

import ekore.anomalous_dimensions.unpolarized.space_like.as1 as ad_as1
from ekore import harmonics as h

NF = 5


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    cache = h.cache.reset()
    np.testing.assert_almost_equal(ad_as1.gamma_ns(N, cache), 0)


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    cache = h.cache.reset()
    np.testing.assert_almost_equal(
        ad_as1.gamma_ns(N, cache) + ad_as1.gamma_gq(N),
        0,
    )


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    cache = h.cache.reset()
    np.testing.assert_almost_equal(
        ad_as1.gamma_qg(N, NF) + ad_as1.gamma_gg(N, cache, NF), 0
    )


def test_gamma_qg_0():
    N = complex(1.0, 0.0)
    res = complex(-20.0 / 3.0, 0.0)
    np.testing.assert_almost_equal(ad_as1.gamma_qg(N, NF), res)


def test_gamma_gq_0():
    N = complex(0.0, 1.0)
    res = complex(4.0, -4.0) / 3.0
    np.testing.assert_almost_equal(ad_as1.gamma_gq(N), res)


def test_gamma_gg_0():
    N = complex(0.0, 1.0)
    cache = h.cache.reset()
    res = complex(5.195725159621, 10.52008856962)
    np.testing.assert_almost_equal(ad_as1.gamma_gg(N, cache, NF), res)
