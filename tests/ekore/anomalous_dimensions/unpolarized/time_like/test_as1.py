import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as1 as ad_as1
import ekore.anomalous_dimensions.unpolarized.space_like.as1 as sl_ad_as1
import ekore.harmonics.w1 as w1

NF = 5


def test_qq():
    np.testing.assert_almost_equal(
        ad_as1.gamma_qq(1), 0
    )
    s1 = w1.S1(1)
    np.testing.assert_almost_equal(
        ad_as1.gamma_qq(1), sl_ad_as1.gamma_ns(1, s1)
    )


def test_qg():
    np.testing.assert_almost_equal(
        ad_as1.gamma_qg(2), -1/3
    )

def test_gq():
    np.testing.assert_almost_equal(
        ad_as1.gamma_gq(3, NF), -140/9
    )

def test_gg():
    np.testing.assert_almost_equal(
        ad_as1.gamma_gg(2, NF), 10/3
    )
    s1 = w1.S1(2)
    np.testing.assert_almost_equal(
        ad_as1.gamma_gg(2, NF), sl_ad_as1.gamma_gg(2, s1, NF)
    )
