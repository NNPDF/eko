import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as1 as ad_as1

NF = 5


def test_qq():
    np.testing.assert_almost_equal(
        ad_as1.gamma_qq(1), 0
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
