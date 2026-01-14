"""Testing values obtained from |MELA| functions and 1905.01310."""

import numpy as np

import ekore.anomalous_dimensions.unpolarized.space_like.as1 as sl_ad_as1
import ekore.anomalous_dimensions.unpolarized.time_like.as1 as ad_as1
from eko.constants import CA, CF
from ekore.harmonics import cache as c

NF = 5


# Thanks Yuxun Guo (@yuxunguo)
def n3(nf: int):
    """Implements 1905.01310 Eq. (A6)"""
    cf = CF
    ca = CA

    gTqq0 = 25 / 6 * cf
    gTgq0 = -7 / 6 * cf
    gTqg0 = -7 / 15 * nf
    gTgg0 = 14 / 5 * ca + 2 / 3 * nf

    return np.array([[gTqq0, gTgq0 * 2 * nf], [gTqg0 / (2 * nf), gTgg0]])


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


def test_singlet_n3():
    cache = c.reset()
    # test against 1905.01310
    for nf in range(3, 6 + 1):
        np.testing.assert_allclose(
            ad_as1.gamma_singlet(3.0, nf, cache), n3(nf), err_msg=f"{nf=}"
        )
