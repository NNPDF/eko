# Test NNLO anomalous dimensions
import numpy as np

import ekore.anomalous_dimensions.as3 as ad_as3
from ekore import harmonics as h

NF = 5


# Reference numbers coming from Mathematica
def test_gamma_2():
    # number conservation - each is 0 on its own, see :cite:`Moch:2004pa`
    N = 1
    sx_n1 = h.sx(N, max_weight=3)
    np.testing.assert_allclose(ad_as3.gamma_nsv(N, NF, sx_n1), -0.000960586, rtol=3e-7)
    np.testing.assert_allclose(ad_as3.gamma_nsm(N, NF, sx_n1), 0.000594225, rtol=6e-7)

    # get singlet sector
    N = 2
    sx_n2 = h.sx(N, max_weight=4)
    gS2 = ad_as3.gamma_singlet(N, NF, sx_n2)

    # gluon momentum conservation
    np.testing.assert_allclose(gS2[0, 1] + gS2[1, 1], -0.00388726, rtol=2e-6)
    # quark momentum conservation
    np.testing.assert_allclose(gS2[0, 0] + gS2[1, 0], 0.00169375, rtol=2e-6)

    assert gS2.shape == (2, 2)

    # test nsv_2 equal to referece value
    np.testing.assert_allclose(ad_as3.gamma_nsv(N, NF, sx_n2), 188.325593, rtol=3e-7)
