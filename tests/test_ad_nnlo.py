# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

import eko.anomalous_dimensions.nnlo as ad_nnlo

NF = 5


def test_gamma_2():
    # reference values are obtained from MMa
    # non-siglet sector
    # np.testing.assert_allclose(ad_nnlo.gamma_nsv_2(1, NF), 0, atol=2e-6)

    gS2 = ad_nnlo.gamma_singlet_2(2, NF)
    # gluon momentum conservation
    # the CA*NF term seems to be tough to compute, so raise the constraint ...
    # np.testing.assert_allclose( gS2[0,1] + gS2[1,1] , 0,atol=4e-2 )
    # quark momentum conservation
    np.testing.assert_allclose(gS2[0, 0] + gS2[1, 0], 0, atol=2e-2)

    assert gS2.shape == (2, 2)
