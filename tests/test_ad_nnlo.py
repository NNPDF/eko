# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

import eko.anomalous_dimensions.nnlo as ad_nnlo

NF = 0


def test_gamma_2():
    # reference values are obtained from MMa
    # non-siglet sector
    np.testing.assert_allclose(
        ad_nnlo.gamma_nsm_2(1, NF) - ad_nnlo.gamma_nsv_2(1, NF), 0, atol=2e-3
    )  # pylint: disable=line-too-long

    gS2 = ad_nnlo.gamma_singlet_2(2, NF)
    # gluon momentum conservation
    np.testing.assert_allclose(gS2[0, 1] + gS2[1, 1], 0, atol=9e-2)
    # quark momentum conservation
    np.testing.assert_allclose(gS2[0, 0] + gS2[1, 0], 0, atol=2e-3)

    assert gS2.shape == (2, 2)
