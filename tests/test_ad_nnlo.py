# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

import eko.anomalous_dimensions.nnlo as ad_nnlo
import eko.anomalous_dimensions.harmonics as harmonics

NF = 5


def test_gamma_2():
    # numeber conservation
    sx = np.full(1, harmonics.harmonic_S1(1))
    sx = np.append(sx, harmonics.harmonic_S2(1))
    sx = np.append(sx, harmonics.harmonic_S3(1))
    sx = np.append(sx, harmonics.harmonic_S4(1))
    print(sx)
    np.testing.assert_allclose(
        ad_nnlo.gamma_nsm_2(1, NF, sx) - ad_nnlo.gamma_nsv_2(1, NF, sx), 0, atol=2e-3
    )

    sx = np.full(1, harmonics.harmonic_S1(2))
    sx = np.append(sx, harmonics.harmonic_S2(2))
    sx = np.append(sx, harmonics.harmonic_S3(2))
    sx = np.append(sx, harmonics.harmonic_S4(2))
    gS2 = ad_nnlo.gamma_singlet_2(2, NF, sx)
    # gluon momentum conservation
    np.testing.assert_allclose(gS2[0, 1] + gS2[1, 1], 0, atol=9e-2)
    # quark momentum conservation
    np.testing.assert_allclose(gS2[0, 0] + gS2[1, 0], 0, atol=2e-3)

    assert gS2.shape == (2, 2)
