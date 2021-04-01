# -*- coding: utf-8 -*-
# Test NLO anomalous dims
import numpy as np

from eko import constants

# constants.CA = 0

import eko.anomalous_dimensions.nlo as ad_nlo
import eko.anomalous_dimensions.harmonics as h

CA = constants.CA
CF = constants.CF

NF = 5


def test_gamma_1():
    # number conservation
    np.testing.assert_allclose(ad_nlo.gamma_nsm_1(1, NF), 0.0, atol=2e-6)

    gS1 = ad_nlo.gamma_singlet_1(2, NF)
    # gluon momentum conservation
    # the CA*NF term seems to be tough to compute, so raise the constraint ...
    np.testing.assert_allclose(gS1[0, 1] + gS1[1, 1], 0.0, atol=4e-5)
    # quark momentum conservation
    np.testing.assert_allclose(gS1[0, 0] + gS1[1, 0], 0.0, atol=2e-6)

    assert gS1.shape == (2, 2)

    # reference values are obtained from MMa
    # Non singlet sector
    np.testing.assert_allclose(
        ad_nlo.gamma_nsp_1(2, NF), (-112.0 * CF + 376.0 * CA - 64.0 * NF) * CF / 27.0
    )
    # singlet sector
    np.testing.assert_allclose(ad_nlo.gamma_ps_1(2, NF), -40.0 * CF * NF / 27.0)
    np.testing.assert_allclose(gS1[0, 1], (-74.0 * CF - 35.0 * CA) * NF / 27.0)  # qg
    np.testing.assert_allclose(
        gS1[1, 0], (112.0 * CF - 376.0 * CA + 104.0 * NF) * CF / 27.0
    )  # gq

    # add additional point at (analytical) continuation point
    np.testing.assert_allclose(
        ad_nlo.gamma_nsm_1(2, NF),
        (
            (34.0 / 27.0 * (-47.0 + 6 * np.pi ** 2) - 16.0 * h.zeta3) * CF
            + (373.0 / 9.0 - 34.0 * np.pi ** 2 / 9.0 + 8.0 * h.zeta3) * CA
            - 64.0 * NF / 27.0
        )
        * CF,
    )
    np.testing.assert_allclose(
        ad_nlo.gamma_nsp_1(3, NF),
        (
            (-34487.0 / 432.0 + 86.0 * np.pi ** 2 / 9.0 - 16.0 * h.zeta3) * CF
            + (459.0 / 8.0 - 43.0 * np.pi ** 2 / 9.0 + 8.0 * h.zeta3) * CA
            - 415.0 * NF / 108.0
        )
        * CF,
    )
    np.testing.assert_allclose(ad_nlo.gamma_ps_1(3, NF), -1391.0 * CF * NF / 5400.0)
    gS1 = ad_nlo.gamma_singlet_1(3, NF)
    np.testing.assert_allclose(
        gS1[1, 0],
        (
            973.0 / 432.0 * CF
            + (2801.0 / 5400.0 - 7.0 * np.pi ** 2 / 9.0) * CA
            + 61.0 / 54.0 * NF
        )
        * CF,
    )  # gq
    np.testing.assert_allclose(
        gS1[1, 1],
        (
            (-79909.0 / 3375.0 + 194.0 * np.pi ** 2 / 45.0 - 8.0 * h.zeta3) * CA ** 2
            - 967.0 / 270.0 * CA * NF
            + 541.0 / 216.0 * CF * NF
        ),
        rtol=6e-7,
    )  # gg
    gS1 = ad_nlo.gamma_singlet_1(4, NF)
    np.testing.assert_allclose(
        gS1[0, 1], (-56317.0 / 18000.0 * CF + 16387.0 / 9000.0 * CA) * NF
    )  # qg
