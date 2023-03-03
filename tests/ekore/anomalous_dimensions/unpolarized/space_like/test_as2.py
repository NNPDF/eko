# Test NLO anomalous dims
import numpy as np

import ekore.anomalous_dimensions.unpolarized.space_like.as2 as ad_as2
import ekore.harmonics as h
from eko import constants as const

NF = 5


def test_gamma_1():
    # number conservation
    sx_n1 = h.sx(1, 2)
    np.testing.assert_allclose(ad_as2.gamma_nsm(1, NF, sx_n1), 0.0, atol=2e-6)

    sx_n2 = h.sx(2, 2)
    gS1 = ad_as2.gamma_singlet(2, NF, sx_n2)
    # gluon momentum conservation
    # the CA*NF term seems to be tough to compute, so raise the constraint ...
    np.testing.assert_allclose(gS1[0, 1] + gS1[1, 1], 0.0, atol=4e-5)
    # quark momentum conservation
    np.testing.assert_allclose(gS1[0, 0] + gS1[1, 0], 0.0, atol=2e-6)

    assert gS1.shape == (2, 2)

    # reference values are obtained from MMa
    # non-singlet sector
    np.testing.assert_allclose(
        ad_as2.gamma_nsp(2, NF, sx_n2),
        (-112.0 * const.CF + 376.0 * const.CA - 64.0 * NF) * const.CF / 27.0,
    )
    # singlet sector
    np.testing.assert_allclose(ad_as2.gamma_ps(2, NF), -40.0 * const.CF * NF / 27.0)
    np.testing.assert_allclose(
        gS1[0, 1], (-74.0 * const.CF - 35.0 * const.CA) * NF / 27.0
    )  # qg
    np.testing.assert_allclose(
        gS1[1, 0], (112.0 * const.CF - 376.0 * const.CA + 104.0 * NF) * const.CF / 27.0
    )  # gq

    # add additional point at (analytical) continuation point
    np.testing.assert_allclose(
        ad_as2.gamma_nsm(2, NF, sx_n2),
        (
            (34.0 / 27.0 * (-47.0 + 6 * np.pi**2) - 16.0 * h.constants.zeta3)
            * const.CF
            + (373.0 / 9.0 - 34.0 * np.pi**2 / 9.0 + 8.0 * h.constants.zeta3)
            * const.CA
            - 64.0 * NF / 27.0
        )
        * const.CF,
    )
    sx_n3 = h.sx(3, 2)
    sx_n4 = h.sx(4, 2)
    np.testing.assert_allclose(
        ad_as2.gamma_nsp(3, NF, sx_n3),
        (
            (-34487.0 / 432.0 + 86.0 * np.pi**2 / 9.0 - 16.0 * h.constants.zeta3)
            * const.CF
            + (459.0 / 8.0 - 43.0 * np.pi**2 / 9.0 + 8.0 * h.constants.zeta3)
            * const.CA
            - 415.0 * NF / 108.0
        )
        * const.CF,
    )
    np.testing.assert_allclose(ad_as2.gamma_ps(3, NF), -1391.0 * const.CF * NF / 5400.0)
    gS1 = ad_as2.gamma_singlet(3, NF, sx_n3)
    np.testing.assert_allclose(
        gS1[1, 0],
        (
            973.0 / 432.0 * const.CF
            + (2801.0 / 5400.0 - 7.0 * np.pi**2 / 9.0) * const.CA
            + 61.0 / 54.0 * NF
        )
        * const.CF,
    )  # gq
    np.testing.assert_allclose(
        gS1[1, 1],
        (
            (-79909.0 / 3375.0 + 194.0 * np.pi**2 / 45.0 - 8.0 * h.constants.zeta3)
            * const.CA**2
            - 967.0 / 270.0 * const.CA * NF
            + 541.0 / 216.0 * const.CF * NF
        ),
        rtol=6e-7,
    )  # gg
    gS1 = ad_as2.gamma_singlet(4, NF, sx_n4)
    np.testing.assert_allclose(
        gS1[0, 1], (-56317.0 / 18000.0 * const.CF + 16387.0 / 9000.0 * const.CA) * NF
    )  # qg

    const.update_colors(4)

    np.testing.assert_allclose(const.CA, 4.0)
    gS1 = ad_as2.gamma_singlet(3, NF, sx_n3)
    np.testing.assert_allclose(
        gS1[1, 0],
        (
            973.0 / 432.0 * const.CF
            + (2801.0 / 5400.0 - 7.0 * np.pi**2 / 9.0) * const.CA
            + 61.0 / 54.0 * NF
        )
        * const.CF,
    )  # gq
    np.testing.assert_allclose(
        gS1[1, 1],
        (
            (-79909.0 / 3375.0 + 194.0 * np.pi**2 / 45.0 - 8.0 * h.constants.zeta3)
            * const.CA**2
            - 967.0 / 270.0 * const.CA * NF
            + 541.0 / 216.0 * const.CF * NF
        ),
        rtol=6e-7,
    )  # gg
    gS1 = ad_as2.gamma_singlet(4, NF, sx_n4)
    np.testing.assert_allclose(
        gS1[0, 1], (-56317.0 / 18000.0 * const.CF + 16387.0 / 9000.0 * const.CA) * NF
    )  # qg
    const.update_colors(3)
