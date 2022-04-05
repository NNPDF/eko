# -*- coding: utf-8 -*-
# Test O(as1aem1) splitting functions
import numpy as np
from test_ad_as3 import get_sx

from eko import anomalous_dimensions as ad
from eko import constants


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = get_sx(N)
    for NF in range(2, 6 + 1):
        # NU = NF - ND
        np.testing.assert_almost_equal(ad.aem2.gamma_nsmu(N, NF, sx), 0, decimal=4)
        np.testing.assert_almost_equal(ad.aem2.gamma_nsmd(N, NF, sx), 0, decimal=4)


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    for NF in range(2, 6 + 1):
        NU = constants.uplike_flavors(NF)
        ND = NF - NU
        np.testing.assert_almost_equal(
            constants.eu2 * ad.aem2.gamma_uph(N, NU, sx)
            + constants.ed2 * ad.aem2.gamma_dph(N, ND, sx)
            + ad.aem2.gamma_phph(N, NF),
            0,
        )


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    NF = 6
    NU = constants.uplike_flavors(NF)
    ND = NF - NU
    # import pdb; pdb.set_trace()
    np.testing.assert_almost_equal(
        ad.aem2.gamma_nspu(N, NF, sx)
        + constants.eu2 * ad.aem2.gamma_ps(N, NU)
        + constants.ed2 * ad.aem2.gamma_ps(N, ND)
        + ad.aem2.gamma_phu(N, NF, sx),
        0,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        ad.aem2.gamma_nspd(N, NF, sx)
        + constants.eu2 * ad.aem2.gamma_ps(N, NU)
        + constants.ed2 * ad.aem2.gamma_ps(N, ND)
        + ad.aem2.gamma_phd(N, NF, sx),
        0,
        decimal=4,
    )
