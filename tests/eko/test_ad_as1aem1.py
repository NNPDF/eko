# -*- coding: utf-8 -*-
# Test O(as1aem1) splitting functions
import numpy as np
from test_ad_nnlo import get_sx

from eko import anomalous_dimensions as ad
from eko import constants

NF = 5
ND = 3
NU = 2


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(ad.as1aem1.gamma_nsm(N, sx), 0, decimal=4)


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    for NF, ND in ((5, 3), (6, 3)):
        NU = NF - ND
        np.testing.assert_almost_equal(
            constants.eu2 * ad.as1aem1.gamma_qg(N, NU, sx)
            + constants.ed2 * ad.as1aem1.gamma_qg(N, ND, sx)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phg(N)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gg(),
            0,
        )


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    for NF, ND in ((5, 3), (6, 3)):
        NU = NF - ND
        np.testing.assert_almost_equal(
            constants.eu2 * ad.as1aem1.gamma_qph(N, NU, sx)
            + constants.ed2 * ad.as1aem1.gamma_qph(N, ND, sx)
            + ad.as1aem1.gamma_phph(NF)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gph(N),
            0,
        )


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(
        ad.as1aem1.gamma_nsp(N, sx)
        + ad.as1aem1.gamma_gq(N, sx)
        + ad.as1aem1.gamma_phq(N, sx),
        0,
        decimal=4,
    )
