# Test LO splitting functions
import numpy as np

import ekore.anomalous_dimensions.unpolarized.space_like as ad_us
import ekore.harmonics as h
from eko import constants


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = h.sx(N, max_weight=1)
    np.testing.assert_almost_equal(ad_us.aem1.gamma_ns(N, sx), 0)


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    sx = h.sx(N, max_weight=1)
    np.testing.assert_almost_equal(
        ad_us.aem1.gamma_ns(N, sx) + ad_us.aem1.gamma_phq(N),
        0,
    )


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    for NF in range(2, 6 + 1):
        NU = constants.uplike_flavors(NF)
        ND = NF - NU
        np.testing.assert_almost_equal(
            constants.eu2 * ad_us.aem1.gamma_qph(N, NU)
            + constants.ed2 * ad_us.aem1.gamma_qph(N, ND)
            + ad_us.aem1.gamma_phph(NF),
            0,
        )
