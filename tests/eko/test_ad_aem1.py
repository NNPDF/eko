# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

from eko import anomalous_dimensions as ad


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    s1 = ad.harmonics.S1(N)
    np.testing.assert_almost_equal(ad.aem1.gamma_ns(N, s1), 0)


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = ad.harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad.aem1.gamma_ns(N, s1) + ad.aem1.gamma_phq(N),
        0,
    )


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    for NF in range(2, 6 + 1):
        np.testing.assert_almost_equal(
            ad.aem1.gamma_qph(N, NF) + ad.aem1.gamma_phph(NF), 0
        )
