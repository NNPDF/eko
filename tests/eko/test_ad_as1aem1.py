# -*- coding: utf-8 -*-
# Test O(as1aem1) splitting functions
import numpy as np
import pytest

from eko import anomalous_dimensions as ad
from eko import constants
from eko import harmonics as h


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = h.sx(N, 3)
    sx_ns_qed = h.compute_qed_ns_cache(N, sx[0])
    np.testing.assert_almost_equal(ad.as1aem1.gamma_nsm(N, sx, sx_ns_qed), 0, decimal=4)


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    sx = h.sx(N, 2)
    for NF in range(2, 6 + 1):
        NU = constants.uplike_flavors(NF)
        ND = NF - NU
        np.testing.assert_almost_equal(
            constants.eu2 * ad.as1aem1.gamma_qg(N, NU, sx)
            + constants.ed2 * ad.as1aem1.gamma_qg(N, ND, sx)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phg(N)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gg(),
            0,
        )
    with pytest.raises(NotImplementedError):
        constants.uplike_flavors(7)


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    sx = h.sx(N, 2)
    for NF in range(2, 6 + 1):
        NU = constants.uplike_flavors(NF)
        ND = NF - NU
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
    sx = h.sx(N, 3)
    sx_ns_qed = h.compute_qed_ns_cache(N, sx[0])
    np.testing.assert_almost_equal(
        ad.as1aem1.gamma_nsp(N, sx, sx_ns_qed)
        + ad.as1aem1.gamma_gq(N, sx)
        + ad.as1aem1.gamma_phq(N, sx),
        0,
        decimal=4,
    )
