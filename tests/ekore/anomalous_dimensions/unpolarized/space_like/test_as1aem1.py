# Test O(as1aem1) splitting functions
import numpy as np
import pytest

import ekore.anomalous_dimensions.unpolarized.space_like as ad
from eko import constants
from ekore import harmonics as h


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    cache = h.cache.reset()
    np.testing.assert_almost_equal(ad.as1aem1.gamma_nsm(N, cache), 0, decimal=4)


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    cache = h.cache.reset()
    for NF in range(2, 6 + 1):
        NU = constants.uplike_flavors(NF)
        ND = NF - NU
        np.testing.assert_almost_equal(
            constants.eu2 * ad.as1aem1.gamma_qg(N, NU, cache)
            + constants.ed2 * ad.as1aem1.gamma_qg(N, ND, cache)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phg(N)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gg(),
            0,
        )
    with pytest.raises(NotImplementedError):
        constants.uplike_flavors(7)


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    cache = h.cache.reset()
    for NF in range(2, 6 + 1):
        NU = constants.uplike_flavors(NF)
        ND = NF - NU
        np.testing.assert_almost_equal(
            constants.eu2 * ad.as1aem1.gamma_qph(N, NU, cache)
            + constants.ed2 * ad.as1aem1.gamma_qph(N, ND, cache)
            + ad.as1aem1.gamma_phph(NF)
            + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gph(N),
            0,
        )


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    cache = h.cache.reset()
    np.testing.assert_almost_equal(
        ad.as1aem1.gamma_nsp(
            N,
            cache,
        )
        + ad.as1aem1.gamma_gq(N, cache)
        + ad.as1aem1.gamma_phq(N, cache),
        0,
        decimal=4,
    )
