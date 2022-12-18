# Test O(as1aem1) splitting functions
import numpy as np

from eko import anomalous_dimensions as ad
from eko import constants
from eko import harmonics as h


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = h.sx(N, 3)
    sx_ns_qed = h.compute_additional_sx_cache(N, False, True, sx)
    for NF in range(2, 6 + 1):
        np.testing.assert_almost_equal(
            ad.aem2.gamma_nsmu(N, NF, sx, sx_ns_qed), 0, decimal=4
        )
        np.testing.assert_almost_equal(
            ad.aem2.gamma_nsmd(N, NF, sx, sx_ns_qed), 0, decimal=4
        )


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    sx = h.sx(N, 2)
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
    sx = h.sx(N, 3)
    sx_ns_qed = h.compute_additional_sx_cache(N, False, True, sx)
    NF = 6
    NU = constants.uplike_flavors(NF)
    ND = NF - NU
    np.testing.assert_almost_equal(
        ad.aem2.gamma_nspu(N, NF, sx, sx_ns_qed)
        + constants.eu2 * ad.aem2.gamma_ps(N, NU)
        + constants.ed2 * ad.aem2.gamma_ps(N, ND)
        + ad.aem2.gamma_phu(N, NF, sx),
        0,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        ad.aem2.gamma_nspd(N, NF, sx, sx_ns_qed)
        + constants.eu2 * ad.aem2.gamma_ps(N, NU)
        + constants.ed2 * ad.aem2.gamma_ps(N, ND)
        + ad.aem2.gamma_phd(N, NF, sx),
        0,
        decimal=4,
    )
