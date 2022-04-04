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
    np.testing.assert_almost_equal(ad.as1aem1.gamma_nsm(N, NF, sx), 0, decimal=4)


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(
        2 * NU * constants.eu2 * ad.as1aem1.gamma_qg(N, NF, sx)
        + 2 * ND * constants.ed2 * ad.as1aem1.gamma_qg(N, NF, sx)
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phg(N)
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gg(),
        0,
    )


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    # import pdb; pdb.set_trace()
    np.testing.assert_almost_equal(
        2 * NU * constants.eu2 * ad.as1aem1.gamma_qph(N, NF, sx)
        + 2 * ND * constants.ed2 * ad.as1aem1.gamma_qph(N, NF, sx)
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phph()
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gph(N),
        0,
    )


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(
        ad.as1aem1.gamma_nsp(N, NF, sx)
        + ad.as1aem1.gamma_gq(N, NF, sx)
        + ad.as1aem1.gamma_phq(N, NF, sx),
        0,
        decimal=4,
    )


NF = 6
ND = 3
NU = 3


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(ad.as1aem1.gamma_nsm(N, NF, sx), 0, decimal=4)


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(
        2 * NU * constants.eu2 * ad.as1aem1.gamma_qg(N, NF, sx)
        + 2 * ND * constants.ed2 * ad.as1aem1.gamma_qg(N, NF, sx)
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phg(N)
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gg(),
        0,
    )


def test_photon_momentum_conservation():
    # photon momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    # import pdb; pdb.set_trace()
    np.testing.assert_almost_equal(
        2 * NU * constants.eu2 * ad.as1aem1.gamma_qph(N, NF, sx)
        + 2 * ND * constants.ed2 * ad.as1aem1.gamma_qph(N, NF, sx)
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_phph()
        + (NU * constants.eu2 + ND * constants.ed2) * ad.as1aem1.gamma_gph(N),
        0,
    )


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    sx = get_sx(N)
    np.testing.assert_almost_equal(
        ad.as1aem1.gamma_nsp(N, NF, sx)
        + ad.as1aem1.gamma_gq(N, NF, sx)
        + ad.as1aem1.gamma_phq(N, NF, sx),
        0,
        decimal=4,
    )
