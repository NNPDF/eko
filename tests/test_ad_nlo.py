# -*- coding: utf-8 -*-
# Test LO splitting functions
import numpy as np

from eko.constants import Constants
import eko.anomalous_dimensions.nlo as ad_nlo

constants = Constants()
CA = constants.CA
CF = constants.CF
NF = 5


def test_gamma_ns_1():
    # reference values are obtained from MMa
    np.testing.assert_allclose(ad_nlo.gamma_nsm_1(1, NF, CA, CF), 0, atol=2e-6)
    np.testing.assert_allclose(
        ad_nlo.gamma_nsp_1(2, NF, CA, CF),
        (-112 * CF + 376 * CA - 64 * NF) * CF / 27
    )
    # singlet
    np.testing.assert_allclose(
        ad_nlo.gamma_qg_1(2, NF, CA, CF),
        (-74 * CF - 35 * CA) * NF / 27
    )
