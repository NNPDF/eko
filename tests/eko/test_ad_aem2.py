# -*- coding: utf-8 -*-
# Test O(as1aem1) splitting functions
import numpy as np
from test_ad_nnlo import get_sx

from eko import anomalous_dimensions as ad
from eko import constants


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    sx = get_sx(N)
    for NF, ND in ((5, 3), (6, 3)):
        # NU = NF - ND
        # import pdb; pdb.set_trace()
        np.testing.assert_almost_equal(ad.aem2.gamma_nsmu(N, NF, sx), 0, decimal=4)
        np.testing.assert_almost_equal(ad.aem2.gamma_nsmd(N, NF, sx), 0, decimal=4)
