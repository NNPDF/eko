# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np

import eko.anomalous_dimensions.as4 as ad_as4

# TODO: move this method  out of matching conditions if it will be used
# also here...
from eko.matching_conditions.operator_matrix_element import compute_harmonics_cache

NF = 5


def test_quark_number_conservation():
    N = 1
    sx_cache = compute_harmonics_cache(N, 3, False)

    # (ns,s)
    # the exact expession has an nonphysical pole at N=1, see also :cite:`Moch:2017uml`
    # and :cite:`Davies:2016jie` eq 3.5 where the \nu term is present.
    # This should cancel when doing the limit, since the given moment for N=1 is 0

    # np.testing.assert_allclose(ad_as4.gamma_nsv(N, NF, sx_cache), 0, rtol=3e-7)

    # (ns,-)
    # (ns,+), (ns,-) common part
    np.testing.assert_allclose(ad_as4.gNSp.A_3(N, sx_cache), 0, atol=0.004)
    # (ns,-) B3m = B3p + delta B3,
    # This difference is due to the parametrization of B3p
    # TODO: can we improve this?
    np.testing.assert_allclose(ad_as4.gNSm.B_3m(N, sx_cache), -0.0184639, atol=6e-6)

    # nf^3 part
    np.testing.assert_allclose(ad_as4.gNSp.gamma_ns_nf3(N, sx_cache), 0, atol=3e-10)
    # nf^2 part
    np.testing.assert_allclose(ad_as4.gNSm.gamma_nsm_nf2(N, sx_cache), 0, atol=3e-3)
