# -*- coding: utf-8 -*-
# Test NNLO anomalous dims
import numpy as np

from eko.matching_conditions.nlo import A_gh_1, A_hh_1, A_ns_1, A_singlet_1
from eko.anomalous_dimensions import harmonics


def test_A_1_intrinsic():
    # gluon momentum conservation
    L = 0.0
    N = 2
    sx = np.array([harmonics.harmonic_S1(N)])
    ahh = A_hh_1(N,sx,L)
    agh = A_gh_1(N,L)
    np.testing.assert_allclose(ahh + agh, 0.0, atol=1e-8)

#  Everthing is proportional to L, not test useful
# def test_A_1():
#     L = 0.0
#     N = 1
#     sx = np.array([harmonics.harmonic_S1(N)])
#     np.testing.assert_allclose(A_ns_1(N, sx, L), 0.0, atol=3e-7)

#     # get singlet sector
#     N = 2
#     sx = np.array([harmonics.harmonic_S1(N)])
#     aS1 = A_singlet_1(N, sx, L)

#     # gluon momentum conservation
#     np.testing.assert_allclose(aS1[0, 1] + aS1[1, 1], 0.0, rtol=1e-6)
#     # quark momentum conservation
#     np.testing.assert_allclose(aS1[0, 0] + aS1[1, 0], 0.0, atol=3e-7)

#     assert aS1.shape == (2, 2)
