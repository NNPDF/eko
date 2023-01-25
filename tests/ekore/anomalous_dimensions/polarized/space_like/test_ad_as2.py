# Test LO Polarized splitting functions
import numpy as np

import ekore.anomalous_dimensions.polarized.space_like.as2 as as2
from ekore import harmonics

nf = 5


def test_qg_helicity_conservation():
    N = complex(1.0, 0.0)
    sx = harmonics.sx(N, max_weight=4)
    np.testing.assert_almost_equal(as2.gamma_qg(N, nf, sx), 0)
