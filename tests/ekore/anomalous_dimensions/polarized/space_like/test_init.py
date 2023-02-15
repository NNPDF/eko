import numpy as np

import ekore.anomalous_dimensions.polarized.space_like as ad_ps


def test_init():
    for k in range(1, 3 + 1):
        assert ad_ps.gamma_ns((k, 0), 10101, 2.0, 4).shape == (k,)
        assert ad_ps.gamma_singlet((k, 0), 2.0, 4).shape == (k, 2, 2)
