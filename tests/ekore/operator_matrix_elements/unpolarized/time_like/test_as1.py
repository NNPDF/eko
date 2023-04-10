r"""Test NLO |OME|."""

import numpy as np

import ekore.operator_matrix_elements.unpolarized.time_like.as1 as ome_as1


def test_A_hg():
    res = [
        complex(-752 / 75, 112 / 25),
        complex(-3268 / 4225, -24872 / 12675),
        complex(437984 / 950625, -4829264 / 2851875),
    ]
    for i, N, L in [
        (0, complex(1, 1), 1),
        (1, complex(2, 2), 2),
        (2, complex(3, 3), 3),
    ]:
        np.testing.assert_almost_equal(ome_as1.A_hg(N, L), res[i])
