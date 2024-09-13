"""Benchmark the Mellin transforms against PEGASUS."""

import numpy as np
import pytest

import ekore.harmonics as h
from eko.constants import zeta2


@pytest.mark.isolated
def benchmark_melling_g3_pegasus():
    for N in [1, 2, 3, 4, 1 + 1j, 1 - 1j, 2 + 1j, 3 + 1j]:
        check_melling_g3_pegasus(N)


def check_melling_g3_pegasus(N):
    S1 = h.S1(N)
    N1 = N + 1.0
    N2 = N + 2.0
    N3 = N + 3.0
    N4 = N + 4.0
    N5 = N + 5.0
    N6 = N + 6.0
    S11 = S1 + 1.0 / N1
    S12 = S11 + 1.0 / N2
    S13 = S12 + 1.0 / N3
    S14 = S13 + 1.0 / N4
    S15 = S14 + 1.0 / N5
    S16 = S15 + 1.0 / N6

    SPMOM = (
        1.0000 * (zeta2 - S1 / N) / N
        - 0.9992 * (zeta2 - S11 / N1) / N1
        + 0.9851 * (zeta2 - S12 / N2) / N2
        - 0.9005 * (zeta2 - S13 / N3) / N3
        + 0.6621 * (zeta2 - S14 / N4) / N4
        - 0.3174 * (zeta2 - S15 / N5) / N5
        + 0.0699 * (zeta2 - S16 / N6) / N6
    )
    np.testing.assert_allclose(h.g_functions.mellin_g3(N, S1), SPMOM)
