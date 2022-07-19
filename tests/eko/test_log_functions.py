# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad

from eko import harmonics as h


def test_lm1pm1():
    # test mellin transformation with some random N values
    Ns = [12.345, 81.113, 27.787]
    for N in Ns:
        sx = h.sx(N, 3)

        ref_values = {
            1: h.log_functions.lm11m1(N, sx[0]),
            2: h.log_functions.lm12m1(N, sx[0], sx[1]),
            3: h.log_functions.lm13m1(N, sx[0], sx[1], sx[2]),
        }

        def mellin_lm1pm1(x, k):
            return x ** (N - 1) * (1 - x) * np.log(1 - x) ** k

        for k in [1, 2, 3]:
            test_value = quad(mellin_lm1pm1, 0, 1, args=(k))[0]
            np.testing.assert_allclose(test_value, ref_values[k], atol=5e-4)


def test_lm1p():
    # test mellin transformation with some random N values
    Ns = [65.780, 56.185, 94.872]
    for N in Ns:
        sx = h.sx(N, 5)

        ref_values = {
            1: h.log_functions.lm11(N, sx[0]),
            3: h.log_functions.lm13(N, sx[0], sx[1], sx[2]),
            4: h.log_functions.lm14(N, sx[0], sx[1], sx[2], sx[3]),
            5: h.log_functions.lm15(N, sx[0], sx[1], sx[2], sx[3], sx[4]),
        }

        def mellin_lm1p(x, k):
            return x ** (N - 1) * np.log(1 - x) ** k

        for k in [1, 3, 4, 5]:
            test_value = quad(mellin_lm1p, 0, 1, args=(k))[0]
            np.testing.assert_allclose(test_value, ref_values[k])
