import numpy as np
from scipy.integrate import quad

from ekore import harmonics as h

from . import sx as hsx


def test_lm1pm2():
    def mellin_lm1pm2(x, k, N):
        return x ** (N - 1) * (1 - x) ** 2 * np.log(1 - x) ** k

    Ns = [1.0, 1.5, 2.0, 2.34, 56.789]
    for N in Ns:
        sx = hsx(N, 4)

        ref_values = {
            1: h.log_functions.lm11m2(N, sx[0]),
            2: h.log_functions.lm12m2(N, sx[0], sx[1]),
            3: h.log_functions.lm13m2(N, sx[0], sx[1], sx[2]),
            4: h.log_functions.lm14m2(N, sx[0], sx[1], sx[2], sx[3]),
        }

        for k, ref in ref_values.items():
            test_value = quad(mellin_lm1pm2, 0, 1, args=(k, N))[0]
            np.testing.assert_allclose(test_value, ref)


def test_lm1pm1():
    # test mellin transformation with some random N values
    def mellin_lm1pm1(x, k, N):
        return x ** (N - 1) * (1 - x) * np.log(1 - x) ** k

    Ns = [1.0, 1.5, 2.0, 2.34, 56.789]
    for N in Ns:
        sx = hsx(N, 4)

        ref_values = {
            1: h.log_functions.lm11m1(N, sx[0]),
            2: h.log_functions.lm12m1(N, sx[0], sx[1]),
            3: h.log_functions.lm13m1(N, sx[0], sx[1], sx[2]),
            4: h.log_functions.lm14m1(N, sx[0], sx[1], sx[2], sx[3]),
        }

        for k in [1, 2, 3, 4]:
            test_value = quad(mellin_lm1pm1, 0, 1, args=(k, N))[0]
            np.testing.assert_allclose(test_value, ref_values[k])


def test_lm1p():
    # test mellin transformation with some random N values
    def mellin_lm1p(x, k, N):
        return x ** (N - 1) * np.log(1 - x) ** k

    Ns = [1.0, 1.5, 2.0, 2.34, 56.789]
    for N in Ns:
        sx = hsx(N, 5)

        ref_values = {
            1: h.log_functions.lm11(N, sx[0]),
            2: h.log_functions.lm12(N, sx[0], sx[1]),
            3: h.log_functions.lm13(N, sx[0], sx[1], sx[2]),
            4: h.log_functions.lm14(N, sx[0], sx[1], sx[2], sx[3]),
            5: h.log_functions.lm15(N, sx[0], sx[1], sx[2], sx[3], sx[4]),
        }

        for k in [1, 3, 4, 5]:
            test_value = quad(mellin_lm1p, 0, 1, args=(k, N))[0]
            np.testing.assert_allclose(test_value, ref_values[k])
