# pylint: skip-file
"""This module contains implemtation of F21.

When using it, please cite :cite:`Blumlein:2009ta`.
Mellin transform is defined with the convention x^(N).
"""
import numba as nb
import numpy as np


@nb.njit(cache=True)
def F21(n, Sm1, Sm2, Sm3):
    """Mellin transform of eq 9.4 of :cite:`Blumlein:2009ta`"""
    return 2.3148148148148148e-7 * (
        2.383072892806571e7 / np.power(-1.0, 1.0 * n)
        + 970000.0 / (2.0 + n)
        - 922800.0 / np.power(3.0 + n, 3)
        - 524120.0 / np.power(3.0 + n, 2)
        - 210625.0 / (3.0 + n)
        - 1.56e6 / np.power(4.0 + n, 3)
        + 1.1232e8 / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        + (9.36e7 * n) / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        + (2.808e7 * np.power(n, 2)) / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        + (3.12e6 * np.power(n, 3)) / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        - 1.4e6 / np.power(4.0 + n, 2)
        + 2.8e7 / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2))
        + (1.68e7 * n) / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2))
        + (2.8e6 * np.power(n, 2)) / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2))
        - 172800.0 / np.power(5.0 + n, 3)
        - 69120.0 / np.power(5.0 + n, 2)
        - 442800.0 / np.power(6.0 + n, 3)
        + 7.1131392e9
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + (9.6920064e9 * n)
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + (5.632416e9 * np.power(n, 2))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + (1.7853696e9 * np.power(n, 3))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + (3.241296e8 * np.power(n, 4))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + (3.18816e7 * np.power(n, 5))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + (1.3284e6 * np.power(n, 6))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - 204120.0 / np.power(6.0 + n, 2)
        + 1.6003008e8
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        + (1.5676416e8 * n)
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        + (5.878656e7 * np.power(n, 2))
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        + (9.79776e6 * np.power(n, 3))
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        + (612360.0 * np.power(n, 4))
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        + 428166.0 / (8.0 + 6.0 * n + np.power(n, 2))
        + (142722.0 * n) / (8.0 + 6.0 * n + np.power(n, 2))
        - 165888.0 / (15.0 + 8.0 * n + np.power(n, 2))
        - (41472.0 * n) / (15.0 + 8.0 * n + np.power(n, 2))
        + (1.4001361e7 * Sm1) / np.power(-1.0, 1.0 * n)
        + (1.024412e7 * Sm2) / np.power(-1.0, 1.0 * n)
        + (6.3228e6 * Sm3) / np.power(-1.0, 1.0 * n)
        + (
            np.power(-1.0, 1.0 - 1.0 * n)
            * (3.0568281e7 + 3.8246842e7 * n + 1.4001361e7 * np.power(n, 2))
            * np.power(-1.0, n)
        )
        / np.power(1.0 + n, 3)
    )
