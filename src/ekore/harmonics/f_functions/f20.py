# pylint: skip-file
"""This module contains implemtation of F20.

When using it, please cite :cite:`Blumlein:2009ta`.
Mellin transform is defined with the convention x^(N).
"""
import numba as nb
import numpy as np


@nb.njit(cache=True)
def F20(n, Sm1, Sm2, Sm3):
    """Mellin transform of eq 9.5 :cite:`Blumlein:2009ta`"""
    return 2.3148148148148148e-7 * (
        1.8204889223408133e7 / np.power(-1.0, 1.0 * n)
        - 650000.0 / (2.0 + n)
        - 382800.0 / np.power(3.0 + n, 3)
        - 254120.0 / np.power(3.0 + n, 2)
        - 109375.0 / (3.0 + n)
        + 600000.0 / np.power(4.0 + n, 3)
        - 4.32e7 / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        - (3.6e7 * n) / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        - (1.08e7 * np.power(n, 2)) / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        - (1.2e6 * np.power(n, 3)) / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3))
        + 760000.0 / np.power(4.0 + n, 2)
        - 1.52e7 / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2))
        - (9.12e6 * n) / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2))
        - (1.52e6 * np.power(n, 2)) / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2))
        - 172800.0 / np.power(5.0 + n, 3)
        - 69120.0 / np.power(5.0 + n, 2)
        + 97200.0 / np.power(6.0 + n, 3)
        - 1.5614208e9
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - (2.1275136e9 * n)
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - (1.236384e9 * np.power(n, 2))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - (3.919104e8 * np.power(n, 3))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - (7.11504e7 * np.power(n, 4))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - (6.9984e6 * np.power(n, 5))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        - (291600.0 * np.power(n, 6))
        / (np.power(2.0 + n, 3) * np.power(4.0 + n, 3) * np.power(6.0 + n, 3))
        + 65880.0 / np.power(6.0 + n, 2)
        - 5.164992e7
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        - (5.059584e7 * n)
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        - (1.897344e7 * np.power(n, 2))
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        - (3.16224e6 * np.power(n, 3))
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        - (197640.0 * np.power(n, 4))
        / (np.power(2.0 + n, 2) * np.power(4.0 + n, 2) * np.power(6.0 + n, 2))
        - 179334.0 / (8.0 + 6.0 * n + np.power(n, 2))
        - (59778.0 * n) / (8.0 + 6.0 * n + np.power(n, 2))
        - 165888.0 / (15.0 + 8.0 * n + np.power(n, 2))
        - (41472.0 * n) / (15.0 + 8.0 * n + np.power(n, 2))
        + (1.2280111e7 * Sm1) / np.power(-1.0, 1.0 * n)
        + (7.81412e6 * Sm2) / np.power(-1.0, 1.0 * n)
        + (3.6228e6 * Sm3) / np.power(-1.0, 1.0 * n)
        + (
            np.power(-1.0, 1.0 - 1.0 * n)
            * (2.3717031e7 + 3.2374342e7 * n + 1.2280111e7 * np.power(n, 2))
            * np.power(-1.0, n)
        )
        / np.power(1.0 + n, 3)
    )
