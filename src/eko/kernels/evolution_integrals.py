# -*- coding: utf-8 -*-
"""
Integrals needed for the exact evolutions.
"""

import numpy as np

import numba as nb

from .. import beta


@nb.njit
def j00(a1, a0, nf):
    return np.log(a1 / a0) / beta.beta(0, nf)


@nb.njit
def j11_exact(a1, a0, nf):
    beta_1 = beta.beta(1, nf)
    b1 = beta.b(1, nf)
    return (1.0 / beta_1) * np.log((1.0 + a1 * b1) / (1.0 + a0 * b1))


@nb.njit
def j11_expanded(a1, a0, nf):
    return 1.0 / beta.beta(0, nf) * (a1 - a0)


@nb.njit
def j01_exact(a1, a0, nf):
    return j00(a1, a0, nf) - beta.b(1, nf) * j11_exact(a1, a0, nf)


@nb.njit
def j01_expanded(a1, a0, nf):
    return j00(a1, a0, nf) - beta.b(1, nf) * j11_expanded(a1, a0, nf)
