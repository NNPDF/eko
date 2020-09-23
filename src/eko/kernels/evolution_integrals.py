# -*- coding: utf-8 -*-

import numpy as np


from eko import strong_coupling as sc


def j00(a1, a0, nf):
    return np.log(a1 / a0) / sc.beta(0, nf)


def j11_exact(a1, a0, nf):
    beta_1 = sc.beta(1, nf)
    b1 = sc.b(1, nf)
    return (1.0 / beta_1) * np.log((1.0 + a1 * b1) / (1.0 + a0 * b1))


def j11_expanded(a1, a0, nf):
    return 1.0 / sc.beta(0, nf) * (a1 - a0)


def j01_exact(a1, a0, nf):
    return j00(a1, a0, nf) - sc.b(1, nf) * j11_exact(a1, a0, nf)


def j01_expanded(a1, a0, nf):
    return j00(a1, a0, nf) - sc.b(1, nf) * j11_expanded(a1, a0, nf)
