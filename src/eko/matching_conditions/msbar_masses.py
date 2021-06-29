# -*- coding: utf-8 -*-
r"""
This module contains the RGE for the ms bar masses
"""
import numpy as np
import scipy.integrate as integrate

from ..beta import beta, b
from ..gamma import gamma

# quad ker
def rge(a, b_vec, g_vec):
    # minus sign goes away
    fgamma = np.sum([a ** k * b for k, b in enumerate(g_vec)])
    fbeta = a * np.sum([a ** k * b for k, b in enumerate(b_vec)])
    return fgamma / fbeta


def msbar_exact(a0, a1, order, nf):
    """MSBar RGE exact"""
    b_vec = [beta(0, nf)]
    g_vec = [gamma(0, nf)]
    if order >= 1:
        b_vec.append(beta(1, nf))
        g_vec.append(gamma(1, nf))
    if order >= 2:
        b_vec.append(beta(2, nf))
        g_vec.append(gamma(2, nf))
    res = integrate.quad(
        rge,
        a0,
        a1,
        args=(b_vec, g_vec),
        epsabs=1e-12,
        epsrel=1e-5,
        limit=100,
        full_output=1,
    )
    val, _ = res[:2]
    return np.exp(val)


def msbar_expanded(a0, a1, order, nf):
    """MSBar RGE exapnded"""
    b0 = beta(0, nf)
    c0 = gamma(0, nf) / b0
    ev_mass = a1 / a0 * np.exp(c0)
    num = 1.0
    den = 1.0
    if order >= 1:
        b1 = b(1, nf)
        c1 = gamma(1, nf) / b0
        r = c1 - b1 * c0
        num += a1 * r
        den += a0 * r
    if order >= 2:
        b2 = b(1, nf)
        c2 = gamma(2, nf) / b0
        r = (c2 - c1 * b1 - b2 * c0 + b1 ** 2 * c0 + (c1 - b1 * b0) ** 2) / 2.0
        num += a1 ** 2 * r
        den += a0 ** 2 * r
    return ev_mass * num / den
