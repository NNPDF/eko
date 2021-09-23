# -*- coding: utf-8 -*-
"""This module contains the |N3LO| |OME| for the matching conditions in the |VFNS| the experssions are taken from :cite:`Bierenbaum_2009`"""
import numba as nb
import numpy as np
from ...anomalous_dimensions import harmonics
from . import g_functions as gf


zeta2 = harmonics.zeta2
zeta3 = harmonics.zeta3
zeta4 = harmonics.zeta4
zeta5 = harmonics.zeta5
li4half = 0.517479
li5half = 0.508401


@nb.njit("c16(c16,c16[:],c16[:],u4,f8)", cache=True)
def A_gg_3(n, sx, spx, nf, L):
    S1, S2, S3, S4, S5 = sx[0], sx[1], sx[2], sx[3], sx[4]
    Sp1p, Sp2p, Sp3p, Sp4p, Sp5p = spx[0], spx[1], spx[2], spx[3], spx[4]
    return 0.0
