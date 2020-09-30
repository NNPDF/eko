# -*- coding: utf-8 -*-
import numpy as np

from eko.kernels import evolution_integrals as ei

def test_zero():
    """No evolution results in exp(0)"""
    nf = 3
    for fnc in [ei.j00, ei.j01_exact,ei.j01_expanded,ei.j11_exact,ei.j11_expanded]:
        np.testing.assert_allclose(fnc(1,1,nf),0)
