# -*- coding: utf-8 -*-
# Test Mellin stuff
import numpy as np
from eko.Mellin import inverse_Mellin_transform,\
    get_path_Talbot,get_path_line

def test_inverse_Mellin_transform():
    """test inverse_Mellin_transform"""
    f = lambda N : 1./(N+1.)
    g = lambda x : x
    p,j = get_path_Talbot()
    for x in [.1,.3,.5,.7]:
        e = g(x)
        a = inverse_Mellin_transform(f,p,j,x,1e-2)
        assert np.abs(e-a[0]) < 1e-6
        assert np.abs(a[1]) < 1e-6

def test_get_path_dev():
    """test derivatives to path"""
    epss = [1e-2,1e-3,1e-4,1e-5]
    for pj in [get_path_Talbot(),get_path_Talbot(2),get_path_line(1),get_path_line(2,2)]:
        (p,j) = pj
        for t0 in [.3,.5,.7]:
            # compute numeric derivative + extrapolation
            num = []
            for eps in epss:
                a = (p(t0 + eps) - p(t0 - eps))/(2. * eps)
                num = np.append(num, a)
            f = np.polyfit(epss,num,2)
            ex = j(t0)
            assert np.abs(ex - f[2]) < 1e-4
