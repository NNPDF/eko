# -*- coding: utf-8 -*-
import numpy as np

from eko.kernels import utils

def test_geomspace():
    for start in [1,2]:
        for end in [3,4]:
            for n in [5,10]:
                np.testing.assert_allclose(utils.geomspace(start,end,n),np.geomspace(start,end,n))
