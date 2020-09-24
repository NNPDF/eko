# -*- coding: utf-8 -*-


import numpy as np

import numba as nb


@nb.njit
def geomspace(start, end, num):
    return np.exp(np.linspace(np.log(start), np.log(end), num))
