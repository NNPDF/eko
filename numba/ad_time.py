# -*- coding: utf-8 -*-
import timeit
import sys
import numpy as np

from eko import ekomath

def f():
    return [ekomath.cern_polygamma(1. + k*1j,0) for k in range(10)]

print(timeit.repeat(f, number=1, repeat=5))