# -*- coding: utf-8 -*-
import timeit
import numpy as np

# import numba as nb
# import scipy.integrate as sint
# import scipy.special as sspec
import eko
import eko.interpolation


def test_grid(n_low, n_mid, deg):
    xg = eko.interpolation.make_grid(n_low, n_mid, x_min=1e-1)
    bfd = eko.interpolation.InterpolatorDispatcher.from_dict(
        dict(interpolation_xgrid=xg, interpolation_polynomial_degree=deg)
    )
    t = timeit.timeit(lambda: [bf(1, np.log(1e-2)) for bf in bfd], number=5)
    return t / len(xg)


print("test distribution of points matters")
print(test_grid(10, 0, 1))
print(test_grid(5, 5, 1))
print(test_grid(0, 10, 1))
print()

print("test performance with n")
for n in [2, 5, 8, 15, 25]:
    print(2*n, test_grid(n, n, 1))
print()

print("test performance with deg")
for d in [1, 2, 3, 4, 5]:
    print(d, test_grid(10, 10, d))
print()
