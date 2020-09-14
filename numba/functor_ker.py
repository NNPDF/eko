# -*- coding: utf-8 -*-
import timeit
import numpy as np
import scipy.integrate as sint
import numba as nb

# import scipy.special as sspec
# import eko

n_number = 25


def ker(z):
    _vals = np.random.random_sample(100)
    return np.log(z)


ker_nb = nb.njit(ker)


def a(to):
    return sint.quad(ker, 1, to)


def a_nb(to):
    return sint.quad(ker_nb, 1, to)


def b(mx):
    b = []
    for to in range(2, mx):
        b.append(a(to))
    return b


def b_nb(mx):
    b = []
    for to in range(2, mx):
        b.append(a_nb(to))
    return b


print(timeit.repeat("b(100)", globals=globals(), number=n_number))
print(timeit.repeat("b_nb(100)", globals=globals(), number=n_number))


def ker_obj(z):
    _vals = np.random.random_sample(100)
    return np.log(z)


ker_obj_nb = nb.njit(ker_obj)


class KerCls:
    def __init__(self, fnc):
        self.fnc = fnc

    def __call__(self, z):
        return self.fnc(z) + 2 * self.fnc(z)


kker_obj = KerCls(ker_obj)
kker_obj_nb = KerCls(ker_obj_nb)


def a_obj(to):
    return sint.quad(kker_obj, 1, to)


def a_obj_nb(to):
    return sint.quad(kker_obj_nb, 1, to)


def b_obj(mx):
    b = []
    for to in range(2, mx):
        b.append(a_obj(to))
    return b


def b_obj_nb(mx):
    b = []
    for to in range(2, mx):
        b.append(a_obj_nb(to))
    return b


print(timeit.repeat("b_obj(100)", globals=globals(), number=n_number))
print(timeit.repeat("b_obj_nb(100)", globals=globals(), number=n_number))
