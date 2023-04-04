# it is possible to obtain result values for testing from vbertone/MELA repo

import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as2 as ad_as2

NF = 5

def test_nsp():
    Nlist = [1, complex(0, 1), complex(1,1)]
    res = [1.278774602, 4.904787164+58.022801851j, 0.910919333+22.344175662j]
    for i in range(3):
        np.testing.assert_almost_equal(
            ad_as2.gamma_nsp(Nlist[i], NF), res[i]
        )
def test_nsm():
    Nlist = [1, complex(0, 1), complex(1,1)]
    res = [1.585785642e-06, 1.527559249+56.736014509j, 1.312189640+22.270151563j]
    for i in range(3):
        np.testing.assert_almost_equal(
            ad_as2.gamma_nsm(Nlist[i], NF), res[i]
        )
def test_qqs():
    Nlist = [2, complex(0, 1), complex(1,1)]
    res = [15.802469135, -11.199999999-65.066666666j, -7.786666666-0.640000000j]
    for i in range(3):
        np.testing.assert_almost_equal(
            ad_as2.gamma_qqs(Nlist[i], NF), res[i]
        )
def test_qg():
    Nlist = [2, complex(0, 1), complex(1,1)]
    res = [4.394040436, 19.069864378-1.489605936j, -0.576926260+1.543864328j]
    for i in range(3):
        np.testing.assert_almost_equal(
            ad_as2.gamma_qg(Nlist[i], NF), res[i]
        )
def test_gq():
    Nlist = [2, complex(0, 1), complex(1,1)]
    res = [-307.172330861, 1098.565548102+120.400835601j, 561.338791208+1936.775511054j]
    for i in range(3):
        np.testing.assert_almost_equal(
            ad_as2.gamma_gq(Nlist[i], NF), res[i]
        )
def test_gg():
    Nlist = [2, complex(0, 1), complex(1,1)]
    res = [-43.940429621, -168.786803436-173.884948858j, 130.912306514+282.257962305j]
    for i in range(3):
        np.testing.assert_almost_equal(
            ad_as2.gamma_gg(Nlist[i], NF), res[i]
        )
