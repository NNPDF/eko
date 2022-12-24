# Test LO Polarised splitting functions 
import numpy as np

import eko.anomalous_dimensions.asp1 as ad_asp1
from eko import harmonics


NF = 5


def test_number_conservation():
    # number
    N = complex(1.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(ad_asp1.gamma_pns(N, s1), 0)


def test_quark_momentum_conservation():
    # quark momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad_asp1.gamma_pns(N, s1) + ad_asp1.gamma_pgq(N),
        0,
    )


def test_gluon_momentum_conservation():
    # gluon momentum
    N = complex(2.0, 0.0)
    s1 = harmonics.S1(N)
    np.testing.assert_almost_equal(
        ad_asp1.gamma_pqg(N, NF) + ad_asp1.gamma_pgg(N, s1, NF), 0
    )


def test_gamma_qg_0():
    N = complex(1.0, 0.0)
    res = complex(-20.0 / 3.0, 0.0)
    np.testing.assert_almost_equal(ad_asp1.gamma_pqg(N, NF), res)


def test_gamma_gq_0():
    N = complex(0.0, 1.0)
    res = complex(4.0, -4.0) / 3.0
    np.testing.assert_almost_equal(ad_asp1.gamma_pgq(N), res)


def test_gamma_gg_0():
    N = complex(0.0, 1.0)
    s1 = harmonics.S1(N)
    res = complex(5.195725159621, 10.52008856962)
    np.testing.assert_almost_equal(ad_asp1.gamma_pgg(N, s1, NF), res)














# #declare parameter values
# order= (1, 0)
# n = complex(1.0, 1.0)
# nf= 4
# p = False
# a1= 2
# a0 = 1
# #modes used
# mode0=br.non_singlet_pids_map["ns+"]
# mode1=0

# #using anomalous dimensions singlet, make that matrix 
# gamma_s= gamma_singlet(order, n, nf, p)
# #make the evolution kernel with starting and ending energy 
# j00= np.log(a1 / a0) / beta.beta_qcd((2, 0), nf)
# lo_ex = ad.exp_singlet(gamma_s[0] * j00)
# print (lo_ex)
# ker_s = lo_ex
# print(np.shape(ker_s))
# k = 0 if mode0 == 100 else 1 
# l = 0 if mode1 == 100 else 1
# print(k)
# print(l)
# select_el= ker_s[k]
# print (select_el)

# spec = [
#     ("is_singlet", nb.boolean),
#     ("is_log", nb.boolean),
#     ("logx", nb.float64),
#     ("u", nb.float64),
# ]

