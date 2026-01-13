"""Testing values obtained from |MELA| functions."""

import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as3 as ad_as3
from ekore.harmonics import cache as c

NF = 5


def test_nsp():
    Nlist = [1, complex(0, 1), complex(1, 1)]
    res = [
        60.770703587,
        -234.850149975 + 1093.204118679j,
        -266.436691842 + 22.506418507j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_nsp(Nlist[i], NF, cache), res[i])


def test_nsm():
    Nlist = [1, complex(0, 1), complex(1, 1)]
    res = [0.000593360, -400.032246675 + 895.182550001j, -239.655009655 + 47.010480494j]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_nsm(Nlist[i], NF, cache), res[i])


def test_nsv():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [114.338228278, 97.529253158 - 453.699848424j, 237.589718980 - 175.574012201j]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_nsv(Nlist[i], NF, cache), res[i])


def test_qq():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        -184.274140748,
        888.259743291 - 3850.903826041j,
        -521.507856510 + 1156.486663262j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_qq(Nlist[i], NF, cache), res[i])


def test_qg():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        -61.228237020,
        -583.289222276 + 122.793826705j,
        -194.069551898 + 260.313594964j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_qg(Nlist[i], NF, cache), res[i])


def test_gq():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        2511.568156988,
        -12289.104690583 + 30792.411034276j,
        41742.218127251 + 68362.479056432j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_gq(Nlist[i], NF, cache), res[i])


def test_gg():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        612.286143736,
        -6040.795385224 - 13212.596652169j,
        14137.203400417 + 9336.761782887j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_gg(Nlist[i], NF, cache), res[i])
