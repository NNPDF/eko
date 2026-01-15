"""Testing values obtained from |MELA| functions and 1905.01310."""

import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as2 as ad_as2
from eko.constants import CA, CF, zeta2, zeta3
from ekore.harmonics import cache as c

NF = 5


# Thanks Yuxun Guo (@yuxunguo)
def n3(nf: int):
    """Implements 1905.01310 Eq. (A7)"""
    z2 = zeta2
    z3 = zeta3
    cf = CF
    ca = CA

    gTqq1 = (
        -5453 / 1800 * cf * nf
        + cf**2 * (-1693 / 48 + 24 * z2 - 16 * z3)
        + ca * cf * (459 / 8 - 86 * z2 / 3 + 8 * z3)
    )
    gTgq1 = ca * cf * (-39451 / 5400 - 14 * z2 / 3) + cf**2 * (
        -2977 / 432 + 28 * z2 / 3
    )
    gTqg1 = (
        -833 / 216 * cf * nf - 4 / 25 * nf**2 + ca * nf * (619 / 2700 + 28 * z2 / 15)
    )
    gTgg1 = (
        12839 / 5400 * cf * nf
        + ca * nf * (3803 / 1350 - 16 * z2 / 3)
        + ca**2 * (2158 / 675 + 52 * z2 / 15 - 8 * z3)
    )

    return np.array([[gTqq1, gTgq1 * 2 * nf], [gTqg1 / (2 * nf), gTgg1]])


def test_nsp():
    Nlist = [1, complex(0, 1), complex(1, 1)]
    res = [1.278774602, 4.904787164 + 58.022801851j, 0.910919333 + 22.344175662j]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as2.gamma_nsp(Nlist[i], NF, cache), res[i])


def test_nsm():
    Nlist = [1, complex(0, 1), complex(1, 1)]
    res = [1.585785642e-06, 1.527559249 + 56.736014509j, 1.312189640 + 22.270151563j]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as2.gamma_nsm(Nlist[i], NF, cache), res[i])


def test_qqs():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [15.802469135, -11.199999999 - 65.066666666j, -7.786666666 - 0.640000000j]
    for i in range(3):
        np.testing.assert_almost_equal(ad_as2.gamma_qqs(Nlist[i], NF), res[i])


def test_qg():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [4.394040436, 19.069864378 - 1.489605936j, -0.576926260 + 1.543864328j]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as2.gamma_qg(Nlist[i], NF, cache), res[i])


def test_gq():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        -307.172330861,
        1098.565548102 + 120.400835601j,
        561.338791208 + 1936.775511054j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as2.gamma_gq(Nlist[i], NF, cache), res[i])


def test_gg():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        -43.940429621,
        -168.786803436 - 173.884948858j,
        130.912306514 + 282.257962305j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as2.gamma_gg(Nlist[i], NF, cache), res[i])


def test_singlet_n3():
    cache = c.reset()
    # test against 1905.01310
    for nf in range(3, 6 + 1):
        np.testing.assert_allclose(
            ad_as2.gamma_singlet(3.0, nf, cache), n3(nf), rtol=1.5e-5, err_msg=f"{nf=}"
        )
