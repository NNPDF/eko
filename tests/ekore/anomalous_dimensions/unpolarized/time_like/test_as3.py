"""Testing values obtained from |MELA| functions."""

import numpy as np

import ekore.anomalous_dimensions.unpolarized.time_like.as3 as ad_as3
from eko.constants import CA, CF, zeta2, zeta3, zeta4, zeta5
from ekore.harmonics import cache as c

NF = 5


# Thanks Yuxun Guo (@yuxunguo) for providing results from 1905.01310
def n3(nf: int):
    z2 = zeta2
    z3 = zeta3
    z4 = zeta4
    z5 = zeta5
    cf = CF
    ca = CA

    gTqq2 = (
        (
            112 * z5
            + 48 * z2 * z3
            - 2083 / 3 * z4
            + 16153 / 18 * z3
            - 13105 / 72 * z2
            - 3049531 / 31104
        )
        * cf
        * ca**2
        + (
            -432 * z5
            - 208 * z2 * z3
            + 8252 / 3 * z4
            - 19424 / 9 * z3
            - 16709 / 27 * z2
            + 20329835 / 15552
        )
        * cf**2
        * ca
        + (
            416 * z5
            + 224 * z2 * z3
            - 6172 / 3 * z4
            + 10942 / 9 * z3
            + 11797 / 18 * z2
            - 17471825 / 15552
        )
        * cf**3
        + (146971 / 2700 * z2 - 5803 / 45 * z3 + 68 / 3 * z4 - 25234031 / 1944000)
        * ca
        * cf
        * nf
        + (-9767 / 225 * z2 + 8176 / 45 * z3 - 136 / 3 * z4 - 4100189 / 64800)
        * cf**2
        * nf
        - 105799 / 162000 * cf * nf**2
    )

    gTgq2 = (
        (-17093053 / 777600 - 50593 / 600 * z2 - 2791 / 90 * z3 + 196 / 3 * z4)
        * cf
        * ca**2
        + (63294389 / 388800 + 123773 / 900 * z2 - 3029 / 9 * z3 + 511 / 3 * z4)
        * cf**2
        * ca
        + (-647639 / 3888 + 3193 / 54 * z2 + 2533 / 9 * z3 - 308 * z4) * cf**3
        + (-73 / 27 * z2 + 182 / 9 * z3 + 246767 / 60750) * ca * cf * nf
        + (-419593 / 81000 + 4 / 9 * z2 - 28 / 9 * z3) * cf**2 * nf
    )

    gTqg2 = (
        (239959 / 13500 * z2 + 343 / 45 * z3 - 252 / 5 * z4 - 1795237 / 1944000)
        * ca**2
        * nf
        + (34127 / 1350 * z2 + 6208 / 75 * z3 - 42 / 5 * z4 - 3607891 / 38880)
        * ca
        * cf
        * nf
        + (-2042 / 225 * z2 - 26102 / 225 * z3 + 448 / 15 * z4 + 9397651 / 97200)
        * cf**2
        * nf
        + (-554 / 135 * z2 - 28 / 9 * z3 + 1215691 / 121500) * ca * nf**2
        + (2738 / 675 * z2 - 10657 / 4050) * cf * nf**2
        - 172 / 1125 * nf**3
    )

    gTgg2 = (
        (
            96 * z5
            + 64 * z2 * z3
            - 2566 / 15 * z4
            - 23702 / 225 * z3
            + 66358 / 1125 * z2
            - 5819653 / 486000
        )
        * ca**3
        + (-12230737 / 1944000 - 51269 / 540 * z2 + 239 / 9 * z3 + 104 * z4)
        * ca**2
        * nf
        + (-1700563 / 108000 - 16291 / 675 * z2 + 282 / 5 * z3) * ca * cf * nf
        + (219077 / 194400 + 2411 / 675 * z2 - 28 / 9 * z3) * cf**2 * nf
        + (-18269 / 10125 - 64 / 9 * z3 + 160 / 27 * z2) * ca * nf**2
        + (-2611 / 162000 - 196 / 135 * z2) * cf * nf**2
    )

    return np.array([[gTqq2, gTgq2 * 2 * nf], [gTqg2 / (2 * nf), gTgg2]])


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


def test_ps():
    Nlist = [2, complex(0, 1), complex(1, 1)]
    res = [
        -184.274140748,
        888.259743291 - 3850.903826041j,
        -521.507856510 + 1156.486663262j,
    ]
    for i in range(3):
        cache = c.reset()
        np.testing.assert_almost_equal(ad_as3.gamma_ps(Nlist[i], NF, cache), res[i])


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


def test_singlet_n3():
    cache = c.reset()
    # test against 1905.01310
    for nf in range(3, 6 + 1):
        np.testing.assert_allclose(
            ad_as3.gamma_singlet(3.0, nf, cache), n3(nf), rtol=1.5e-4
        )
