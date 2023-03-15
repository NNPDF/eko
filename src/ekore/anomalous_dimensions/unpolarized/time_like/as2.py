"""The unpolarized time-like NLO Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

# from eko.constants import zeta2
from numpy import power as npp

from eko import constants

from ....harmonics import w1, w2, w3
from ....harmonics.constants import zeta2, zeta3
from ....harmonics.polygamma import cern_polygamma as polygamma

# from ....harmonics import cache as c


@nb.njit(cache=True)
def gamma_nsp(N, nf, cache=None, is_singlet=None):
    r"""Compute the NLO non-singlet positive anomalous dimension.

    Implements Eqn. (B.7) from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    is_singlet : boolean
        True for singlet, False for non-singlet, None otherwise

    Returns
    -------
    gamma_nsp : complex
        NLO non-singlet positive anomalous dimension
        :math:`\gamma_{ns}^{(1)+}(N)`

    """
    s1 = w1.S1(N)  # c.get(c.S1, cache, N, is_singlet)
    s2 = w2.S2(N)  # c.get(c.S2, cache, N, is_singlet)
    s3 = w3.S3(N)  # c.get(c.S3, cache, N, is_singlet)
    sm1 = w1.Sm1(N, s1, is_singlet)  # to be removed
    sm2 = w2.Sm2(N, s2, is_singlet)  # c.get(c.Sm2, cache, N, is_singlet)
    sm3 = w3.Sm3(N, s3, is_singlet)  # c.get(c.Sm3, cache, N, is_singlet)
    sm21 = w3.Sm21(N, s1, sm1, is_singlet)  # c.get(c.Sm21, cache, N, is_singlet)

    if is_singlet == True:
        m1tpN = 1
    elif is_singlet == False:
        m1tpN = -1
    else:
        m1tpn = npp(-1, N)

    nsp1 = (
        constants.CF
        * constants.CF
        * (
            (
                8
                + 24 * N
                + 32 * npp(N, 2)
                - 11 * npp(N, 3)
                - 9 * npp(N, 4)
                - 9 * npp(N, 5)
                - 3 * npp(N, 6)
                + 32 * zeta2 * npp(N, 2)
                + 112 * zeta2 * npp(N, 3)
                + 176 * zeta2 * npp(N, 4)
                + 144 * zeta2 * npp(N, 5)
                + 48 * zeta2 * npp(N, 6)
                + 32 * npp(N, 3) * m1tpN
            )
            / (2 * npp(N, 3) * npp(1 + N, 3))
            - 16 * sm3
            + sm2 * ((16) / (N * (1 + N)) - 32 * s1)
            - (4 * (2 + 3 * N + 3 * npp(N, 2)) * s2) / (N * (1 + N))
            + s1
            * (
                16 * s2
                - (
                    8
                    * (
                        1
                        + 2 * N
                        + 4 * zeta2 * npp(N, 2)
                        + 8 * zeta2 * npp(N, 3)
                        + 4 * zeta2 * npp(N, 4)
                    )
                )
                / (npp(N, 2) * npp(1 + N, 2))
            )
            - 16 * s3
            + 32 * sm21
        )
    )
    nsp2 = (
        constants.CF
        * constants.CA
        * (
            (
                132
                - 208 * N
                - 851 * npp(N, 2)
                - 757 * npp(N, 3)
                - 153 * npp(N, 4)
                - 51 * npp(N, 5)
                - 144 * npp(N, 2) * m1tpN
            )
            / (18 * npp(N, 2) * npp(1 + N, 3))
            + 8 * sm3
            + s1 * (268 / 9)
            + sm2 * (16 * s1 - 8 / (N * (1 + N)))
            - s2 * (44 / 3)
            + 8 * s3
            - 16 * sm21
        )
    )
    nsp3 = (
        nf
        * constants.CF
        * (
            (-12 + 20 * N + 47 * npp(N, 2) + 6 * npp(N, 3) + 3 * npp(N, 4))
            / (9 * npp(N, 2) * npp(1 + N, 2))
            - s1 * (40 / 9)
            + s2 * (8 / 3)
        )
    )
    result = nsp1 + nsp2 + nsp3
    return result


@nb.njit(cache=True)
def gamma_nsm(N, nf):
    r"""Compute the NLO non-singlet negative anomalous dimension.

    Implements Eqn. (B.8) from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    is_singlet : boolean
        True for singlet, False for non-singlet, None otherwise

    Returns
    -------
    gamma_nsm : complex
        NLO non-singlet negative anomalous dimension
        :math:`\gamma_{ns}^{(1)-}(N)`

    """
    s1 = w1.S1(N)  # c.get(c.S1, cache, N, is_singlet)
    s2 = w2.S2(N)  # c.get(c.S2, cache, N, is_singlet)
    s3 = w3.S3(N)  # c.get(c.S3, cache, N, is_singlet)
    sm1 = w1.Sm1(N, s1, is_singlet=None)  # to be removed
    sm2 = w2.Sm2(N, s2, is_singlet=None)  # c.get(c.Sm2, cache, N, is_singlet)
    sm3 = w3.Sm3(N, s3, is_singlet=None)  # c.get(c.Sm3, cache, N, is_singlet)
    sm21 = w3.Sm21(N, s1, sm1, is_singlet=None)  # c.get(c.Sm21, cache, N, is_singlet)

    m1tpN = -1

    nsm1 = (
        constants.CF
        * constants.CF
        * (
            (
                40
                + 88 * N
                + 96 * npp(N, 2)
                + 53 * npp(N, 3)
                - 9 * npp(N, 4)
                - 9 * npp(N, 5)
                - 3 * npp(N, 6)
                + 32 * zeta2 * npp(N, 2)
                + 112 * zeta2 * npp(N, 3)
                + 176 * zeta2 * npp(N, 4)
                + 144 * zeta2 * npp(N, 5)
                + 48 * zeta2 * npp(N, 6)
                + 32 * npp(N, 3) * m1tpN
            )
            / (2 * npp(N, 3) * npp(1 + N, 3))
            - 16 * sm3
            + sm2 * (16 / (N * (1 + N)) - 32 * s1)
            - (4 * (2 + 3 * N + 3 * npp(N, 2)) * s2) / (N * (1 + N))
            + s1
            * (
                -(
                    (
                        8
                        * (
                            1
                            + 2 * N
                            + 4 * zeta2 * npp(N, 2)
                            + 8 * zeta2 * npp(N, 3)
                            + 4 * zeta2 * npp(N, 4)
                        )
                    )
                    / (npp(N, 2) * npp(1 + N, 2))
                )
                + 16 * s2
            )
            - 16 * s3
            + 32 * sm21
        )
    )
    nsm2 = (
        constants.CF
        * constants.CA
        * (
            (
                -144
                - 156 * N
                - 496 * npp(N, 2)
                - 1139 * npp(N, 3)
                - 757 * npp(N, 4)
                - 153 * npp(N, 5)
                - 51 * npp(N, 6)
                - 144 * npp(N, 3) * m1tpN
            )
            / (18 * npp(N, 3) * npp(1 + N, 3))
            + 8 * sm3
            + s1 * (268 / 9)
            + sm2 * (16 * s1 - 8 / (N * (1 + N)))
            - s2 * (44 / 3)
            + 8 * s3
            - 16 * sm21
        )
    )
    nsm3 = (
        nf
        * constants.CF
        * (
            (-12 + 20 * N + 47 * npp(N, 2) + 6 * npp(N, 3) + 3 * npp(N, 4))
            / (9 * npp(N, 2) * npp(1 + N, 2))
            - s1 * (40 / 9)
            + s2 * (8 / 3)
        )
    )
    result = nsm1 + nsm2 + nsm3
    return result


@nb.njit(cache=True)
def gamma_qqs(N, nf):
    r"""Compute the NLO quark-quark singlet anomalous dimension.

    Implements Eqn. (B.9) from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors

    Returns
    -------
    gamma_qqs : complex
        NLO quark-quark singlet anomalous dimension
        :math:`\gamma_{qq}^{(1)s}(N)`

    """
    qqs1 = (
        constants.CF
        * nf
        * (
            (
                4
                * (
                    8
                    + 44 * N
                    + 46 * npp(N, 2)
                    + 21 * npp(N, 3)
                    + 14 * npp(N, 4)
                    + 15 * npp(N, 5)
                    + 10 * npp(N, 6)
                    + 2 * npp(N, 7)
                )
            )
            / ((-1 + N) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 2))
        )
    )
    return qqs1


@nb.njit(cache=True)
def gamma_qg(N, nf):
    r"""Compute the NLO quark-gluon anomalous dimension.

    Implements Eqn. (B.10) from :cite:`Mitov:2006wy`
    and Eqn. (A1) from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    is_singlet : boolean
        True for singlet, False for non-singlet, None otherwise

    Returns
    -------
    gamma_qg : complex
        NLO quark-gluon anomalous dimension
        :math:`\gamma_{qg}^{(1)}(N)`

    """
    s1 = w1.S1(N)  # c.get(c.S1, cache, N, is_singlet)
    s2 = w2.S2(N)  # c.get(c.S2, cache, N, is_singlet)
    sm2 = w2.Sm2(N, s2, is_singlet=None)  # c.get(c.Sm2, cache, N, is_singlet)

    m1tpN = 1

    qg1 = (
        nf
        * constants.CF
        * (
            (
                2
                * (
                    8
                    + 12 * N
                    + 18 * npp(N, 2)
                    + 77 * npp(N, 3)
                    + 127 * npp(N, 4)
                    + 104 * npp(N, 5)
                    + 45 * npp(N, 6)
                    + 9 * npp(N, 7)
                )
            )
            / (npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 2))
            - (
                4
                * s1
                * (
                    -8
                    - 12 * N
                    + 22 * npp(N, 2)
                    + 25 * npp(N, 3)
                    + 10 * npp(N, 4)
                    + 3 * npp(N, 5)
                )
            )
            / (npp(N, 2) * npp(1 + N, 2) * npp(2 + N, 2))
            + (4 * npp(s1, 2) * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
            - (20 * s2 * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
        )
    )
    qg2 = (
        nf
        * constants.CA
        * (
            (
                4
                * (
                    144
                    + 600 * N
                    + 980 * npp(N, 2)
                    + 2366 * npp(N, 3)
                    + 2564 * npp(N, 4)
                    + 379 * npp(N, 5)
                    - 1177 * npp(N, 6)
                    - 1037 * npp(N, 7)
                    - 423 * npp(N, 8)
                    - 76 * npp(N, 9)
                    - 288 * zeta2 * npp(N, 2)
                    - 720 * zeta2 * npp(N, 3)
                    - 504 * zeta2 * npp(N, 4)
                    + 180 * zeta2 * npp(N, 5)
                    + 576 * zeta2 * npp(N, 6)
                    + 504 * zeta2 * npp(N, 7)
                    + 216 * zeta2 * npp(N, 8)
                    + 36 * zeta2 * npp(N, 9)
                    - 180 * m1tpN * npp(N, 3)
                    - 72 * m1tpN * npp(N, 4)
                    + 108 * m1tpN * npp(N, 5)
                    + 108 * m1tpN * npp(N, 6)
                    + 36 * m1tpN * npp(N, 7)
                )
            )
            / (9 * (-1 + N) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 3))
            + (8 * sm2 * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
            + (
                4
                * s1
                * (
                    -48
                    - 100 * N
                    + 40 * npp(N, 2)
                    + 77 * npp(N, 3)
                    + 32 * npp(N, 4)
                    + 11 * npp(N, 5)
                )
            )
            / (3 * npp(N, 2) * npp(1 + N, 2) * npp(2 + N, 2))
            - (4 * npp(s1, 2) * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
            + (12 * s2 * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
        )
    )
    qg3 = (
        nf
        * nf
        * (
            (
                8
                * (
                    -12
                    - 16 * N
                    + 37 * npp(N, 2)
                    + 41 * npp(N, 3)
                    + 17 * npp(N, 4)
                    + 5 * npp(N, 5)
                )
            )
            / (9 * npp(N, 2) * npp(1 + N, 2) * npp(2 + N, 2))
            - (8 * s1 * (2 + N + npp(N, 2))) / (3 * N * (1 + N) * (2 + N))
        )
    )
    result = (1 / (2 * nf)) * (qg1 + qg2 + qg3)
    return result


@nb.njit(cache=True)
def gamma_gq(N, nf):
    r"""Compute the NLO gluon-quark anomalous dimension.

    Implements Eqn. (B.11) from :cite:`Mitov:2006wy`
    and Eqn. (A1) from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    is_singlet : boolean
        True for singlet, False for non-singlet, None otherwise

    Returns
    -------
    gamma_gq : complex
        NLO gluon-quark anomalous dimension
        :math:`\gamma_{gq}^{(1)}(N)`

    """
    s1 = w1.S1(N)  # c.get(c.S1, cache, N, is_singlet)
    s2 = w2.S2(N)  # c.get(c.S2, cache, N, is_singlet)
    sm2 = w2.Sm2(N, s2, True)  # c.get(c.Sm2, cache, N, is_singlet)

    m1tpN = 1

    gq1 = (
        constants.CF
        * constants.CF
        * (
            (
                2
                * (
                    -4
                    - 4 * N
                    + 41 * npp(N, 2)
                    + 83 * npp(N, 3)
                    + 41 * npp(N, 4)
                    - 11 * npp(N, 5)
                    - 10 * npp(N, 6)
                    - 8 * npp(N, 7)
                    - 16 * zeta2 * npp(N, 2)
                    - 24 * zeta2 * npp(N, 3)
                    + 16 * zeta2 * npp(N, 5)
                    + 16 * zeta2 * npp(N, 6)
                    + 8 * zeta2 * npp(N, 7)
                )
            )
            / (npp(-1 + N, 2) * npp(N, 3) * npp(1 + N, 3))
            + (
                8
                * (4 - 2 * N - 16 * npp(N, 2) - npp(N, 3) - 2 * npp(N, 4) + npp(N, 5))
                * s1
            )
            / (npp(-1 + N, 2) * npp(N, 2) * npp(1 + N, 2))
            - (4 * (2 + N + npp(N, 2)) * npp(s1, 2)) / ((-1 + N) * N * (1 + N))
            + (12 * (2 + N + npp(N, 2)) * s2) / ((-1 + N) * N * (1 + N))
        )
    )
    gq2 = (
        constants.CF
        * constants.CA
        * (
            -(
                4
                * (
                    16
                    - 144 * npp(N, 2)
                    - 156 * npp(N, 3)
                    - 101 * npp(N, 4)
                    - 77 * npp(N, 5)
                    - 75 * npp(N, 6)
                    - 44 * npp(N, 7)
                    - npp(N, 8)
                    + 5 * npp(N, 9)
                    + npp(N, 10)
                    + 16 * N * m1tpN
                    + 32 * npp(N, 2) * m1tpN
                    - 20 * npp(N, 3) * m1tpN
                    - 44 * npp(N, 4) * m1tpN
                    - 26 * npp(N, 5) * m1tpN
                    + 14 * npp(N, 6) * m1tpN
                    + 22 * npp(N, 7) * m1tpN
                    + 6 * npp(N, 8) * m1tpN
                )
            )
            / (npp(-1 + N, 3) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 2))
            + (8 * (2 + N + npp(N, 2)) * sm2) / ((-1 + N) * N * (1 + N))
            - (8 * (2 - 2 * N - 9 * npp(N, 2) + npp(N, 3) - npp(N, 4) + npp(N, 5)) * s1)
            / (npp(-1 + N, 2) * npp(N, 2) * npp(1 + N, 2))
            + (4 * (2 + N + npp(N, 2)) * npp(s1, 2)) / ((-1 + N) * N * (1 + N))
            - (20 * (2 + N + npp(N, 2)) * s2) / ((-1 + N) * N * (1 + N))
        )
    )
    result = (2 * nf) * (gq1 + gq2)
    return result


@nb.njit(cache=True)
def gamma_gg(N, nf):
    r"""Compute the NLO gluon-gluon anomalous dimension.

    Implements Eqn. (B.12) from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    is_singlet : boolean
        True for singlet, False for non-singlet, None otherwise

    Returns
    -------
    gamma_gg : complex
        NLO gluon-gluon anomalous dimension
        :math:`\gamma_{gg}^{(1)}(N)`

    """
    s1 = w1.S1(N)  # c.get(c.S1, cache, N, is_singlet)
    s2 = w2.S2(N)  # c.get(c.S2, cache, N, is_singlet)
    s3 = w3.S3(N)  # c.get(c.S3, cache, N, is_singlet)
    sm1 = w1.Sm1(N, s1, is_singlet=None)  # to be removed
    sm2 = w2.Sm2(N, s2, is_singlet=None)  # c.get(c.Sm2, cache, N, is_singlet)
    sm3 = w3.Sm3(N, s3, is_singlet=None)  # c.get(c.Sm3, cache, N, is_singlet)
    sm21 = w3.Sm21(N, s1, sm1, is_singlet=None)  # c.get(c.Sm21, cache, N, is_singlet)

    m1tpN = 1

    gg1 = (
        nf
        * constants.CF
        * (
            (
                2
                * (
                    -16
                    + 8 * N
                    + 108 * npp(N, 2)
                    + 162 * npp(N, 3)
                    + 106 * npp(N, 4)
                    + 11 * npp(N, 5)
                    - 5 * npp(N, 6)
                    - 2 * npp(N, 7)
                    + 6 * npp(N, 8)
                    + 5 * npp(N, 9)
                    + npp(N, 10)
                )
            )
            / (npp(-1 + N, 2) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 2))
        )
    )
    gg2 = (
        constants.CA
        * constants.CA
        * (
            (
                2
                * (
                    -576
                    + 240 * N
                    + 3824 * npp(N, 2)
                    + 1240 * npp(N, 3)
                    + 1928 * npp(N, 4)
                    + 8303 * npp(N, 5)
                    + 10651 * npp(N, 6)
                    + 6614 * npp(N, 7)
                    + 1238 * npp(N, 8)
                    - 1133 * npp(N, 9)
                    - 889 * npp(N, 10)
                    - 288 * npp(N, 11)
                    - 48 * npp(N, 12)
                    + 1152 * zeta2 * npp(N, 2)
                    + 1248 * zeta2 * npp(N, 3)
                    - 1296 * zeta2 * npp(N, 4)
                    - 792 * zeta2 * npp(N, 5)
                    + 876 * zeta2 * npp(N, 6)
                    - 1368 * zeta2 * npp(N, 7)
                    - 2340 * zeta2 * npp(N, 8)
                    + 120 * zeta2 * npp(N, 9)
                    + 1476 * zeta2 * npp(N, 10)
                    + 792 * zeta2 * npp(N, 11)
                    + 132 * zeta2 * npp(N, 12)
                    - 576 * m1tpN * N
                    - 1440 * m1tpN * npp(N, 2)
                    + 216 * m1tpN * npp(N, 3)
                    + 1800 * m1tpN * npp(N, 4)
                    + 1800 * m1tpN * npp(N, 5)
                    - 72 * m1tpN * npp(N, 6)
                    - 1008 * m1tpN * npp(N, 7)
                    - 576 * m1tpN * npp(N, 8)
                    - 144 * m1tpN * npp(N, 9)
                )
            )
            / (9 * npp(-1 + N, 3) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 3))
            - (8 * sm3)
            + (sm2)
            * (
                (32 * (1 + N + npp(N, 2))) / ((-1 + N) * N * (1 + N) * (2 + N))
                - 16 * s1
            )
            - (8 * s2 * (12 - 10 * N + npp(N, 2) + 22 * npp(N, 3) + 11 * npp(N, 4)))
            / (3 * (-1 + N) * N * (1 + N) * (2 + N))
            + (s1)
            * (
                -(
                    4
                    * (
                        -144
                        - 144 * N
                        + 236 * npp(N, 2)
                        + 308 * npp(N, 3)
                        + 829 * npp(N, 4)
                        + 680 * npp(N, 5)
                        - 134 * npp(N, 6)
                        - 268 * npp(N, 7)
                        - 67 * npp(N, 8)
                        + 288 * zeta2 * npp(N, 2)
                        + 288 * zeta2 * npp(N, 3)
                        - 504 * zeta2 * npp(N, 4)
                        - 576 * zeta2 * npp(N, 5)
                        + 144 * zeta2 * npp(N, 6)
                        + 288 * zeta2 * npp(N, 7)
                        + 72 * zeta2 * npp(N, 8)
                    )
                )
                / (9 * npp(-1 + N, 2) * npp(N, 2) * npp(1 + N, 2) * npp(2 + N, 2))
                + 16 * s2
            )
            - (8 * s3)
            + (16 * sm21)
        )
    )
    gg3 = (
        nf
        * constants.CA
        * (
            -(
                8
                * (
                    -12
                    + 26 * N
                    + 132 * npp(N, 2)
                    + 85 * npp(N, 3)
                    + 34 * npp(N, 4)
                    - 9 * npp(N, 5)
                    - 25 * npp(N, 6)
                    - 12 * npp(N, 7)
                    - 3 * npp(N, 8)
                    + 24 * zeta2 * npp(N, 2)
                    + 24 * zeta2 * npp(N, 3)
                    - 42 * zeta2 * npp(N, 4)
                    - 48 * zeta2 * npp(N, 5)
                    + 12 * zeta2 * npp(N, 6)
                    + 24 * zeta2 * npp(N, 7)
                    + 6 * zeta2 * npp(N, 8)
                )
            )
            / (9 * npp(-1 + N, 2) * npp(N, 2) * npp(1 + N, 2) * npp(2 + N, 2))
            - (s1 * (40 / 9))
            + (s2 * (16 / 3))
        )
    )
    result = gg1 + gg2 + gg3
    return result


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the NLO singlet anomalous dimension matrix.

    Implements Eqn. (2.13) from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache : numpy.ndarray
        Harmonic sum cache
    is_singlet : boolean
        True for singlet, False for non-singlet, None otherwise

    Returns
    -------
    gamma_singlet : numpy.ndarray
        NLO singlet anomalous dimension matrix
        :math:`\gamma_{s}^{(1)}`

    """
    gamma_qq = gamma_nsp(N, nf, cache, True) + gamma_qqs(N, nf)

    result = np.array(
        [
            [gamma_qq, gamma_gq(N, nf)],
            [gamma_qg(N, nf), gamma_gg(N, nf)],
        ],
        np.complex_,
    )
    return result
