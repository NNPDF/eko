"""The unpolarized, time-like |NLO| Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np
from numpy import power as npp

from eko import constants
from eko.constants import zeta2, zeta3

from ....harmonics import cache as c


@nb.njit(cache=True)
def gamma_nsp(N, nf, cache):
    r"""Compute the |NLO| non-singlet positive anomalous dimension.

    Implements :eqref:`A6` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NLO non-singlet positive anomalous dimension
        :math:`\gamma_{ns}^{(1)+}(N)`
    """
    NS = N * N
    NT = NS * N
    NFO = NT * N

    N1 = N + 1
    N1S = N1 * N1
    N1T = N1S * N1

    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S1h = c.get(c.S1h, cache, N)
    S2h = c.get(c.S2h, cache, N)
    S3h = c.get(c.S3h, cache, N)
    S1ph = c.get(c.S1ph, cache, N)
    g3 = c.get(c.g3, cache, N)

    SLC = -5 / 8 * zeta3
    SLV = g3 + S1 / N**2 - (zeta2 / 2) * (-2 / (1 + N) + 2 / N + S1ph - S1h)

    SSCHLP = SLC + SLV

    PNPA = (
        16 * S1 * (2 * N + 1) / (NS * N1S)
        + 16 * (2 * S1 - 1 / (N * N1)) * (S2 - S2h)
        + 64 * SSCHLP
        + 24 * S2
        - 3
        - 8 * S3h
        - 8 * (3 * NT + NS - 1) / (NT * N1T)
        - 16 * (2 * NS + 2 * N + 1) / (NT * N1T)
    ) * (-0.5)
    PNSB = (
        S1 * (536 / 9 + 8 * (2 * N + 1) / (NS * N1S))
        - (16 * S1 + 52 / 3 - 8 / (N * N1)) * S2
        - 43 / 6
        - (151 * NFO + 263 * NT + 97 * NS + 3 * N + 9) * 4 / (9 * NT * N1T)
    ) * (-0.5)
    PNSC = (
        -160 / 9 * S1
        + 32 / 3.0 * S2
        + 4 / 3
        + 16 * (11 * NS + 5 * N - 3) / (9 * NS * N1S)
    ) * (-0.5)
    PNSTL = (-4 * S1 + 3 + 2 / (N * N1)) * (
        2 * S2 - 2 * zeta2 - (2 * N + 1) / (NS * N1S)
    )

    result = (
        constants.CF
        * (
            (constants.CF - constants.CA / 2) * PNPA
            + constants.CA * PNSB
            + (1 / 2) * nf * PNSC
        )
        + constants.CF**2 * PNSTL * 4
    )
    return -result


@nb.njit(cache=True)
def gamma_nsm(N, nf, cache):
    r"""Compute the |NLO| non-singlet negative anomalous dimension.

    Based on :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NLO non-singlet negative anomalous dimension
        :math:`\gamma_{ns}^{(1)-}(N)`
    """
    NS = N * N
    NT = NS * N
    NFO = NT * N

    N1 = N + 1
    N1S = N1 * N1
    N1T = N1S * N1

    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S1h = c.get(c.S1h, cache, N)
    S2mh = c.get(c.S2mh, cache, N)
    S3mh = c.get(c.S3mh, cache, N)
    S1ph = c.get(c.S1ph, cache, N)
    g3 = c.get(c.g3, cache, N)

    SLC = -5 / 8 * zeta3
    SLV = g3 + S1 / N**2 - (zeta2 / 2) * (-2 / (1 + N) + 2 / N + S1ph - S1h)
    SSCHLM = SLC - SLV

    PNMA = (
        16 * S1 * (2 * N + 1) / (NS * N1S)
        + 16 * (2 * S1 - 1 / (N * N1)) * (S2 - S2mh)
        + 64 * SSCHLM
        + 24 * S2
        - 3
        - 8 * S3mh
        - 8 * (3 * NT + NS - 1) / (NT * N1T)
        + 16 * (2 * NS + 2 * N + 1) / (NT * N1T)
    ) * (-0.5)
    PNSB = (
        S1 * (536 / 9 + 8 * (2 * N + 1) / (NS * N1S))
        - (16 * S1 + 52 / 3 - 8 / (N * N1)) * S2
        - 43 / 6
        - (151 * NFO + 263 * NT + 97 * NS + 3 * N + 9) * 4 / (9 * NT * N1T)
    ) * (-0.5)
    PNSC = (
        -160 / 9 * S1
        + 32 / 3.0 * S2
        + 4 / 3
        + 16 * (11 * NS + 5 * N - 3) / (9 * NS * N1S)
    ) * (-0.5)
    PNSTL = (-4 * S1 + 3 + 2 / (N * N1)) * (
        2 * S2 - 2 * zeta2 - (2 * N + 1) / (NS * N1S)
    )

    result = (
        constants.CF
        * (
            (constants.CF - constants.CA / 2) * PNMA
            + constants.CA * PNSB
            + (1 / 2) * nf * PNSC
        )
        + constants.CF**2 * PNSTL * 4
    )
    return -result


@nb.njit(cache=True)
def gamma_qqs(N, nf):
    r"""Compute the |NLO| quark-quark singlet anomalous dimension.

    Implements :eqref:`B.9` from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors

    Returns
    -------
    complex
        NLO quark-quark singlet anomalous dimension
        :math:`\gamma_{qq}^{(1)s}(N)`
    """
    qqS1 = (
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
    return qqS1


@nb.njit(cache=True)
def gamma_qg(N, nf, cache):
    r"""Compute the |NLO| quark-gluon anomalous dimension.

    Implements :eqref:`B.10` from :cite:`Mitov:2006wy`
    and :eqref:`A1` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NLO quark-gluon anomalous dimension
        :math:`\gamma_{qg}^{(1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    Sm2 = c.get(c.Sm2, cache, N, is_singlet=True)

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
                * S1
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
            + (4 * npp(S1, 2) * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
            - (20 * S2 * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
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
                    - 180 * npp(N, 3)
                    - 72 * npp(N, 4)
                    + 108 * npp(N, 5)
                    + 108 * npp(N, 6)
                    + 36 * npp(N, 7)
                )
            )
            / (9 * (-1 + N) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 3))
            + (8 * Sm2 * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
            + (
                4
                * S1
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
            - (4 * npp(S1, 2) * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
            + (12 * S2 * (2 + N + npp(N, 2))) / (N * (1 + N) * (2 + N))
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
            - (8 * S1 * (2 + N + npp(N, 2))) / (3 * N * (1 + N) * (2 + N))
        )
    )
    result = (1 / (2 * nf)) * (qg1 + qg2 + qg3)
    return result


@nb.njit(cache=True)
def gamma_gq(N, nf, cache):
    r"""Compute the |NLO| gluon-quark anomalous dimension.

    Implements :eqref:`B.11` from :cite:`Mitov:2006wy`
    and :eqref:`A1` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NLO gluon-quark anomalous dimension
        :math:`\gamma_{gq}^{(1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    Sm2 = c.get(c.Sm2, cache, N, is_singlet=True)
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
                * S1
            )
            / (npp(-1 + N, 2) * npp(N, 2) * npp(1 + N, 2))
            - (4 * (2 + N + npp(N, 2)) * npp(S1, 2)) / ((-1 + N) * N * (1 + N))
            + (12 * (2 + N + npp(N, 2)) * S2) / ((-1 + N) * N * (1 + N))
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
                    + 16 * N
                    + 32 * npp(N, 2)
                    - 20 * npp(N, 3)
                    - 44 * npp(N, 4)
                    - 26 * npp(N, 5)
                    + 14 * npp(N, 6)
                    + 22 * npp(N, 7)
                    + 6 * npp(N, 8)
                )
            )
            / (npp(-1 + N, 3) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 2))
            + (8 * (2 + N + npp(N, 2)) * Sm2) / ((-1 + N) * N * (1 + N))
            - (8 * (2 - 2 * N - 9 * npp(N, 2) + npp(N, 3) - npp(N, 4) + npp(N, 5)) * S1)
            / (npp(-1 + N, 2) * npp(N, 2) * npp(1 + N, 2))
            + (4 * (2 + N + npp(N, 2)) * npp(S1, 2)) / ((-1 + N) * N * (1 + N))
            - (20 * (2 + N + npp(N, 2)) * S2) / ((-1 + N) * N * (1 + N))
        )
    )
    result = (2 * nf) * (gq1 + gq2)
    return result


@nb.njit(cache=True)
def gamma_gg(N, nf, cache):
    r"""Compute the |NLO| gluon-gluon anomalous dimension.

    Implements :eqref:`B.12` from :cite:`Mitov:2006wy`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        NLO gluon-gluon anomalous dimension
        :math:`\gamma_{gg}^{(1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    Sm2 = c.get(c.Sm2, cache, N, is_singlet=True)
    Sm3 = c.get(c.Sm3, cache, N, is_singlet=True)
    Sm21 = c.get(c.Sm21, cache, N, is_singlet=True)

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
                    - 576 * N
                    - 1440 * npp(N, 2)
                    + 216 * npp(N, 3)
                    + 1800 * npp(N, 4)
                    + 1800 * npp(N, 5)
                    - 72 * npp(N, 6)
                    - 1008 * npp(N, 7)
                    - 576 * npp(N, 8)
                    - 144 * npp(N, 9)
                )
            )
            / (9 * npp(-1 + N, 3) * npp(N, 3) * npp(1 + N, 3) * npp(2 + N, 3))
            - (8 * Sm3)
            + (Sm2)
            * (
                (32 * (1 + N + npp(N, 2))) / ((-1 + N) * N * (1 + N) * (2 + N))
                - 16 * S1
            )
            - (8 * S2 * (12 - 10 * N + npp(N, 2) + 22 * npp(N, 3) + 11 * npp(N, 4)))
            / (3 * (-1 + N) * N * (1 + N) * (2 + N))
            + (S1)
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
                + 16 * S2
            )
            - (8 * S3)
            + (16 * Sm21)
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
            - (S1 * (40 / 9))
            + (S2 * (16 / 3))
        )
    )
    result = gg1 + gg2 + gg3
    return result


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the |NLO| singlet anomalous dimension matrix.

    Implements :eqref:`2.13` from :cite:`Gluck:1992zx`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        No. of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        NLO singlet anomalous dimension matrix
        :math:`\gamma_{s}^{(1)}`
    """
    gamma_qq = gamma_nsp(N, nf, cache) + gamma_qqs(N, nf)

    result = np.array(
        [
            [gamma_qq, gamma_gq(N, nf, cache)],
            [gamma_qg(N, nf, cache), gamma_gg(N, nf, cache)],
        ],
        np.complex128,
    )
    return result
