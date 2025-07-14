"""The |NNLO| polarized Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko.constants import zeta2

from ....harmonics import cache as c

# Non Singlet sector is swapped
from ...unpolarized.space_like.as3 import gamma_nsm as gamma_nsp
from ...unpolarized.space_like.as3 import gamma_nsp as gamma_nsm


@nb.njit(cache=True)
def gamma_gg(N, nf, cache):
    r"""Compute the parametrized |NNLO| gluon-gluon polarized anomalous
    dimension.

    Implement Eq. (4.12) of :cite:`Moch:2014sna`.

    Parameters
    ----------
    N : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NNLO| gluon-gluon anomalous dimension :math:`\\gamma_{gg}^{(2)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    E1 = S1 / N**2 + (-zeta2 + S2) / N
    gg_nf0 = (
        +4427.76
        - 12292 * E1
        + 12096 / N**5
        - 22665.0 / N**4
        + 21804 / N**3
        - 23091 / N**2
        + 33631.5 / N
        - 7002 / (1 + N) ** 4
        - 1726 / (1 + N) ** 3
        - 13247 / (1 + N) ** 2
        - 39925 / (1 + N)
        + 13447 / (2 + N)
        - 4576 / (3 + N)
        - 2643.52 * S1
        + (9446 * S1) / N
        - (13247 * S1) / (1 + N)
    )
    gg_nf1 = (
        -528.536
        - 7932 * E1
        - 6128 / (9 * N**5)
        + 2146.79 / N**4
        - 3754.4 / N**3
        + 3524 / N**2
        - 1585.67 / N
        - 786.0 / (1 + N) ** 4
        + 1226.2 / (1 + N) ** 3
        - 6746 / (1 + N) ** 2
        + 2648.6 / (1 + N)
        - 2160.8 / (2 + N)
        + 1251.7 / (3 + N)
        + 412.172 * S1
        + (7041.7 * S1) / N
        - (6746 * S1) / (1 + N)
    )
    gg_nf2 = (
        6.4607
        - 16.944 * E1
        + 7.0854 / N**4
        - 13.358 / N**3
        + 13.29 / N**2
        - 18.3838 / N
        + 31.528 / (1 + N) ** 3
        + 32.905 / (1 + N)
        - 18.3 / (2 + N)
        + 2.637 / (3 + N)
        + (16 * S1) / 9
        + (0.21 * S1) / N
    )
    return -(gg_nf0 + gg_nf1 * nf + gg_nf2 * nf**2)


@nb.njit(cache=True)
def gamma_qg(N, nf, cache):
    r"""Compute the parametrized |NNLO| quark-gluon polarized anomalous
    dimension.

    Implement Eq. (4.10) of :cite:`Moch:2014sna`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NNLO| quark-gluon anomalous dimension :math:`\\gamma_{qg}^{(2)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S4 = c.get(c.S4, cache, N)
    B3 = (-(S1**3) - 3 * S1 * S2 - 2 * S3) / N
    B4 = (S1**4 + 6 * S1**2 * S2 + 3 * S2**2 + 8 * S1 * S3 + 6 * S4) / N
    E1 = S1 / N**2 + (-zeta2 + S2) / N
    qg_nf1 = (
        -5.3 * B3
        + 3.784 * B4
        + 825.4 * E1
        - 1208 / N**5
        + 2313.84 / N**4
        - 1789.6 / N**3
        + 1461.2 / N**2
        - 2972.4 / N
        + 439.8 / (1 + N) ** 4
        + 2290.6 / (1 + N) ** 3
        + 4672 / (1 + N)
        - 1221.6 / (2 + N)
        - 18 / (3 + N)
        - (278.32 * S1) / N
        - (90.26 * S1**2) / N
        - (90.26 * S2) / N
    )
    qg_nf2 = (
        0.7374 * B3
        - 47.3 * E1
        + 128 / (3 * N**5)
        - 184.434 / N**4
        + 393.92 / N**3
        - 526.3 / N**2
        + 499.65 / N
        - 61.116 / (1 + N) ** 4
        + 358.2 / (1 + N) ** 3
        - 432.18 / (1 + N)
        - 141.63 / (2 + N)
        - 11.34 / (3 + N)
        + (6.256 * S1) / N
        + (7.32 * S1**2) / N
        + (7.32 * S2) / N
    )
    return -(qg_nf1 * nf + qg_nf2 * nf**2)


@nb.njit(cache=True)
def gamma_gq(N, nf, cache):
    r"""Compute the parametrized |NNLO| gluon-quark polarized anomalous
    dimension.

    Implement Eq. (4.11) of :cite:`Moch:2014sna`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NNLO| gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(2)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S4 = c.get(c.S4, cache, N)
    B3 = (-(S1**3) - 3 * S1 * S2 - 2 * S3) / N
    B4 = (S1**4 + 6 * S1**2 * S2 + 3 * S2**2 + 8 * S1 * S3 + 6 * S4) / N
    E1 = S1 / N**2 + (-zeta2 + S2) / N
    gq_nf0 = (
        59.3 * B3
        + 5.143 * B4
        - 1424.8 * E1
        + 92096 / (27 * N**5)
        - 5328.02 / N**4
        + 4280 / N**3
        - 4046.6 / N**2
        + 6159 / N
        - 1050.6 / (1 + N) ** 4
        - 1701.4 / (1 + N) ** 3
        - 3825.9 / (1 + N)
        + 1942.0 / (2 + N)
        - 742.1 / (3 + N)
        - (1843.7 * S1) / N
        + (451.55 * S1**2) / N
        + (451.55 * S2) / N
    )
    gq_nf1 = (
        -4.963 * B3
        - 16.18 * E1
        - 1024 / (9 * N**5)
        + 236.323 / N**4
        - 404.92 / N**3
        + 308.98 / N**2
        - 301.07 / N
        + 180.138 / (1 + N) ** 4
        - 253.06 / (1 + N) ** 3
        - 296 / (1 + N)
        + 406.13 / (2 + N)
        - 101.62 / (3 + N)
        + (171.78 * S1) / N
        - (47.86 * S1**2) / N
        - (47.86 * S2) / N
    )
    gq_nf2 = (
        -(64 / (9 * N))
        - 32 / (9 * (1 + N) ** 3)
        - 32 / (27 * (1 + N) ** 2)
        + 160 / (27 * (1 + N))
        - (128 * S1) / (27 * N)
        - (32 * S1) / (9 * (1 + N) ** 2)
        - (32 * S1) / (27 * (1 + N))
        + (32 * S1**2) / (9 * N)
        - (16 * S1**2) / (9 * (1 + N))
        + (32 * S2) / (9 * N)
        - (16 * S2) / (9 * (1 + N))
    )
    return -(gq_nf0 + gq_nf1 * nf + gq_nf2 * nf**2)


@nb.njit(cache=True)
def gamma_ps(N, nf, cache):
    r"""Compute the parametrized |NNLO| pure-singlet quark-quark polarized
    anomalous dimension.

    Implement Eq. (4.9) of :cite:`Moch:2014sna`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NNLO| pure-singlet quark-quark anomalous dimension :math:`\\gamma_{ps}^{(2)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    B3 = (-(S1**3) - 3 * S1 * S2 - 2 * S3) / N
    B31 = (
        -((1 / (1 + N) + S1) ** 3)
        - 3 * (1 / (1 + N) + S1) * (1 / (1 + N) ** 2 + S2)
        - 2 * (1 / (1 + N) ** 3 + S3)
    ) / (1 + N)
    E1 = S1 / N**2 + (-zeta2 + S2) / N
    E11 = (1 / (1 + N) + S1) / (1 + N) ** 2 + (1 / (1 + N) ** 2 - zeta2 + S2) / (1 + N)
    ps_nf1 = (
        1
        / 9
        * (
            -58.869 * B3
            + 58.869 * B31
            + 2093.13 * E1
            - 2093.13 * E11
            - 2752 / N**5
            + 4909.67 / N**4
            - 6634.8 / N**3
            + 6651.0 / N**2
            - 12263.4 / N
            + 2752 / (1 + N) ** 5
            - 508.669 / (1 + N) ** 4
            + 13160.0 / (1 + N) ** 3
            - 8493.84 / (1 + N) ** 2
            + 26820.0 / (1 + N)
            - 4401.0 / (2 + N) ** 4
            - 6298.2 / (2 + N) ** 3
            - 20629.8 / (2 + N)
            + 7579.89 / (3 + N)
            - 1506.69 / (4 + N)
            + (1842.84 * S1) / N
            + (226.98 * S1) / (1 + N) ** 2
            - (1842.84 * S1) / (1 + N)
            - (113.49 * S1**2) / N
            + (113.49 * S1**2) / (1 + N)
            - (113.49 * S2) / N
            + (113.49 * S2) / (1 + N)
        )
    )
    ps_nf2 = (
        1
        / 9
        * (
            -(63.4014 / N**4)
            + 239.166 / N**3
            - 409.338 / N**2
            + 442.17 / N
            + 107.968 / (1 + N) ** 4
            - 79.389 / (1 + N) ** 3
            + 494.991 / (1 + N) ** 2
            - 719.1 / (1 + N)
            - 44.5662 / (2 + N) ** 4
            - 191.826 / (2 + N) ** 3
            + 238.167 / (2 + N)
            + 34.1784 / (3 + N)
            + 4.5846 / (4 + N)
            - (85.653 * S1) / N
            - (32.049 * S1) / (1 + N) ** 2
            + (85.653 * S1) / (1 + N)
            + (16.0245 * S1**2) / N
            - (16.0245 * S1**2) / (1 + N)
            + (16.0245 * S2) / N
            - (16.0245 * S2) / (1 + N)
        )
    )
    return -(ps_nf1 * nf + ps_nf2 * nf**2)


@nb.njit(cache=True)
def gamma_nss(N, nf, cache):
    r"""Compute the |NNLO| sea-like polarized non-singlet anomalous dimension.

    Implement Eq. (24) of :cite:`Moch:2015usa`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NNLO| valence non-singlet anomalous dimension
        :math:`\\gamma_{ns,s}^{(2)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S3 = c.get(c.S3, cache, N)
    Sm2 = c.get(c.Sm2, cache, N, is_singlet=True)
    Sm3 = c.get(c.Sm3, cache, N, is_singlet=True)
    Sm21 = c.get(c.Sm21, cache, N, is_singlet=True)
    nss_nf1 = (
        40
        / 9
        * (
            1 / N**2
            + (
                -(2 / N**2)
                - 12 / (N**4 * (1 + N) ** 4)
                - 42 / (N**3 * (1 + N) ** 3)
                - 14 / (N**2 * (1 + N) ** 2)
                + 8 / (N * (1 + N))
            )
            * S1
            + (4 / (N**2 * (1 + N) ** 2) + 6 / (N * (1 + N))) * S3
            + (
                -(4 / N**2)
                + 8 / (N**3 * (1 + N) ** 3)
                + 20 / (N**2 * (1 + N) ** 2)
                + 8 / (N * (1 + N))
            )
            * Sm2
            + (32 * Sm21) / (N * (1 + N))
            + (8 / (N**2 * (1 + N) ** 2) - 20 / (N * (1 + N))) * Sm3
            + (-(16 / (N**2 * (1 + N) ** 2)) + 8 / (N * (1 + N)))
            * (S1 * Sm2 - Sm21 + Sm3)
        )
    )
    return -(nss_nf1 * nf)


@nb.njit(cache=True)
def gamma_nsv(N, nf, cache):
    r"""Compute the |NNLO| valence polarized non-singlet anomalous dimension.

    Implement Eq. (23) of :cite:`Moch:2015usa`.

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        |NNLO| valence non-singlet anomalous dimension
        :math:`\\gamma_{ns,v}^{(2)}(N)`
    """
    return gamma_nsm(N, nf, cache) + gamma_nss(N, nf, cache)


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the |NNLO| polarized singlet anomalous dimension matrix.

        .. math::
            \gamma_S^{(1)} = \left(\begin{array}{cc}
            \gamma_{qq}^{(2)} & \gamma_{qg}^{(2)}\\
            \gamma_{gq}^{(2)} & \gamma_{gg}^{(2)}
            \end{array}\right)

    Parameters
    ----------
    n : complex
        Mellin moment
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        |NNLO| singlet anomalous dimension matrix :math:`\gamma_{S}^{(2)}(N)`
    """
    gamma_qq = gamma_nsp(N, nf, cache) + gamma_ps(N, nf, cache)
    gamma_S_0 = np.array(
        [
            [gamma_qq, gamma_qg(N, nf, cache)],
            [gamma_gq(N, nf, cache), gamma_gg(N, nf, cache)],
        ],
        np.complex128,
    )
    return gamma_S_0
