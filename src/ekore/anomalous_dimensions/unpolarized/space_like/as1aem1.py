"""The :math:`O(a_s^1a_{em}^1)` Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants
from eko.constants import zeta2, zeta3

from ....harmonics import cache as c


@nb.njit(cache=True)
def gamma_phq(N, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` photon-quark anomalous dimension.

    Implements Eq. (36) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache


    Returns
    -------
    complex
        :math:`O(a_s^1a_{em}^1)` photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(1,1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    tmp_const = (
        2.0
        * (
            -4.0
            - 12.0 * N
            - N**2
            + 28.0 * N**3
            + 43.0 * N**4
            + 30.0 * N**5
            + 12.0 * N**6
        )
        / ((-1.0 + N) * N**3 * (1.0 + N) ** 3)
    )
    tmp_S1 = (
        -4.0
        * (10.0 + 27.0 * N + 25.0 * N**2 + 13.0 * N**3 + 5.0 * N**4)
        / ((-1.0 + N) * N * (1.0 + N) ** 3)
    )
    tmp_S12 = 4.0 * (2.0 + N + N**2) / ((-1.0 + N) * N * (1.0 + N))
    tmp_S2 = 4.0 * (2.0 + N + N**2) / ((-1.0 + N) * N * (1.0 + N))

    return constants.CF * (tmp_const + tmp_S1 * S1 + tmp_S12 * S1**2 + tmp_S2 * S2)


@nb.njit(cache=True)
def gamma_qph(N, nf, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` quark-photon anomalous dimension.

    Implements Eq. (26) of :cite:`deFlorian:2015ujt`.

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
        :math:`O(a_s^1a_{em}^1)` quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(1,1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    tmp_const = (
        -2.0
        * (
            4.0
            + 8.0 * N
            + 25.0 * N**2
            + 51.0 * N**3
            + 36.0 * N**4
            + 15.0 * N**5
            + 5.0 * N**6
        )
        / (N**3 * (1.0 + N) ** 3 * (2.0 + N))
    )
    tmp_S1 = 8.0 / N**2
    tmp_S12 = -4.0 * (2.0 + N + N**2) / (N * (1.0 + N) * (2.0 + N))
    tmp_S2 = 4.0 * (2.0 + N + N**2) / (N * (1.0 + N) * (2.0 + N))
    return (
        2.0
        * nf
        * constants.CA
        * constants.CF
        * (tmp_const + tmp_S1 * S1 + tmp_S12 * S1**2 + tmp_S2 * S2)
    )


@nb.njit(cache=True)
def gamma_gph(N):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` gluon-photon anomalous dimension.

    Implements Eq. (27) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    complex
        :math:`O(a_s^1a_{em}^1)` gluon-photon anomalous dimension :math:`\\gamma_{g \\gamma}^{(1,1)}(N)`
    """
    return (
        constants.CF
        * constants.CA
        * (
            8.0
            * (
                -4.0
                + N * (-4.0 + N * (-5.0 + N * (-10.0 + N + 2.0 * N**2 * (2.0 + N))))
            )
        )
        / (N**3 * (1.0 + N) ** 3 * (-2.0 + N + N**2))
    )


@nb.njit(cache=True)
def gamma_phg(N):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` photon-gluon anomalous dimension.

    Implements Eq. (30) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    complex
        :math:`O(a_s^1a_{em}^1)` photon-gluon anomalous dimension :math:`\\gamma_{\\gamma g}^{(1,1)}(N)`
    """
    return constants.TR / constants.CF / constants.CA * constants.NC * gamma_gph(N)


@nb.njit(cache=True)
def gamma_qg(N, nf, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` quark-gluon singlet anomalous
    dimension.

    Implements Eq. (29) of :cite:`deFlorian:2015ujt`.

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
        :math:`O(a_s^1a_{em}^1)` quark-gluon singlet anomalous dimension
        :math:`\\gamma_{qg}^{(1,1)}(N)`
    """
    return (
        constants.TR
        / constants.CF
        / constants.CA
        * constants.NC
        * gamma_qph(N, nf, cache)
    )


@nb.njit(cache=True)
def gamma_gq(N, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` gluon-quark singlet anomalous
    dimension.

    Implements Eq. (35) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        :math:`O(a_s^1a_{em}^1)` gluon-quark singlet anomalous dimension
        :math:`\\gamma_{gq}^{(1,1)}(N)`
    """
    return gamma_phq(N, cache)


@nb.njit(cache=True)
def gamma_phph(nf):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` photon-photon singlet anomalous
    dimension.

    Implements Eq. (28) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    nf : int
        Number of active flavors

    Returns
    -------
    gamma_gg : complex
        :math:`O(a_s^1a_{em}^1)` photon-photon singlet anomalous dimension
        :math:`\\gamma_{\\gamma \\gamma}^{(1,1)}(N)`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return 4.0 * constants.CF * constants.CA * (nu * constants.eu2 + nd * constants.ed2)


@nb.njit(cache=True)
def gamma_gg():
    r"""Compute the :math:`O(a_s^1a_{em}^1)` gluon-gluon singlet anomalous
    dimension.

    Implements Eq. (31) of :cite:`deFlorian:2015ujt`.

    Returns
    -------
    complex
        :math:`O(a_s^1a_{em}^1)` gluon-gluon singlet anomalous dimension
        :math:`\\gamma_{gg}^{(1,1)}(N)`
    """
    return 4.0 * constants.TR * constants.NC


@nb.njit(cache=True)
def gamma_nsp(N, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` singlet-like non-singlet anomalous
    dimension.

    Implements sum of Eqs. (33-34) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    gamma_nsp : complex
        :math:`O(a_s^1a_{em}^1)` singlet-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,+}^{(1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S1h = c.get(c.S1h, cache, N)
    S2h = c.get(c.S2h, cache, N)
    S3h = c.get(c.S3h, cache, N)
    S1p1h = c.get(c.S1ph, cache, N)
    S2p1h = c.get(c.S2ph, cache, N)
    S3p1h = c.get(c.S3ph, cache, N)
    g3N = c.get(c.g3, cache, N)
    g3Np2 = c.get(c.g3p2, cache, N)
    result = (
        +32.0 * zeta2 * S1h
        - 32.0 * zeta2 * S1p1h
        + 8.0 / (N + N**2) * S2h
        - 4.0 * S3h
        + (24.0 + 16.0 / (N + N**2)) * S2
        - 32.0 * S3
        - 8.0 / (N + N**2) * S2p1h
        + S1
        * (
            +16.0 * (3.0 / N**2 - 3.0 / (1.0 + N) ** 2 + 2.0 * zeta2)
            - 16.0 * S2h
            - 32.0 * S2
            + 16.0 * S2p1h
        )
        + (
            -8.0
            + N
            * (
                -32.0
                + N
                * (
                    -8.0
                    - 3.0 * N * (3.0 + N) * (3.0 + N**2)
                    - 48.0 * (1.0 + N) ** 2 * zeta2
                )
            )
        )
        / (N**3 * (1.0 + N) ** 3)
        + 32.0 * (g3N + g3Np2)
        + 4.0 * S3p1h
        - 16.0 * zeta3
    )
    return constants.CF * result


@nb.njit(cache=True)
def gamma_nsm(N, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` valence-like non-singlet anomalous
    dimension.

    Implements difference between Eqs. (33-34) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        :math:`O(a_s^1a_{em}^1)` singlet-like non-singlet anomalous dimension
        :math:`\\gamma_{ns,-}^{(1,1)}(N)`
    """
    S1 = c.get(c.S1, cache, N)
    S2 = c.get(c.S2, cache, N)
    S3 = c.get(c.S3, cache, N)
    S1h = c.get(c.S1h, cache, N)
    S2h = c.get(c.S2h, cache, N)
    S3h = c.get(c.S3h, cache, N)
    S1p1h = c.get(c.S1ph, cache, N)
    S2p1h = c.get(c.S2ph, cache, N)
    S3p1h = c.get(c.S3ph, cache, N)
    g3N = c.get(c.g3, cache, N)
    g3Np2 = c.get(c.g3p2, cache, N)
    result = (
        -32.0 * zeta2 * S1h
        - 8.0 / (N + N**2) * S2h
        + (24.0 + 16.0 / (N + N**2)) * S2
        + 8.0 / (N + N**2) * S2p1h
        + S1
        * (
            16.0 * (-1.0 / N**2 + 1.0 / (1.0 + N) ** 2 + 2.0 * zeta2)
            + 16.0 * S2h
            - 32.0 * S2
            - 16.0 * S2p1h
        )
        + (
            72.0
            + N
            * (
                96.0
                - 3.0 * N * (8.0 + 3.0 * N * (3.0 + N) * (3.0 + N**2))
                + 48.0 * N * (1.0 + N) ** 2 * zeta2
            )
        )
        / (3.0 * N**3 * (1.0 + N) ** 3)
        - 32.0 * (g3N + g3Np2)
        + 32.0 * zeta2 * S1p1h
        + 4.0 * S3h
        - 32.0 * S3
        - 4.0 * S3p1h
        - 16.0 * zeta3
    )
    return constants.CF * result


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` singlet sector.

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
    numpy.ndarray
        :math:`O(a_s^1a_{em}^1)` singlet anomalous dimension :math:`\\gamma_{S}^{(1,1)}(N,nf,cache)`
    """
    e2avg, vue2m, vde2m, e2delta = constants.charge_combinations(nf)
    e2_tot = nf * e2avg
    gamma_g_q = gamma_gq(N, cache)
    gamma_ph_q = gamma_phq(N, cache)
    gamma_q_g = gamma_qg(N, nf, cache)
    gamma_q_ph = gamma_qph(N, nf, cache)
    gamma_ns_p = gamma_nsp(N, cache)
    gamma_S_11 = np.array(
        [
            [
                e2_tot * gamma_gg(),
                e2_tot * gamma_gph(N),
                e2avg * gamma_g_q,
                vue2m * gamma_g_q,
            ],
            [
                e2_tot * gamma_phg(N),
                gamma_phph(nf),
                e2avg * gamma_ph_q,
                vue2m * gamma_ph_q,
            ],
            [
                e2avg * gamma_q_g,
                e2avg * gamma_q_ph,
                e2avg * gamma_ns_p,
                vue2m * gamma_ns_p,
            ],
            [
                vde2m * gamma_q_g,
                vde2m * gamma_q_ph,
                vde2m * gamma_ns_p,
                e2delta * gamma_ns_p,
            ],
        ],
        np.complex128,
    )
    return gamma_S_11


@nb.njit(cache=True)
def gamma_valence(N, nf, cache):
    r"""Compute the :math:`O(a_s^1a_{em}^1)` valence sector.

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
    numpy.ndarray
        :math:`O(a_s^1a_{em}^1)` valence anomalous dimension :math:`\\gamma_{V}^{(1,1)}(N,nf,cache)`
    """
    e2avg, vue2m, vde2m, e2delta = constants.charge_combinations(nf)
    gamma_V_11 = np.array(
        [
            [e2avg, vue2m],
            [vde2m, e2delta],
        ],
        np.complex128,
    )
    return gamma_V_11 * gamma_nsm(N, cache)
