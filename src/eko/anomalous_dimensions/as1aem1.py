# -*- coding: utf-8 -*-
"""Contains the O(as1aem1) Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from .. import constants, harmonics
from ..harmonics.constants import zeta2, zeta3


@nb.njit(cache=True)
def gamma_phq(N, sx):
    r"""Compute the O(as1aem1) photon-quark anomalous dimension.

    Implements Eq. (36) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        N : complex
            Mellin moment
        sx : np array
            List of harmonic sums

    Returns
    -------
      gamma_phq : complex
        O(as1aem1) photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(1,1)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
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
def gamma_qph(N, nf, sx):
    r"""Compute the O(as1aem1) quark-photon anomalous dimension.

    Implements Eq. (26) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        N : complex
            Mellin moment
        nf : int
            Number of active flavors
        sx : np array
            List of harmonic sums

    Returns
    -------
      gamma_qph : complex
        O(as1aem1) quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(1,1)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
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
    r"""Compute the O(as1aem1) gluon-photon anomalous dimension.

    Implements Eq. (27) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_qph : complex
        O(as1aem1) gluon-photon anomalous dimension :math:`\\gamma_{g \\gamma}^{(1,1)}(N)`

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
    r"""Compute the O(as1aem1) photon-gluon anomalous dimension.

    Implements Eq. (30) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_qph : complex
        O(as1aem1) photon-gluon anomalous dimension :math:`\\gamma_{\\gamma g}^{(1,1)}(N)`

    """
    return constants.TR / constants.CF / constants.CA * constants.NC * gamma_gph(N)


@nb.njit(cache=True)
def gamma_qg(N, nf, sx):
    r"""Compute the O(as1aem1) quark-gluon singlet anomalous dimension.

    Implements Eq. (29) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        N : complex
            Mellin moment
        nf : int
            Number of active flavors
        sx : np array
            List of harmonic sums

    Returns
    -------
        gamma_qg : complex
            O(as1aem1) quark-gluon singlet anomalous dimension
            :math:`\\gamma_{qg}^{(1,1)}(N)`

    """
    return constants.TR / constants.CF / constants.CA * constants.NC * gamma_qph(N, nf, sx)


@nb.njit(cache=True)
def gamma_gq(N, sx):
    r"""Compute the O(as1aem1) gluon-quark singlet anomalous dimension.

    Implements Eq. (35) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        N : complex
            Mellin moment
        sx : np array
            List of harmonic sums

    Returns
    -------
        gamma_gq : complex
            O(as1aem1) gluon-quark singlet anomalous dimension
            :math:`\\gamma_{gq}^{(1,1)}(N)`

    """
    return gamma_phq(N, sx)


@nb.njit(cache=True)
def gamma_phph(nf):
    r"""Compute the O(as1aem1) photon-photon singlet anomalous dimension.

    Implements Eq. (28) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_gg : complex
            O(as1aem1) photon-photon singlet anomalous dimension
            :math:`\\gamma_{\\gamma \\gamma}^{(1,1)}(N)`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return 4.0 * constants.CF * constants.CA * (nu * constants.eu2 + nd * constants.ed2)


@nb.njit(cache=True)
def gamma_gg():
    r"""Compute the O(as1aem1) gluon-gluon singlet anomalous dimension.

    Implements Eq. (31) of :cite:`deFlorian:2015ujt`.

    Returns
    -------
        gamma_gg : complex
            O(as1aem1) gluon-gluon singlet anomalous dimension
            :math:`\\gamma_{gg}^{(1,1)}(N)`

    """
    return 4.0 * constants.TR * constants.NC


@nb.njit(cache=True)
def gamma_nsp(N, sx):
    r"""Compute the O(as1aem1) singlet-like non-singlet anomalous dimension.

    Implements sum of Eqs. (33-34) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        N : complex
            Mellin moment
        sx : np array
            List of harmonic sums

    Returns
    -------
        gamma_nsp : complex
            O(as1aem1) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(1)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
    S3 = sx[2]
    S1h = harmonics.S1(N / 2.0)
    S2h = harmonics.S2(N / 2.0)
    S3h = harmonics.S3(N / 2.0)
    S1p1h = harmonics.S1((N + 1.0) / 2.0)
    S2p1h = harmonics.S2((N + 1.0) / 2.0)
    S3p1h = harmonics.S3((N + 1.0) / 2.0)
    g3N = harmonics.g_functions.mellin_g3(N, S1)
    S1p2 = harmonics.polygamma.recursive_harmonic_sum(S1, N, 2, 1)
    g3Np2 = harmonics.g_functions.mellin_g3(N + 2.0, S1p2)
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
def gamma_nsm(N, sx):
    r"""Compute the O(as1aem1) valence-like non-singlet anomalous dimension.

    Implements difference between Eqs. (33-34) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------
        N : complex
            Mellin moment
        sx : np array
            List of harmonic sums

    Returns
    -------
        gamma_nsm : complex
            O(as1aem1) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-}^{(1,1)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
    S3 = sx[2]
    S1h = harmonics.S1(N / 2.0)
    S2h = harmonics.S2(N / 2.0)
    S3h = harmonics.S3(N / 2.0)
    S1p1h = harmonics.S1((N + 1.0) / 2.0)
    S2p1h = harmonics.S2((N + 1.0) / 2.0)
    S3p1h = harmonics.S3((N + 1.0) / 2.0)
    g3N = harmonics.g_functions.mellin_g3(N, S1)
    S1p2 = harmonics.polygamma.recursive_harmonic_sum(S1, N, 2, 1)
    g3Np2 = harmonics.g_functions.mellin_g3(N + 2.0, S1p2)

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
def gamma_singlet(N, nf, sx):
    r"""Compute the O(as1aem1) singlet sector.

    Parameters
    ----------
        N : complex
            Mellin moment
        nf : int
            Number of active flavors
        sx : np array
            List of harmonic sums

    Returns
    -------
        gamma_singlet : numpy.ndarray
            O(as1aem1) singlet anomalous dimension :math:`\\gamma_{S}^{(1,1)}(N,nf,sx)`
    """
    e2avg = constants.e2avg(nf)
    e2delta = constants.vde2m(nf) - constants.vue2m(nf) + constants.e2avg(nf)
    e2_tot = nf * e2avg
    vue2m = constants.vue2m(nf)
    vde2m = constants.vde2m(nf)
    gamma_S_11 = np.array(
        [
            [
                e2_tot * gamma_gg(),
                e2_tot * gamma_gph(N),
                e2avg * gamma_gq(N, sx),
                vue2m * gamma_gq(N, sx),
            ],
            [
                e2_tot * gamma_phg(N),
                gamma_phph(nf),
                e2avg * gamma_phq(N, sx),
                vue2m * gamma_phq(N, sx),
            ],
            [
                e2avg * gamma_qg(N, nf, sx),
                e2avg * gamma_qph(N, nf, sx),
                e2avg * gamma_nsp(N, sx),
                vue2m * gamma_nsp(N, sx),
            ],
            [
                vde2m * gamma_qg(N, nf, sx),
                vde2m * gamma_qph(N, nf, sx),
                vde2m * gamma_nsp(N, sx),
                e2delta * gamma_nsp(N, sx),
            ],
        ],
        np.complex_,
    )
    return gamma_S_11


@nb.njit(cache=True)
def gamma_valence(N, nf, sx):
    r"""Compute the O(as1aem1) valence sector.

    Parameters
    ----------
        N : complex
            Mellin moment
        nf : int
            Number of active flavors
        sx : np array
            List of harmonic sums

    Returns
    -------
        gamma_singlet : numpy.ndarray
            O(as1aem1) valence anomalous dimension :math:`\\gamma_{V}^{(1,1)}(N,nf,sx)`
    """
    e2avg = constants.e2avg(nf)
    vue2m = constants.vue2m(nf)
    vde2m = constants.vde2m(nf)
    e2delta = vde2m - vue2m + e2avg
    gamma_V_11 = np.array(
        [
            [e2avg, vue2m],
            [vde2m, e2delta],
        ],
        np.complex_,
    )
    return gamma_V_11 * gamma_nsm(N, sx)
