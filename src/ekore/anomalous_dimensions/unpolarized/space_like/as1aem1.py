"""
This file contains the O(as1aem1) Altarelli-Parisi splitting kernels.
"""

import numba as nb

from eko import constants
from .... import harmonics
from ....harmonics.constants import zeta2, zeta3


@nb.njit(cache=True)
def gamma_phq(N, sx):
    """Computes the O(as1aem1) photon-quark anomalous dimension

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
        2
        * (-4 - 12 * N - N**2 + 28 * N**3 + 43 * N**4 + 30 * N**5 + 12 * N**6)
        / ((-1 + N) * N**3 * (1 + N) ** 3)
    )
    tmp_S1 = (
        -4
        * (10 + 27 * N + 25 * N**2 + 13 * N**3 + 5 * N**4)
        / ((-1 + N) * N * (1 + N) ** 3)
    )
    tmp_S12 = 4 * (2 + N + N**2) / ((-1 + N) * N * (1 + N))
    tmp_S2 = 4 * (2 + N + N**2) / ((-1 + N) * N * (1 + N))

    return constants.CF * (tmp_const + tmp_S1 * S1 + tmp_S12 * S1**2 + tmp_S2 * S2)


@nb.njit(cache=True)
def gamma_qph(N, nf, sx):
    """Computes the O(as1aem1) quark-photon anomalous dimension

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
        -2
        * (
            4
            + 8 * N
            + 25 * N**2
            + 51 * N**3
            + 36 * N**4
            + 15 * N**5
            + 5 * N**6
        )
        / (N**3 * (1 + N) ** 3 * (2 + N))
    )
    tmp_S1 = 8 / N**2
    tmp_S12 = -4 * (2 + N + N**2) / (N * (1 + N) * (2 + N))
    tmp_S2 = 4 * (2 + N + N**2) / (N * (1 + N) * (2 + N))
    return (
        2
        * nf
        * constants.CA
        * constants.CF
        * (tmp_const + tmp_S1 * S1 + tmp_S12 * S1**2 + tmp_S2 * S2)
    )


@nb.njit(cache=True)
def gamma_gph(N):
    """Computes the O(as1aem1) gluon-photon anomalous dimension

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
        * (8 * (-4 + N * (-4 + N * (-5 + N * (-10 + N + 2 * N**2 * (2 + N))))))
        / (N**3 * (1 + N) ** 3 * (-2 + N + N**2))
    )


@nb.njit(cache=True)
def gamma_phg(N):
    """Computes the O(as1aem1) photon-gluon anomalous dimension

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
    return constants.TR / constants.CF / constants.CA * gamma_gph(N)


@nb.njit(cache=True)
def gamma_qg(N, nf, sx):
    """Computes the O(as1aem1) quark-gluon singlet anomalous dimension.

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
    return constants.TR / constants.CF / constants.CA * gamma_qph(N, nf, sx)


@nb.njit(cache=True)
def gamma_gq(N, sx):
    """Computes the O(as1aem1) gluon-quark singlet anomalous dimension.

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
    """Computes the O(as1aem1) photon-photon singlet anomalous dimension.

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
    return 4 * constants.CF * constants.CA * (nu * constants.eu2 + nd * constants.ed2)


@nb.njit(cache=True)
def gamma_gg():
    """Computes the O(as1aem1) gluon-gluon singlet anomalous dimension.

    Implements Eq. (31) of :cite:`deFlorian:2015ujt`.

    Parameters
    ----------

    Returns
    -------
        gamma_gg : complex
            O(as1aem1) gluon-gluon singlet anomalous dimension
            :math:`\\gamma_{gg}^{(1,1)}(N)`

    """
    return 4 * constants.TR


@nb.njit(cache=True)
def gamma_nsp(N, sx):
    """Computes the O(as1aem1) singlet-like non-singlet anomalous dimension.

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
    S1h = harmonics.S1(N / 2)
    S2h = harmonics.S2(N / 2)
    S3h = harmonics.S3(N / 2)
    S1p1h = harmonics.S1((N + 1.0) / 2)
    S2p1h = harmonics.S2((N + 1) / 2)
    S3p1h = harmonics.S3((N + 1) / 2)
    g3N = harmonics.g_functions.mellin_g3(N, S1)
    S1p2 = harmonics.polygamma.recursive_harmonic_sum(S1, N, 2, 1)
    g3Np2 = harmonics.g_functions.mellin_g3(N + 2, S1p2)
    result = (
        +32 * zeta2 * S1h
        - 32 * zeta2 * S1p1h
        + 8.0 / (N + N**2) * S2h
        - 4 * S3h
        + (24 + 16 / (N + N**2)) * S2
        - 32 * S3
        - 8.0 / (N + N**2) * S2p1h
        + S1
        * (
            +16 * (3 / N**2 - 3 / (1 + N) ** 2 + 2 * zeta2)
            - 16 * S2h
            - 32 * S2
            + 16 * S2p1h
        )
        + (
            -8
            + N
            * (
                -32
                + N * (-8 - 3 * N * (3 + N) * (3 + N**2) - 48 * (1 + N) ** 2 * zeta2)
            )
        )
        / (N**3 * (1 + N) ** 3)
        + 32 * (g3N + g3Np2)
        + 4 * S3p1h
        - 16 * zeta3
    )
    return constants.CF * result


@nb.njit(cache=True)
def gamma_nsm(N, sx):
    """Computes the O(as1aem1) valence-like non-singlet anomalous dimension.

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
    S1h = harmonics.S1(N / 2)
    S2h = harmonics.S2(N / 2)
    S3h = harmonics.S3(N / 2)
    S1p1h = harmonics.S1((N + 1.0) / 2)
    S2p1h = harmonics.S2((N + 1) / 2)
    S3p1h = harmonics.S3((N + 1) / 2)
    g3N = harmonics.g_functions.mellin_g3(N, S1)
    S1p2 = harmonics.polygamma.recursive_harmonic_sum(S1, N, 2, 1)
    g3Np2 = harmonics.g_functions.mellin_g3(N + 2, S1p2)

    result = (
        -32.0 * zeta2 * S1h
        - 8.0 / (N + N**2) * S2h
        + (24 + 16 / (N + N**2)) * S2
        + 8.0 / (N + N**2) * S2p1h
        + S1
        * (
            16 * (-1 / N**2 + 1 / (1 + N) ** 2 + 2 * zeta2)
            + 16 * S2h
            - 32 * S2
            - 16 * S2p1h
        )
        + (
            72
            + N
            * (
                96
                - 3 * N * (8 + 3 * N * (3 + N) * (3 + N**2))
                + 48 * N * (1 + N) ** 2 * zeta2
            )
        )
        / (3.0 * N**3 * (1 + N) ** 3)
        - 32 * (g3N + g3Np2)
        + 32.0 * zeta2 * S1p1h
        + 4 * S3h
        - 32 * S3
        - 4 * S3p1h
        - 16 * zeta3
    )
    return constants.CF * result
