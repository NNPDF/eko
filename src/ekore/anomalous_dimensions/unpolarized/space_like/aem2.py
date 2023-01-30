"""
This file contains the O(aem2) Altarelli-Parisi splitting kernels.
"""

import numba as nb

from eko import constants
from . import as1aem1


@nb.njit(cache=True)
def gamma_phph(N, nf):
    """Computes the O(aem2) photon-photon singlet anomalous dimension.

    Implements Eq. (68) of :cite:`deFlorian:2016gvk`.

    Parameters
    ----------

    Returns
    -------
        gamma_gg : complex
            O(aem2) photon-photon singlet anomalous dimension
            :math:`\\gamma_{\\gamma \\gamma}^{(0,2)}(N)`

    """

    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return (
        constants.NC
        * (nu * constants.eu2**2 + nd * constants.ed2**2)
        * (as1aem1.gamma_gph(N) / constants.CF / constants.CA + 4)
    )


@nb.njit(cache=True)
def gamma_uph(N, nf, sx):
    """Computes the O(aem2) quark-photon anomalous dimension for up quarks.

    Implements Eq. (55) of :cite:`deFlorian:2016gvk` for q=u.

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
      gamma_uph : complex
        O(aem2) quark-photon anomalous dimension :math:`\\gamma_{u \\gamma}^{(0,2)}(N)`

    """
    return constants.eu2 * as1aem1.gamma_qph(N, nf, sx) / constants.CF


@nb.njit(cache=True)
def gamma_dph(N, nf, sx):
    """Computes the O(aem2) quark-photon anomalous dimension for down quarks.

    Implements Eq. (55) of :cite:`deFlorian:2016gvk` for q=d.

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
      gamma_dph : complex
        O(aem2) quark-photon anomalous dimension :math:`\\gamma_{d \\gamma}^{(0,2)}(N)`

    """
    return constants.ed2 * as1aem1.gamma_qph(N, nf, sx) / constants.CF


@nb.njit(cache=True)
def gamma_phu(N, nf, sx):
    """Computes the O(aem2) photon-quark anomalous dimension for up quarks.

    Implements Eq. (56) of :cite:`deFlorian:2016gvk` for q=u.

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
      gamma_phu : complex
        O(aem2) photon-quark anomalous dimension :math:`\\gamma_{\\gamma u}^{(0,2)}(N)`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    S1 = sx[0]
    tmp = (-16 * (-16 - 27 * N - 13 * N**2 - 8 * N**3)) / (
        9.0 * (-1 + N) * N * (1 + N) ** 2
    ) - 16 * (2 + 3 * N + 2 * N**2 + N**3) / (
        3.0 * (-1 + N) * N * (1 + N) ** 2
    ) * S1
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    return constants.eu2 * as1aem1.gamma_phq(N, sx) / constants.CF + eSigma2 * tmp


@nb.njit(cache=True)
def gamma_phd(N, nf, sx):
    """Computes the O(aem2) photon-quark anomalous dimension for down quarks.

    Implements Eq. (56) of :cite:`deFlorian:2016gvk` for q=d.

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
      gamma_phd : complex
        O(aem2) photon-quark anomalous dimension :math:`\\gamma_{\\gamma d}^{(0,2)}(N)`

    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    S1 = sx[0]
    tmp = (-16 * (-16 - 27 * N - 13 * N**2 - 8 * N**3)) / (
        9.0 * (-1 + N) * N * (1 + N) ** 2
    ) - 16 * (2 + 3 * N + 2 * N**2 + N**3) / (
        3.0 * (-1 + N) * N * (1 + N) ** 2
    ) * S1
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    return constants.ed2 * as1aem1.gamma_phq(N, sx) / constants.CF + eSigma2 * tmp


@nb.njit(cache=True)
def gamma_nspu(N, nf, sx):
    """Computes the O(aem2) singlet-like non-singlet anomalous dimension for up quarks.

    Implements sum of Eqs. (57-58) of :cite:`deFlorian:2016gvk` for q=u.

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
        gamma_nspu : complex
            O(aem2) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+,u}^{(0,2)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    tmp = (
        2
        * (-12 + 20 * N + 47 * N**2 + 6 * N**3 + 3 * N**4)
        / (9.0 * N**2 * (1 + N) ** 2)
        - 80 / 9 * S1
        + 16 / 3 * S2
    ) * eSigma2
    return constants.eu2 * as1aem1.gamma_nsp(N, sx) / constants.CF / 2 + tmp


@nb.njit(cache=True)
def gamma_nspd(N, nf, sx):
    """Computes the O(aem2) singlet-like non-singlet anomalous dimension for down quarks.

    Implements sum of Eqs. (57-58) of :cite:`deFlorian:2016gvk` for q=d.

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
        gamma_nspd : complex
            O(aem2) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+,d}^{(0,2)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    tmp = (
        2
        * (-12 + 20 * N + 47 * N**2 + 6 * N**3 + 3 * N**4)
        / (9.0 * N**2 * (1 + N) ** 2)
        - 80 / 9 * S1
        + 16 / 3 * S2
    ) * eSigma2
    return constants.ed2 * as1aem1.gamma_nsp(N, sx) / constants.CF / 2 + tmp


@nb.njit(cache=True)
def gamma_nsmu(N, nf, sx):
    """Computes the O(aem2) valence-like non-singlet anomalous dimension for up quarks.

    Implements difference between Eqs. (57-58) of :cite:`deFlorian:2016gvk` for q=u.

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
        gamma_nsp : complex
            O(aem2) valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-,u}^{(0,2)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    tmp = (
        2
        * (-12 + 20 * N + 47 * N**2 + 6 * N**3 + 3 * N**4)
        / (9.0 * N**2 * (1 + N) ** 2)
        - 80 / 9 * S1
        + 16 / 3 * S2
    ) * eSigma2
    return constants.eu2 * as1aem1.gamma_nsm(N, sx) / constants.CF / 2 + tmp


@nb.njit(cache=True)
def gamma_nsmd(N, nf, sx):
    """Computes the O(aem2) valence-like non-singlet anomalous dimension for down quarks.

    Implements difference between Eqs. (57-58) of :cite:`deFlorian:2016gvk` for q=d.

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
        gamma_nsp : complex
            O(aem2) valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-,d}^{(0,2)}(N)`

    """
    S1 = sx[0]
    S2 = sx[1]
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    tmp = (
        2
        * (-12 + 20 * N + 47 * N**2 + 6 * N**3 + 3 * N**4)
        / (9.0 * N**2 * (1 + N) ** 2)
        - 80 / 9 * S1
        + 16 / 3 * S2
    ) * eSigma2
    return constants.ed2 * as1aem1.gamma_nsm(N, sx) / constants.CF / 2 + tmp


@nb.njit(cache=True)
def gamma_ps(N, nf):
    """Computes the O(aem2) pure-singlet quark-quark anomalous dimension.

    Implements Eq. (59) of :cite:`deFlorian:2016gvk`.

    Parameters
    ----------
        N : complex
            Mellin moment
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_ps : complex
            O(aem2) pure-singlet quark-quark anomalous dimension
            :math:`\\gamma_{ps}^{(0,2)}(N)`

    """
    result = (
        -4
        * (2 + N * (5 + N))
        * (4 + N * (4 + N * (7 + 5 * N)))
        / ((-1 + N) * N**3 * (1 + N) ** 3 * (2 + N) ** 2)
    )
    return 2 * nf * constants.CA * result
