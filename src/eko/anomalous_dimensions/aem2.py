# -*- coding: utf-8 -*-
"""Contains the O(aem2) Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from .. import constants
from . import as1aem1


@nb.njit(cache=True)
def gamma_phph(N, nf):
    r"""Compute the O(aem2) photon-photon singlet anomalous dimension.

    Implements Eq. (68) of :cite:`deFlorian:2016gvk`.

    Parameters
    ----------
        N : complex
            Mellin moment
        nf : int
            Number of active flavors

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
    r"""Compute the O(aem2) quark-photon anomalous dimension for up quarks.

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
    r"""Compute the O(aem2) quark-photon anomalous dimension for down quarks.

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
    r"""Compute the O(aem2) photon-quark anomalous dimension for up quarks.

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
    tmp = (-16.0 * (-16.0 - 27.0 * N - 13.0 * N**2 - 8.0 * N**3)) / (
        9.0 * (-1.0 + N) * N * (1.0 + N) ** 2
    ) - 16.0 * (2.0 + 3.0 * N + 2.0 * N**2 + N**3) / (
        3.0 * (-1.0 + N) * N * (1.0 + N) ** 2
    ) * S1
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    return constants.eu2 * as1aem1.gamma_phq(N, sx) / constants.CF + eSigma2 * tmp


@nb.njit(cache=True)
def gamma_phd(N, nf, sx):
    r"""Compute the O(aem2) photon-quark anomalous dimension for down quarks.

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
    tmp = (-16.0 * (-16.0 - 27.0 * N - 13.0 * N**2 - 8.0 * N**3)) / (
        9.0 * (-1.0 + N) * N * (1.0 + N) ** 2
    ) - 16.0 * (2.0 + 3.0 * N + 2.0 * N**2 + N**3) / (
        3.0 * (-1.0 + N) * N * (1.0 + N) ** 2
    ) * S1
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    return constants.ed2 * as1aem1.gamma_phq(N, sx) / constants.CF + eSigma2 * tmp


@nb.njit(cache=True)
def gamma_nspu(N, nf, sx):
    r"""Compute the O(aem2) singlet-like non-singlet anomalous dimension for up quarks.

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
        2.0
        * (-12.0 + 20.0 * N + 47.0 * N**2 + 6.0 * N**3 + 3.0 * N**4)
        / (9.0 * N**2 * (1.0 + N) ** 2)
        - 80.0 / 9.0 * S1
        + 16.0 / 3.0 * S2
    ) * eSigma2
    return constants.eu2 * as1aem1.gamma_nsp(N, sx) / constants.CF / 2.0 + tmp


@nb.njit(cache=True)
def gamma_nspd(N, nf, sx):
    r"""Compute the O(aem2) singlet-like non-singlet anomalous dimension for down quarks.

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
        2.0
        * (-12.0 + 20.0 * N + 47.0 * N**2 + 6.0 * N**3 + 3.0 * N**4)
        / (9.0 * N**2 * (1.0 + N) ** 2)
        - 80.0 / 9.0 * S1
        + 16.0 / 3.0 * S2
    ) * eSigma2
    return constants.ed2 * as1aem1.gamma_nsp(N, sx) / constants.CF / 2.0 + tmp


@nb.njit(cache=True)
def gamma_nsmu(N, nf, sx):
    r"""Compute the O(aem2) valence-like non-singlet anomalous dimension for up quarks.

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
    r"""Compute the O(aem2) valence-like non-singlet anomalous dimension for down quarks.

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
        2.0
        * (-12.0 + 20.0 * N + 47.0 * N**2 + 6.0 * N**3 + 3.0 * N**4)
        / (9.0 * N**2 * (1.0 + N) ** 2)
        - 80.0 / 9.0 * S1
        + 16.0 / 3.0 * S2
    ) * eSigma2
    return constants.ed2 * as1aem1.gamma_nsm(N, sx) / constants.CF / 2.0 + tmp


@nb.njit(cache=True)
def gamma_ps(N, nf):
    r"""Compute the O(aem2) pure-singlet quark-quark anomalous dimension.

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
        -4.0
        * (2.0 + N * (5.0 + N))
        * (4.0 + N * (4.0 + N * (7.0 + 5.0 * N)))
        / ((-1.0 + N) * N**3 * (1.0 + N) ** 3 * (2.0 + N) ** 2)
    )
    return 2 * nf * constants.CA * result


@nb.njit(cache=True)
def gamma_singlet(N, nf, sx):
    r"""Compute the O(aem2) singlet sector.

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
            O(aem2) singlet anomalous dimension :math:`\\gamma_{S}^{(0,2)}(N,nf,sx)`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    vu = nu / nf
    vd = nd / nf
    e2avg = constants.e2avg(nf)
    e2m = constants.eu2 - constants.ed2
    gamma_S_02 = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [
                0.0 + 0.0j,
                gamma_phph(N, nf),
                vu * constants.eu2 * gamma_phu(N, nf, sx)
                + vd * constants.ed2 * gamma_phd(N, nf, sx),
                vu
                * (
                    constants.eu2 * gamma_phu(N, nf, sx)
                    - constants.ed2 * gamma_phd(N, nf, sx)
                ),
            ],
            [
                0.0 + 0.0j,
                vu * constants.eu2 * gamma_uph(N, nf, sx)
                + vd * constants.ed2 * gamma_dph(N, nf, sx),
                vu * constants.eu2 * gamma_nspu(N, nf, sx)
                + vd * constants.ed2 * gamma_nspd(N, nf, sx)
                + e2avg**2 * gamma_ps(N, nf),
                vu
                * (
                    constants.eu2 * gamma_nspu(N, nf, sx)
                    - constants.ed2 * gamma_nspd(N, nf, sx)
                    + e2m * e2avg * gamma_ps(N, nf)
                ),
            ],
            [
                0.0 + 0.0j,
                vd
                * (
                    constants.eu2 * gamma_uph(N, nf, sx)
                    - constants.ed2 * gamma_dph(N, nf, sx)
                ),
                vd
                * (
                    constants.eu2 * gamma_nspu(N, nf, sx)
                    - constants.ed2 * gamma_nspd(N, nf, sx)
                    + e2m * e2avg * gamma_ps(N, nf)
                ),
                vd * constants.eu2 * gamma_nspu(N, nf, sx)
                + vu * constants.ed2 * gamma_nspd(N, nf, sx)
                + vu * vd * e2m**2 * gamma_ps(N, nf),
            ],
        ],
        np.complex_,
    )
    return gamma_S_02


@nb.njit(cache=True)
def gamma_valence(N, nf, sx):
    r"""Compute the O(aem2) valence sector.

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
            O(aem2) valence anomalous dimension :math:`\\gamma_{V}^{(0,2)}(N,nf,sx)`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    vu = nu / nf
    vd = nd / nf
    gamma_V_02 = np.array(
        [
            [
                vu * constants.eu2 * gamma_nsmu(N, nf, sx)
                + vd * constants.ed2 * gamma_nsmd(N, nf, sx),
                vu
                * (
                    constants.eu2 * gamma_nsmu(N, nf, sx)
                    - constants.ed2 * gamma_nsmd(N, nf, sx)
                ),
            ],
            [
                vd
                * (
                    constants.eu2 * gamma_nsmu(N, nf, sx)
                    - constants.ed2 * gamma_nsmd(N, nf, sx)
                ),
                vd * constants.eu2 * gamma_nsmu(N, nf, sx)
                + vu * constants.ed2 * gamma_nsmd(N, nf, sx),
            ],
        ],
        np.complex_,
    )
    return gamma_V_02
