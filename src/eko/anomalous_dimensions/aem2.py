# -*- coding: utf-8 -*-
"""
This file contains the O(aem2) Altarelli-Parisi splitting kernels.

These expression have been obtained using the procedure described in the
`wiki <https://github.com/N3PDF/eko/wiki/Parse-NLO-expressions>`_
involving ``FormGet`` :cite:`Hahn:2016ebn`.
"""

import numba as nb
import numpy as np

from .. import constants
from . import as1aem1, harmonics


@nb.njit("c16(c16,u1)", cache=True)
def gamma_phph(N, nf):
    """
    Computes the O(aem2) photon-photon singlet anomalous dimension.

    Implements Eq. (68) of :cite:`deFlorian:2016gvk`.

    Parameters
    ----------

    Returns
    -------
        gamma_gg : complex
            O(as1aem1) photon-photon singlet anomalous dimension
            :math:`\\gamma_{\\gamma \\gamma}^{(1,1)}(N)`
    """

    nu = as1aem1.uplike_flavors(nf)
    nd = nf - nu
    return (
        constants.NC
        * (nu * constants.eu2**2 + nd * constants.ed2**2)
        * (as1aem1.gamma_gph(N) / constants.CF / constants.CA + 4)
    )


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_uph(N, nf, sx):
    """
    Computes the O(aem2) quark-photon anomalous dimension

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
      gamma_qph : complex
        O(as1aem1) quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(1,1)}(N)`
    """

    return constants.eu2 * as1aem1.gamma_qph(N, nf, sx) / constants.CF


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_dph(N, nf, sx):
    """
    Computes the O(aem2) quark-photon anomalous dimension

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
      gamma_qph : complex
        O(as1aem1) quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(1,1)}(N)`
    """

    return constants.ed2 * as1aem1.gamma_qph(N, nf, sx) / constants.CF


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_phu(N, nf, sx):
    """
    Computes the O(aem2) photon-quark anomalous dimension

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
      gamma_phq : complex
        O(as1aem1) photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(1,1)}(N)`
    """

    nu = as1aem1.uplike_flavors(nf)
    nd = nf - nu
    S1 = sx[0]
    tmp = (-16 * (-16 - 27 * N - 13 * N**2 - 8 * N**3)) / (
        9.0 * (-1 + N) * N * (1 + N) ** 2
    ) - 16 * (2 + 3 * N + 2 * N**2 + N**3) / (
        3.0 * (-1 + N) * N * (1 + N) ** 2
    ) * S1
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    return constants.eu2 * as1aem1.gamma_phq(N, sx) / constants.CF + eSigma2 * tmp


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_phd(N, nf, sx):
    """
    Computes theO(aem2) photon-quark anomalous dimension

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
      gamma_phq : complex
        O(as1aem1) photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(1,1)}(N)`
    """

    nu = as1aem1.uplike_flavors(nf)
    nd = nf - nu
    S1 = sx[0]
    tmp = (-16 * (-16 - 27 * N - 13 * N**2 - 8 * N**3)) / (
        9.0 * (-1 + N) * N * (1 + N) ** 2
    ) - 16 * (2 + 3 * N + 2 * N**2 + N**3) / (
        3.0 * (-1 + N) * N * (1 + N) ** 2
    ) * S1
    eSigma2 = constants.NC * (nu * constants.eu2 + nd * constants.ed2)
    return constants.ed2 * as1aem1.gamma_phq(N, sx) / constants.CF + eSigma2 * tmp


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_nspu(N, nf, sx):
    """
    Computes the O(aem2) singlet-like non-singlet anomalous dimension.

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
        gamma_nsp : complex
            O(as1aem1) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(1)}(N)`
    """

    S1 = sx[0]
    S2 = sx[1]
    nu = as1aem1.uplike_flavors(nf)
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


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_nspd(N, nf, sx):
    """
    Computes the O(aem2) singlet-like non-singlet anomalous dimension.

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
        gamma_nsp : complex
            O(as1aem1) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(1)}(N)`
    """

    S1 = sx[0]
    S2 = sx[1]
    nu = as1aem1.uplike_flavors(nf)
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


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_nsmu(N, nf, sx):
    """
    Computes the O(aem2) singlet-like non-singlet anomalous dimension.

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
            O(as1aem1) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(1)}(N)`
    """

    S1 = sx[0]
    S2 = sx[1]
    nu = as1aem1.uplike_flavors(nf)
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


@nb.njit("c16(c16,u1,c16[:])", cache=True)
def gamma_nsmd(N, nf, sx):
    """
    Computes the O(aem2) singlet-like non-singlet anomalous dimension.

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
            O(as1aem1) singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(1)}(N)`
    """

    S1 = sx[0]
    S2 = sx[1]
    nu = as1aem1.uplike_flavors(nf)
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


@nb.njit("c16(c16,u1)", cache=True)
def gamma_ps(N, nf):
    """
    Computes the O(aem2) pure-singlet quark-quark anomalous dimension.

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
            |NLO| pure-singlet quark-quark anomalous dimension
            :math:`\\gamma_{ps}^{(1)}(N)`
    """

    result = (
        -4
        * (2 + N * (5 + N))
        * (4 + N * (4 + N * (7 + 5 * N)))
        / ((-1 + N) * N**3 * (1 + N) ** 3 * (2 + N) ** 2)
    )
    return 2 * nf * constants.CA * result
