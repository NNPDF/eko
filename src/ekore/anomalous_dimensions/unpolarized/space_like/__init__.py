r"""The unpolarized, space-like Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
The 3-loop references for the non-singlet :cite:`Moch:2004pa`
and singlet :cite:`Vogt:2004mw` case contain also the lower
order results. The results are also determined in Mellin space in
terms of the anomalous dimensions (note the additional sign!)

.. math::
    \gamma(N) = - \mathcal{M}[\mathbf{P}(x)](N)
"""

import numba as nb
import numpy as np

from eko import constants

from ....harmonics import cache as c
from . import aem1, aem2, as1, as1aem1, as2, as3, as4


@nb.njit(cache=True)
def gamma_ns(order, mode, n, nf, n3lo_ad_variation, use_fhmruvv=True):
    r"""Compute the tower of the non-singlet anomalous dimensions.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    mode : 10201 | 10101 | 10200
        sector identifier
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv: bool
        if True use the |FHMRUVV| N3LO anomalous dimensions

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions
    """
    cache = c.reset()
    # now combine
    gamma_ns = np.zeros(order[0], np.complex128)
    gamma_ns[0] = as1.gamma_ns(n, cache)
    # NLO and beyond
    if order[0] >= 2:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, cache)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, cache)
        else:
            raise NotImplementedError("Non-singlet sector is not implemented")
        gamma_ns[1] = gamma_ns_1
    # NNLO and beyond
    if order[0] >= 3:
        gamma_ns_2 = 0.0
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf, cache)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf, cache)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf, cache)
        gamma_ns[2] = gamma_ns_2
    # N3LO
    if order[0] >= 4:
        gamma_ns_3 = 0.0
        if use_fhmruvv:
            if mode == 10101:
                gamma_ns_3 = as4.fhmruvv.gamma_nsp(
                    n, nf, cache, variation=n3lo_ad_variation[4]
                )
            elif mode == 10201:
                gamma_ns_3 = as4.fhmruvv.gamma_nsm(
                    n, nf, cache, variation=n3lo_ad_variation[5]
                )
            elif mode == 10200:
                gamma_ns_3 = as4.fhmruvv.gamma_nsv(
                    n, nf, cache, variation=n3lo_ad_variation[6]
                )
        else:
            if mode == 10101:
                gamma_ns_3 = as4.gamma_nsp(n, nf, cache)
            elif mode == 10201:
                gamma_ns_3 = as4.gamma_nsm(n, nf, cache)
            elif mode == 10200:
                gamma_ns_3 = as4.gamma_nsv(n, nf, cache)
        gamma_ns[3] = gamma_ns_3
    return gamma_ns


@nb.njit(cache=True)
def gamma_singlet(order, n, nf, n3lo_ad_variation, use_fhmruvv=True):
    r"""Compute the tower of the singlet anomalous dimensions matrices.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv: bool
        if True use the |FHMRUVV| N3LO anomalous dimensions

    Returns
    -------
    numpy.ndarray
        singlet anomalous dimensions matrices
    """
    cache = c.reset()
    gamma_s = np.zeros((order[0], 2, 2), np.complex128)
    gamma_s[0] = as1.gamma_singlet(n, cache, nf)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf, cache)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf, cache)
    if order[0] >= 4:
        if use_fhmruvv:
            gamma_s[3] = as4.fhmruvv.gamma_singlet(n, nf, cache, n3lo_ad_variation)
        else:
            gamma_s[3] = as4.gamma_singlet(n, nf, cache, n3lo_ad_variation)
    return gamma_s


@nb.njit(cache=True)
def gamma_ns_qed(order, mode, n, nf, n3lo_ad_variation, use_fhmruvv=True):
    r"""Compute the grid of the QED non-singlet anomalous dimensions.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    mode : 10102 | 10103 | 10202 | 10203
        sector identifier
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv: bool
        if True use the |FHMRUVV| N3LO anomalous dimensions

    Returns
    -------
        gamma_ns : numpy.ndarray
            non-singlet QED anomalous dimensions
    """
    cache = c.reset()
    # now combine
    gamma_ns = np.zeros((order[0] + 1, order[1] + 1), np.complex128)
    gamma_ns[1, 0] = as1.gamma_ns(n, cache)
    gamma_ns[0, 1] = choose_ns_ad_aem1(mode, n, cache)
    gamma_ns[1, 1] = choose_ns_ad_as1aem1(mode, n, cache)
    # NLO and beyond
    if order[0] >= 2:
        if mode in [10102, 10103]:
            gamma_ns[2, 0] = as2.gamma_nsp(n, nf, cache)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        else:
            gamma_ns[2, 0] = as2.gamma_nsm(n, nf, cache)
    if order[1] >= 2:
        gamma_ns[0, 2] = choose_ns_ad_aem2(mode, n, nf, cache)
    # NNLO and beyond
    if order[0] >= 3:
        if mode in [10102, 10103]:
            gamma_ns[3, 0] = as3.gamma_nsp(n, nf, cache)
        elif mode in [10202, 10203]:
            gamma_ns[3, 0] = as3.gamma_nsm(n, nf, cache)
    if order[0] >= 4:
        if use_fhmruvv:
            if mode in [10102, 10103]:
                gamma_ns[4, 0] = as4.fhmruvv.gamma_nsp(
                    n, nf, cache, n3lo_ad_variation[4]
                )
            elif mode in [10202, 10203]:
                gamma_ns[4, 0] = as4.fhmruvv.gamma_nsm(
                    n, nf, cache, n3lo_ad_variation[5]
                )
        else:
            if mode in [10102, 10103]:
                gamma_ns[4, 0] = as4.gamma_nsp(n, nf, cache)
            elif mode in [10202, 10203]:
                gamma_ns[4, 0] = as4.gamma_nsm(n, nf, cache)
    return gamma_ns


@nb.njit(cache=True)
def choose_ns_ad_aem1(mode, n, cache):
    r"""Select the non-singlet anomalous dimension at O(aem1) with the correct
    charge factor.

    Parameters
    ----------
    mode : 10102 | 10202 | 10103 | 10203
        sector identifier
    n : complex
        Mellin variable
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions
    """
    if mode in [10102, 10202]:
        return constants.eu2 * aem1.gamma_ns(n, cache)
    elif mode in [10103, 10203]:
        return constants.ed2 * aem1.gamma_ns(n, cache)
    raise NotImplementedError("Non-singlet sector is not implemented")


@nb.njit(cache=True)
def choose_ns_ad_as1aem1(mode, n, cache):
    r"""Select the non-singlet anomalous dimension at O(as1aem1) with the
    correct charge factor.

    Parameters
    ----------
    mode : 10102 | 10202 | 10103 | 10203
        sector identifier
    n : complex
        Mellin variable
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions
    """
    if mode == 10102:
        return constants.eu2 * as1aem1.gamma_nsp(n, cache)
    elif mode == 10103:
        return constants.ed2 * as1aem1.gamma_nsp(n, cache)
    elif mode == 10202:
        return constants.eu2 * as1aem1.gamma_nsm(n, cache)
    elif mode == 10203:
        return constants.ed2 * as1aem1.gamma_nsm(n, cache)
    raise NotImplementedError("Non-singlet sector is not implemented")


@nb.njit(cache=True)
def choose_ns_ad_aem2(mode, n, nf, cache):
    r"""Select the non-singlet anomalous dimension at O(aem2) with the correct
    charge factor.

    Parameters
    ----------
    mode : 10102 | 10202 | 10103 | 10203
        sector identifier
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions
    """
    if mode == 10102:
        return constants.eu2 * aem2.gamma_nspu(n, nf, cache)
    elif mode == 10103:
        return constants.ed2 * aem2.gamma_nspd(n, nf, cache)
    elif mode == 10202:
        return constants.eu2 * aem2.gamma_nsmu(n, nf, cache)
    elif mode == 10203:
        return constants.ed2 * aem2.gamma_nsmd(n, nf, cache)
    raise NotImplementedError("Non-singlet sector is not implemented")


@nb.njit(cache=True)
def gamma_singlet_qed(order, n, nf, n3lo_ad_variation, use_fhmruvv=True):
    r"""Compute the grid of the QED singlet anomalous dimensions matrices.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv: bool
        if True use the |FHMRUVV| N3LO anomalous dimensions

    Returns
    -------
    numpy.ndarray
        singlet anomalous dimensions matrices
    """
    cache = c.reset()
    gamma_s = np.zeros((order[0] + 1, order[1] + 1, 4, 4), np.complex128)
    gamma_s[1, 0] = as1.gamma_singlet_qed(n, cache, nf)
    gamma_s[0, 1] = aem1.gamma_singlet(n, nf, cache)
    gamma_s[1, 1] = as1aem1.gamma_singlet(n, nf, cache)
    if order[0] >= 2:
        gamma_s[2, 0] = as2.gamma_singlet_qed(n, nf, cache)
    if order[1] >= 2:
        gamma_s[0, 2] = aem2.gamma_singlet(n, nf, cache)
    if order[0] >= 3:
        gamma_s[3, 0] = as3.gamma_singlet_qed(n, nf, cache)
    if order[0] >= 4:
        if use_fhmruvv:
            gamma_s[4, 0] = as4.fhmruvv.gamma_singlet_qed(
                n, nf, cache, n3lo_ad_variation
            )
        else:
            gamma_s[4, 0] = as4.gamma_singlet_qed(n, nf, cache, n3lo_ad_variation)
    return gamma_s


@nb.njit(cache=True)
def gamma_valence_qed(order, n, nf, n3lo_ad_variation, use_fhmruvv=True):
    r"""Compute the grid of the QED valence anomalous dimensions matrices.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    n : complex
        Mellin variable
    nf : int
        Number of active flavors
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv: bool
        if True use the |FHMRUVV| N3LO anomalous dimensions


    Returns
    -------
    numpy.ndarray
        valence anomalous dimensions matrices
    """
    cache = c.reset()
    gamma_v = np.zeros((order[0] + 1, order[1] + 1, 2, 2), np.complex128)
    gamma_v[1, 0] = as1.gamma_valence_qed(n, cache)
    gamma_v[0, 1] = aem1.gamma_valence(n, nf, cache)
    gamma_v[1, 1] = as1aem1.gamma_valence(n, nf, cache)
    if order[0] >= 2:
        gamma_v[2, 0] = as2.gamma_valence_qed(n, nf, cache)
    if order[1] >= 2:
        gamma_v[0, 2] = aem2.gamma_valence(n, nf, cache)
    if order[0] >= 3:
        gamma_v[3, 0] = as3.gamma_valence_qed(n, nf, cache)
    if order[0] >= 4:
        if use_fhmruvv:
            gamma_v[4, 0] = as4.fhmruvv.gamma_valence_qed(
                n, nf, cache, n3lo_ad_variation
            )
        else:
            gamma_v[4, 0] = as4.gamma_valence_qed(n, nf, cache)
    return gamma_v
