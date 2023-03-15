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

from .... import harmonics
from . import aem1, aem2, as1, as1aem1, as2, as3, as4


@nb.njit(cache=True)
def gamma_ns(order, mode, n, nf):
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

    Returns
    -------
    numpy.ndarray
        non-singlet anomalous dimensions

    See Also
    --------
    ekore.anomalous_dimensions.unpolarized.space_like.as1.gamma_ns : :math:`\gamma_{ns}^{(0)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as2.gamma_nsp : :math:`\gamma_{ns,+}^{(1)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as2.gamma_nsm : :math:`\gamma_{ns,-}^{(1)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as3.gamma_nsp : :math:`\gamma_{ns,+}^{(2)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as3.gamma_nsm : :math:`\gamma_{ns,-}^{(2)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as3.gamma_nsv : :math:`\gamma_{ns,v}^{(2)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as4.gamma_nsp : :math:`\gamma_{ns,+}^{(3)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as4.gamma_nsm : :math:`\gamma_{ns,-}^{(3)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as4.gamma_nsv : :math:`\gamma_{ns,v}^{(3)}(N)`

    """
    # cache the s-es
    if order[0] >= 4:
        full_sx_cache = harmonics.compute_cache(n, 5, is_singlet=False)
        sx = np.array(
            [
                full_sx_cache[0][0],
                full_sx_cache[1][0],
                full_sx_cache[2][0],
                full_sx_cache[3][0],
            ]
        )
    else:
        sx = harmonics.sx(n, max_weight=order[0] + 1)
    # now combine
    gamma_ns = np.zeros(order[0], np.complex_)
    gamma_ns[0] = as1.gamma_ns(n, sx[0])
    # NLO and beyond
    if order[0] >= 2:
        if mode == 10101:
            gamma_ns_1 = as2.gamma_nsp(n, nf, sx)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        elif mode in [10201, 10200]:
            gamma_ns_1 = as2.gamma_nsm(n, nf, sx)
        else:
            raise NotImplementedError("Non-singlet sector is not implemented")
        gamma_ns[1] = gamma_ns_1
    # NNLO and beyond
    if order[0] >= 3:
        if mode == 10101:
            gamma_ns_2 = as3.gamma_nsp(n, nf, sx)
        elif mode == 10201:
            gamma_ns_2 = as3.gamma_nsm(n, nf, sx)
        elif mode == 10200:
            gamma_ns_2 = as3.gamma_nsv(n, nf, sx)
        gamma_ns[2] = gamma_ns_2
    # N3LO
    if order[0] >= 4:
        if mode == 10101:
            gamma_ns_3 = as4.gamma_nsp(n, nf, full_sx_cache)
        elif mode == 10201:
            gamma_ns_3 = as4.gamma_nsm(n, nf, full_sx_cache)
        elif mode == 10200:
            gamma_ns_3 = as4.gamma_nsv(n, nf, full_sx_cache)
        gamma_ns[3] = gamma_ns_3
    return gamma_ns


@nb.njit(cache=True)
def gamma_singlet(order, n, nf):
    r"""Compute the tower of the singlet anomalous dimensions matrices.

    Parameters
    ----------
    order : tuple(int,int)
        perturbative orders
    n : complex
        Mellin variable
    nf : int
        Number of active flavors

    Returns
    -------
    numpy.ndarray
        singlet anomalous dimensions matrices

    See Also
    --------
    ekore.anomalous_dimensions.unpolarized.space_like.as1.gamma_singlet : :math:`\gamma_{S}^{(0)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as2.gamma_singlet : :math:`\gamma_{S}^{(1)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as3.gamma_singlet : :math:`\gamma_{S}^{(2)}(N)`
    ekore.anomalous_dimensions.unpolarized.space_like.as4.gamma_singlet : :math:`\gamma_{S}^{(3)}(N)`

    """
    # cache the s-es
    if order[0] >= 4:
        full_sx_cache = harmonics.compute_cache(n, 5, is_singlet=False)
        sx = np.array(
            [
                full_sx_cache[0][0],
                full_sx_cache[1][0],
                full_sx_cache[2][0],
                full_sx_cache[3][0],
            ]
        )
    elif order[0] >= 3:
        # here we need only S1,S2,S3,S4
        sx = harmonics.sx(n, max_weight=order[0] + 1)
    else:
        sx = harmonics.sx(n, max_weight=order[0])

    gamma_s = np.zeros((order[0], 2, 2), np.complex_)
    gamma_s[0] = as1.gamma_singlet(n, sx[0], nf)
    if order[0] >= 2:
        gamma_s[1] = as2.gamma_singlet(n, nf, sx)
    if order[0] >= 3:
        gamma_s[2] = as3.gamma_singlet(n, nf, sx)
    if order[0] >= 4:
        gamma_s[3] = as4.gamma_singlet(n, nf, full_sx_cache)
    return gamma_s


@nb.njit(cache=True)
def gamma_ns_qed(order, mode, n, nf):
    r"""
    Compute the grid of the QED non-singlet anomalous dimensions.

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

    Returns
    -------
        gamma_ns : numpy.ndarray
            non-singlet QED anomalous dimensions

    See Also
    --------
        eko.anomalous_dimensions.as1.gamma_ns : :math:`\gamma_{ns}^{(0)}(N)`
        eko.anomalous_dimensions.as2.gamma_nsp : :math:`\gamma_{ns,+}^{(1)}(N)`
        eko.anomalous_dimensions.as2.gamma_nsm : :math:`\gamma_{ns,-}^{(1)}(N)`
        eko.anomalous_dimensions.as3.gamma_nsp : :math:`\gamma_{ns,+}^{(2)}(N)`
        eko.anomalous_dimensions.as3.gamma_nsm : :math:`\gamma_{ns,-}^{(2)}(N)`
        eko.anomalous_dimensions.as3.gamma_nsv : :math:`\gamma_{ns,v}^{(2)}(N)`
        eko.anomalous_dimensions.aem1.gamma_ns : :math:`\gamma_{ns}^{(0,1)}(N)`
        eko.anomalous_dimensions.aem1.gamma_ns : :math:`\gamma_{ns,v}^{(0,1)}(N)`
        eko.anomalous_dimensions.as1aem1.gamma_nsp : :math:`\gamma_{ns,p}^{(1,1)}(N)`
        eko.anomalous_dimensions.as1aem1.gamma_nsm : :math:`\gamma_{ns,m}^{(1,1)}(N)`
        eko.anomalous_dimensions.aem2.gamma_nspu : :math:`\gamma_{ns,pu}^{(0,2)}(N)`
        eko.anomalous_dimensions.aem2.gamma_nsmu : :math:`\gamma_{ns,mu}^{(0,2)}(N)`
        eko.anomalous_dimensions.aem2.gamma_nspd : :math:`\gamma_{ns,pd}^{(0,2)}(N)`
        eko.anomalous_dimensions.aem2.gamma_nsmd : :math:`\gamma_{ns,md}^{(0,2)}(N)`
    """
    # cache the s-es
    max_weight = max(order)
    if max_weight >= 3:
        # here we need only S1,S2,S3,S4
        sx = harmonics.sx(n, max_weight=max_weight + 1)
    else:
        sx = harmonics.sx(n, max_weight=3)
    sx_ns_qed = harmonics.compute_qed_ns_cache(n, sx[0])
    # now combine
    gamma_ns = np.zeros((order[0] + 1, order[1] + 1), np.complex_)
    gamma_ns[1, 0] = as1.gamma_ns(n, sx[0])
    gamma_ns[0, 1] = choose_ns_ad_aem1(mode, n, sx)
    gamma_ns[1, 1] = choose_ns_ad_as1aem1(mode, n, sx, sx_ns_qed)
    # NLO and beyond
    if order[0] >= 2:
        if mode in [10102, 10103]:
            gamma_ns[2, 0] = as2.gamma_nsp(n, nf, sx)
        # To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
        else:
            gamma_ns[2, 0] = as2.gamma_nsm(n, nf, sx)
    if order[1] >= 2:
        gamma_ns[0, 2] = choose_ns_ad_aem2(mode, n, nf, sx, sx_ns_qed)
    # NNLO and beyond
    if order[0] >= 3:
        if mode in [10102, 10103]:
            gamma_ns[3, 0] = as3.gamma_nsp(n, nf, sx)
        elif mode in [10202, 10203]:
            gamma_ns[3, 0] = as3.gamma_nsm(n, nf, sx)
    return gamma_ns


@nb.njit(cache=True)
def choose_ns_ad_aem1(mode, n, sx):
    r"""
    Select the non-singlet anomalous dimension at O(aem1) with the correct charge factor.

    Parameters
    ----------
        mode : 10102 | 10202 | 10103 | 10203
            sector identifier
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
    """
    if mode in [10102, 10202]:
        return constants.eu2 * aem1.gamma_ns(n, sx)
    elif mode in [10103, 10203]:
        return constants.ed2 * aem1.gamma_ns(n, sx)
    raise NotImplementedError("Non-singlet sector is not implemented")


@nb.njit(cache=True)
def choose_ns_ad_as1aem1(mode, n, sx, sx_ns_qed):
    r"""
    Select the non-singlet anomalous dimension at O(as1aem1) with the correct charge factor.

    Parameters
    ----------
        mode : 10102 | 10202 | 10103 | 10203
            sector identifier
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
    """
    if mode == 10102:
        return constants.eu2 * as1aem1.gamma_nsp(n, sx, sx_ns_qed)
    elif mode == 10103:
        return constants.ed2 * as1aem1.gamma_nsp(n, sx, sx_ns_qed)
    elif mode == 10202:
        return constants.eu2 * as1aem1.gamma_nsm(n, sx, sx_ns_qed)
    elif mode == 10203:
        return constants.ed2 * as1aem1.gamma_nsm(n, sx, sx_ns_qed)
    raise NotImplementedError("Non-singlet sector is not implemented")


@nb.njit(cache=True)
def choose_ns_ad_aem2(mode, n, nf, sx, sx_ns_qed):
    r"""
    Select the non-singlet anomalous dimension at O(aem2) with the correct charge factor.

    Parameters
    ----------
        mode : 10102 | 10202 | 10103 | 10203
            sector identifier
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
    """
    if mode == 10102:
        return constants.eu2 * aem2.gamma_nspu(n, nf, sx, sx_ns_qed)
    elif mode == 10103:
        return constants.ed2 * aem2.gamma_nspd(n, nf, sx, sx_ns_qed)
    elif mode == 10202:
        return constants.eu2 * aem2.gamma_nsmu(n, nf, sx, sx_ns_qed)
    elif mode == 10203:
        return constants.ed2 * aem2.gamma_nsmd(n, nf, sx, sx_ns_qed)
    raise NotImplementedError("Non-singlet sector is not implemented")


@nb.njit(cache=True)
def gamma_singlet_qed(order, n, nf):
    r"""
    Compute the grid of the QED singlet anomalous dimensions matrices.

    Parameters
    ----------
        order : tuple(int,int)
            perturbative orders
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_singlet : numpy.ndarray
            singlet anomalous dimensions matrices

    See Also
    --------
        eko.anomalous_dimensions.as1.gamma_singlet_qed : :math:`\gamma_{S}^{(0)}(N)`
        eko.anomalous_dimensions.as2.gamma_singlet_qed : :math:`\gamma_{S}^{(1)}(N)`
        eko.anomalous_dimensions.as3.gamma_singlet_qed : :math:`\gamma_{S}^{(2)}(N)`
        eko.anomalous_dimensions.aem1.gamma_singlet : :math:`\gamma_{S}^{(0,1)}(N)`
        eko.anomalous_dimensions.as1aem1.gamma_singlet : :math:`\gamma_{S}^{(1,1)}(N)`
        eko.anomalous_dimensions.aem2.gamma_singlet : :math:`\gamma_{S}^{(0,2)}(N)`
    """
    # cache the s-es
    max_weight = max(order)
    if max_weight >= 3:
        # here we need only S1,S2,S3,S4
        sx = harmonics.sx(n, max_weight=max_weight + 1)
    else:
        sx = harmonics.sx(n, max_weight=3)
    sx_ns_qed = harmonics.compute_qed_ns_cache(n, sx[0])
    gamma_s = np.zeros((order[0] + 1, order[1] + 1, 4, 4), np.complex_)
    gamma_s[1, 0] = as1.gamma_singlet_qed(n, sx[0], nf)
    gamma_s[0, 1] = aem1.gamma_singlet(n, nf, sx)
    gamma_s[1, 1] = as1aem1.gamma_singlet(n, nf, sx, sx_ns_qed)
    if order[0] >= 2:
        gamma_s[2, 0] = as2.gamma_singlet_qed(n, nf, sx)
    if order[1] >= 2:
        gamma_s[0, 2] = aem2.gamma_singlet(n, nf, sx, sx_ns_qed)
    if order[0] >= 3:
        gamma_s[3, 0] = as3.gamma_singlet_qed(n, nf, sx)
    return gamma_s


@nb.njit(cache=True)
def gamma_valence_qed(order, n, nf):
    r"""
    Compute the grid of the QED valence anomalous dimensions matrices.

    Parameters
    ----------
        order : tuple(int,int)
            perturbative orders
        n : complex
            Mellin variable
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_valence : numpy.ndarray
            valence anomalous dimensions matrices

    See Also
    --------
        eko.anomalous_dimensions.as1.gamma_valence_qed : :math:`\gamma_{V}^{(0)}(N)`
        eko.anomalous_dimensions.as2.gamma_valence_qed : :math:`\gamma_{V}^{(1)}(N)`
        eko.anomalous_dimensions.as3.gamma_valence_qed : :math:`\gamma_{V}^{(2)}(N)`
        eko.anomalous_dimensions.aem1.gamma_valence : :math:`\gamma_{V}^{(0,1)}(N)`
        eko.anomalous_dimensions.as1aem1.gamma_valence : :math:`\gamma_{V}^{(1,1)}(N)`
        eko.anomalous_dimensions.aem2.gamma_valence : :math:`\gamma_{V}^{(0,2)}(N)`
    """
    # cache the s-es
    max_weight = max(order)
    if max_weight >= 3:
        # here we need only S1,S2,S3,S4
        sx = harmonics.sx(n, max_weight=max_weight + 1)
    else:
        sx = harmonics.sx(n, max_weight=3)
    sx_ns_qed = harmonics.compute_qed_ns_cache(n, sx[0])
    gamma_v = np.zeros((order[0] + 1, order[1] + 1, 2, 2), np.complex_)
    gamma_v[1, 0] = as1.gamma_valence_qed(n, sx[0])
    gamma_v[0, 1] = aem1.gamma_valence(n, nf, sx)
    gamma_v[1, 1] = as1aem1.gamma_valence(n, nf, sx, sx_ns_qed)
    if order[0] >= 2:
        gamma_v[2, 0] = as2.gamma_valence_qed(n, nf, sx)
    if order[1] >= 2:
        gamma_v[0, 2] = aem2.gamma_valence(n, nf, sx, sx_ns_qed)
    if order[0] >= 3:
        gamma_v[3, 0] = as3.gamma_valence_qed(n, nf, sx)
    return gamma_v
