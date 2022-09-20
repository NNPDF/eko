# -*- coding: utf-8 -*-
"""Contains the O(aem1) Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from .. import constants
from . import as1


@nb.njit(cache=True)
def gamma_phq(N):
    r"""
    Compute the leading-order photon-quark anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_phq : complex
        Leading-order photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(0)}(N)`
    """
    return as1.gamma_gq(N) / constants.CF


@nb.njit(cache=True)
def gamma_qph(N, nf):
    r"""
    Compute the leading-order quark-photon anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.
    But adding the :math:`N_C` and the :math:`2n_f` factors from :math:`\\theta` inside the
    definition of :math:`\\gamma_{q \\gamma}^{(0)}(N)`.

    Parameters
    ----------
      N : complex
        Mellin moment
      nf : int
        Number of active flavors

    Returns
    -------
      gamma_qph : complex
        Leading-order quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(0)}(N)`
    """
    return as1.gamma_qg(N, nf) / constants.TR * constants.NC


@nb.njit(cache=True)
def gamma_phph(nf):
    r"""
    Compute the leading-order photon-photon anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
      nf : int
        Number of active flavors

    Returns
    -------
      gamma_phph : complex
        Leading-order phton-photon anomalous dimension :math:`\\gamma_{\\gamma \\gamma}^{(0)}(N)`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return 4.0 / 3.0 * constants.NC * (nu * constants.eu2 + nd * constants.ed2)


@nb.njit(cache=True)
def gamma_ns(N, sx):
    r"""
    Compute the leading-order non-singlet QED anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
      N : complex
        Mellin moment
      s1 : complex
        S1(N)

    Returns
    -------
      gamma_ns : complex
        Leading-order non-singlet QED anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    s1 = sx[0]
    return as1.gamma_ns(N, s1) / constants.CF


@nb.njit(cache=True)
def gamma_singlet(N, nf, sx):
    r"""
    Compute the leading-order singlet anomalous dimension matrix.

    .. math::
        \gamma_S^{(0)} = \left(\begin{array}{cc}
        \gamma_{qq}^{(0)} & \gamma_{qg}^{(0)}\\
        \gamma_{gq}^{(0)} & \gamma_{gg}^{(0)}
        \end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    s1 : complex
        harmonic sum :math:`S_{1}`
    nf : int
        Number of active flavors

    Returns
    -------
    gamma_S_0 : numpy.ndarray
        Leading-order singlet anomalous dimension matrix :math:`\gamma_{S}^{(0)}(N)`

    See Also
    --------
    gamma_ns : :math:`\gamma_{qq}^{(0)}`
    gamma_qg : :math:`\gamma_{qg}^{(0)}`
    gamma_gq : :math:`\gamma_{gq}^{(0)}`
    gamma_gg : :math:`\gamma_{gg}^{(0)}`
    """
    e2avg = constants.e2avg(nf)
    vue2m = constants.vue2m(nf)
    vde2m = constants.vde2m(nf)
    e2delta = vde2m - vue2m + e2avg
    gamma_S_01 = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [
                0.0 + 0.0j,
                gamma_phph(nf),
                e2avg * gamma_phq(N),
                vue2m * gamma_phq(N),
            ],
            [
                0.0 + 0.0j,
                e2avg * gamma_qph(N, nf),
                e2avg * gamma_ns(N, sx),
                vue2m * gamma_ns(N, sx),
            ],
            [
                0.0 + 0.0j,
                vde2m * gamma_qph(N, nf),
                vde2m * gamma_ns(N, sx),
                e2delta * gamma_ns(N, sx),
            ],
        ],
        np.complex_,
    )
    return gamma_S_01


@nb.njit(cache=True)
def gamma_valence(N, nf, sx):
    r"""
    Compute the leading-order valence anomalous dimension matrix.

    .. math::
        \gamma_S^{(0)} = \left(\begin{array}{cc}
        \gamma_{qq}^{(0)} & \gamma_{qg}^{(0)}\\
        \gamma_{gq}^{(0)} & \gamma_{gg}^{(0)}
        \end{array}\right)

    Parameters
    ----------
    N : complex
        Mellin moment
    s1 : complex
        harmonic sum :math:`S_{1}`

    Returns
    -------
    gamma_S_0 : numpy.ndarray
        Leading-order singlet anomalous dimension matrix :math:`\gamma_{S}^{(0)}(N)`

    See Also
    --------
    gamma_ns : :math:`\gamma_{qq}^{(0)}`
    gamma_qg : :math:`\gamma_{qg}^{(0)}`
    gamma_gq : :math:`\gamma_{gq}^{(0)}`
    gamma_gg : :math:`\gamma_{gg}^{(0)}`
    """
    e2avg = constants.e2avg(nf)
    vue2m = constants.vue2m(nf)
    vde2m = constants.vde2m(nf)
    e2delta = vde2m - vue2m + e2avg
    gamma_V_01 = np.array(
        [
            [e2avg, vue2m],
            [vde2m, e2delta],
        ],
        np.complex_,
    )
    return gamma_V_01 * gamma_ns(N, sx)
