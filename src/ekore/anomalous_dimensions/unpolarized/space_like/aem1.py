"""The :math:`O(a_{em}^1)` Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants

from . import as1


@nb.njit(cache=True)
def gamma_phq(N):
    r"""Compute the leading-order photon-quark anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
    N : complex
        Mellin moment

    Returns
    -------
    complex
        Leading-order photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(0,1)}(N)`
    """
    return as1.gamma_gq(N) / constants.CF


@nb.njit(cache=True)
def gamma_qph(N, nf):
    r"""Compute the leading-order quark-photon anomalous dimension.

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
    complex
        Leading-order quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(0,1)}(N)`
    """
    return as1.gamma_qg(N, nf) / constants.TR * constants.NC


@nb.njit(cache=True)
def gamma_phph(nf):
    r"""Compute the leading-order photon-photon anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
    nf : int
        Number of active flavors

    Returns
    -------
    complex
        Leading-order phton-photon anomalous dimension :math:`\\gamma_{\\gamma \\gamma}^{(0,1)}(N)`
    """
    nu = constants.uplike_flavors(nf)
    nd = nf - nu
    return 4.0 / 3.0 * constants.NC * (nu * constants.eu2 + nd * constants.ed2)


@nb.njit(cache=True)
def gamma_ns(N, cache):
    r"""Compute the leading-order non-singlet QED anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
    N : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache

    Returns
    -------
    complex
        Leading-order non-singlet QED anomalous dimension :math:`\\gamma_{ns}^{(0,1)}(N)`
    """
    return as1.gamma_ns(N, cache) / constants.CF


@nb.njit(cache=True)
def gamma_singlet(N, nf, cache):
    r"""Compute the QED leading-order singlet anomalous dimension matrix.

    .. math::
        \\gamma_S^{(0)} = \\left(\begin{array}{cc}
        0 & 0 & 0 & 0 \\
        0 & \\gamma_{\\gamma \\gamma}^{(0,1)} & \\langle e^2 \rangle \\gamma_{\\gamma q}^{(0,1)} & \nu_u e^2_- \\gamma_{\\gamma q}^{(0,1)}\\
        0 & \\langle e^2 \rangle\\gamma_{q \\gamma}^{(0,1)} & \\langle e^2 \rangle \\gamma_{ns}^{(0,1)} & \nu_u e^2_- \\gamma_{ns}^{(0,1)}\\
        0 & \nu_d e^2_- \\gamma_{q \\gamma}^{(0,1)} & \nu_d e^2_- \\gamma_{ns}^{(0,1)} & e^2_\\Delta \\gamma_{ns}^{(0,1)}
        \\end{array}\right)

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
        Leading-order singlet anomalous dimension matrix :math:`\\gamma_{S}^{(0)}(N)`
    """
    e2avg, vue2m, vde2m, e2delta = constants.charge_combinations(nf)
    gamma_ph_q = gamma_phq(N)
    gamma_q_ph = gamma_qph(N, nf)
    gamma_nonsinglet = gamma_ns(N, cache)
    gamma_S_01 = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [
                0.0 + 0.0j,
                gamma_phph(nf),
                e2avg * gamma_ph_q,
                vue2m * gamma_ph_q,
            ],
            [
                0.0 + 0.0j,
                e2avg * gamma_q_ph,
                e2avg * gamma_nonsinglet,
                vue2m * gamma_nonsinglet,
            ],
            [
                0.0 + 0.0j,
                vde2m * gamma_q_ph,
                vde2m * gamma_nonsinglet,
                e2delta * gamma_nonsinglet,
            ],
        ],
        np.complex128,
    )
    return gamma_S_01


@nb.njit(cache=True)
def gamma_valence(N, nf, cache):
    r"""Compute the QED leading-order valence anomalous dimension matrix.

    .. math::
        \\gamma_V^{(0,1)} = \\left(\begin{array}{cc}
        \\langle e^2 \rangle \\gamma_{ns}^{(0,1)} & \nu_u e^2_- \\gamma_{ns}^{(0,1)}\\
        \nu_d e^2_- \\gamma_{ns}^{(0,1)} & e^2_\\Delta \\gamma_{ns}^{(0,1)}
        \\end{array}\right)

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
        Leading-order singlet anomalous dimension matrix :math:`\\gamma_{S}^{(0)}(N)`
    """
    e2avg, vue2m, vde2m, e2delta = constants.charge_combinations(nf)
    gamma_V_01 = np.array(
        [
            [e2avg, vue2m],
            [vde2m, e2delta],
        ],
        np.complex128,
    )
    return gamma_V_01 * gamma_ns(N, cache)
