# -*- coding: utf-8 -*-
"""
This module contains the anomalous dimension :math:`\\gamma_{ns,-}^{(3)}`
"""
import numba as nb

from ...constants import CA, CF
from .gNSp import A_3, B_3p, gamma_ns_nf3


def B_3m(n, sx):
    """
    Parametrization of eq. 3.3 of :cite:`Davies:2016jie`.
    Note the :math:`\\deltaB^{(\\pm)}` is taken exact from eq 3.4.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        B_3m : complex
            |N3LO| valence-like non-singlet anomalous dimension part
            proportional to :math:`C_F (C_A - 2 C_F) nf^2`
    """
    S1, _ = sx[0]
    S2, Sm2 = sx[1]
    deltaB = (
        16
        / (81 * n**5 * (1 + n) ** 5)
        * (
            -54
            + 3 * n**8 * (-13 + 6 * S1)
            - 12 * n * (5 + 6 * S1)
            + 6 * n**7 * (-23 + 12 * S1)
            + n**4 * (73 + 696 * S1 - 252 * S1**2 - 252 * S2 - 252 * Sm2)
            + n**3 * (367 + 174 * S1 - 144 * S1**2 - 144 * S2 - 144 * Sm2)
            - n**2 * (-211 + 132 * S1 + 36 * S1**2 + 36 * S2 + 36 * Sm2)
            - n**6 * (475 - 474 * S1 + 72 * S1**2 + 72 * S2 + 72 * Sm2)
            - 3 * n**5 * (85 - 294 * S1 + 72 * S1**2 + 72 * S2 + 72 * Sm2)
        )
    )
    return B_3p(n, sx) + deltaB


@nb.njit(cache=True)
def gamma_nsm_nf2(n, sx):
    """
    Implements the valence-like non-singlet part proportional to :math:`nf^2`
    as in eq. 2.12 of :cite:`Davies:2016jie`.

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : list
            harmonic sums cache

    Returns
    -------
        g_ns_nf2 : complex
            |N3LO| valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-}^{(3)}|_{nf^2}`

    See Also
    --------
        A_3: :math:`A^{(3)}`
        B_3m: :math:`B^{(3)}_{-}`
    """
    return CF**2 * A_3(n, sx) + CF * (CA - 2 * CF) * B_3m(n, sx)


@nb.njit(cache=True)
def gamma_nsm_nf1(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_nsm_nf0(n, sx):
    return 0


@nb.njit(cache=True)
def gamma_nsm(n, nf, sx):
    """
    Computes the |N3LO| valence-like non-singlet anomalous dimension.

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            Number of active flavors
        sx : list
            harmonic sums cache

    Returns
    -------
        gamma_nsp : complex
            |N3LO| valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-}^{(3)}(N)`

    See Also
    --------
        gamma_nsm_nf0: :math:`\\gamma_{ns,-}^{(3)}|_{nf^0}`
        gamma_nsm_nf1: :math:`\\gamma_{ns,-}^{(3)}|_{nf^1}`
        gamma_nsm_nf2: :math:`\\gamma_{ns,-}^{(3)}|_{nf^2}`
        gamma_ns_nf3: :math:`\\gamma_{ns}^{(3)}|_{nf^3}`
    """
    return (
        gamma_nsm_nf0(n, sx)
        + nf * gamma_nsm_nf1(n, sx)
        + nf**2 * gamma_nsm_nf2(n, sx)
        + nf**3 * gamma_ns_nf3(n, sx)
    )
