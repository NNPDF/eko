# -*- coding: utf-8 -*-
import numba as nb

from .. import constants
from . import as1


@nb.njit("c16(c16)", cache=True)
def gamma_phq_0(N):
    """
    Computes the leading-order photon-quark anomalous dimension

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_phq_0 : complex
        Leading-order photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(0)}(N)`
    """

    return as1.gamma_gq_0(N) / constants.CF


@nb.njit("c16(c16,u1)", cache=True)
def gamma_qph_0(N, nf: int):
    """
    Computes the leading-order quark-photon anomalous dimension

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
      gamma_qph_0 : complex
        Leading-order quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(0)}(N)`
    """
    return as1.gamma_qg_0(N, nf) / constants.TR * constants.NC


@nb.njit("c16()", cache=True)
def gamma_phph_0():
    """
    Computes the leading-order photon-photon anomalous dimension

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Returns
    -------
      gamma_phph_0 : complex
        Leading-order phton-photon anomalous dimension :math:`\\gamma_{\\gamma \\gamma}^{(0)}(N)`
    """

    return -4.0 / 3


@nb.njit("c16(c16,c16)", cache=True)
def gamma_ns_0(N, s1):
    """
    Computes the leading-order non-singlet QED anomalous dimension.

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
      N : complex
        Mellin moment
      s1 : complex
        S1(N)

    Returns
    -------
      gamma_ns_0 : complex
        Leading-order non-singlet QED anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    return as1.gamma_ns_0(N, s1) / constants.CF
