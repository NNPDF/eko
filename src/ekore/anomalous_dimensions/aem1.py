"""
This file contains the O(aem1) Altarelli-Parisi splitting kernels.
"""

import numba as nb

from eko import constants
from . import as1


@nb.njit(cache=True)
def gamma_phq(N):
    """
    Computes the leading-order photon-quark anomalous dimension

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
      gamma_qph : complex
        Leading-order quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(0)}(N)`
    """
    return as1.gamma_qg(N, nf) / constants.TR * constants.NC


@nb.njit(cache=True)
def gamma_phph(nf):
    """
    Computes the leading-order photon-photon anomalous dimension

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

    return 2 / 3 * constants.NC * 2 * nf


@nb.njit(cache=True)
def gamma_ns(N, s1):
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
      gamma_ns : complex
        Leading-order non-singlet QED anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    return as1.gamma_ns(N, s1) / constants.CF
