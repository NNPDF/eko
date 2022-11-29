# -*- coding: utf-8 -*-
"""
This file contains the O(aem1) Altarelli-Parisi splitting kernels.
"""

import numba as nb

from eko import constants
from eko.anomalous_dimensions import asp1
from eko.anomalous_dimensions.aem1 import gamma_phph

@nb.njit(cache=True)
def gamma_phpq(N):
    """
    Computes the leading-order polarised photon-quark anomalous dimension

    Implements Eq. (2.5) of :cite:`Carrazza:2015dea`.

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_phq : complex
        Leading-order polarised photon-quark anomalous dimension :math:`\\gamma_{\\gamma q}^{(0)}(N)`
    """

    return asp1.gamma_pgq(N) / constants.CF


@nb.njit(cache=True)
def gamma_pqph(N, nf):
    """
    Computes the leading-order polarised quark-photon anomalous dimension

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
        Leading-order polarised quark-photon anomalous dimension :math:`\\gamma_{q \\gamma}^{(0)}(N)`
    """
    return asp1.gamma_pqg(N, nf) / constants.TR * constants.NC



@nb.njit(cache=True)
def gamma_pns(N, s1):
    """
    Computes the leading-order polarised non-singlet QED anomalous dimension.

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
        Leading-order polarised non-singlet QED anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    return asp1.gamma_pns(N, s1) / constants.CF
