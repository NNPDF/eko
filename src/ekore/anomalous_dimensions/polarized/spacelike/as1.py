# -*- coding: utf-8 -*-
"""This file contains the leading-order Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants
from ...unpolarized.space_like.as1 import gamma_ns


@nb.njit(cache=True)
def gamma_qg(N, nf):
    """
    Computes the leading-order polarised quark-gluon anomalous dimension
    :cite:`Gluck:1995yr` (eq A.1)

    Parameters
    ----------
      N : complex
        Mellin moment
      nf : int
        Number of active flavors

    Returns
    -------
      gamma_qg : complex
        Leading-order polarised quark-gluon anomalous dimension :math:`\\gamma_{qg}^{(0)}(N)`
    """
    gamma = -(N - 1) / N / (N + 1)
    result = 2.0 * constants.TR * 2.0 * nf * gamma
    return result


@nb.njit(cache=True)
def gamma_gq(N):
    """
    Computes the leading-order polarised gluon-quark anomalous dimension
    :cite:`Gluck:1995yr` (eq A.1)

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_gq : complex
        Leading-order gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(0)}(N)`
    """
    gamma = -(N + 2) / N / (N + 1)
    result = 2.0 * constants.CF * gamma
    return result


@nb.njit(cache=True)
def gamma_gg(N, s1, nf):
    """
    Computes the leading-order polarised gluon-gluon anomalous dimension
    :cite:`Gluck:1995yr` (eq A.1)

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
      gamma_gg : complex
        Leading-order gluon-gluon anomalous dimension :math:`\\gamma_{gg}^{(0)}(N)`
    """
    gamma = -s1 + 2 / N / (N + 1)
    result = constants.CA * (-4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * constants.TR * nf
    return result


@nb.njit(cache=True)
def gamma_singlet(N, s1, nf):
    r"""
      Computes the leading-order polarised singlet anomalous dimension matrix

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
    """
    gamma_qq = gamma_ns(N, s1)
    gamma_S_0 = np.array(
        [[gamma_qq, gamma_qg(N, nf)], [gamma_gq(N), gamma_gg(N, s1, nf)]],
        np.complex_,
    )
    return gamma_S_0
