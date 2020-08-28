# -*- coding: utf-8 -*-
"""
  This file contains the leading-order Altarelli-Parisi splitting kernels.

  For the sake of unification we keep a unique function signature for *all* coefficients.
"""

import numpy as np

import numba as nb

from eko.ekomath import harmonic_S1 as S1


@nb.njit
def gamma_ns_0(N, nf: int, CA: float, CF: float):  # pylint: disable=unused-argument
    """
      Computes the leading-order non-singlet anomalous dimension.

      Implements Eq. (3.4) of :cite:`Moch:2004pa`.

      Parameters
      ----------
        N : complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : float
          Casimir constant of adjoint representation
        CF : float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_ns_0 : complex
          Leading-order non-singlet anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    gamma = -(3 - 4 * S1(N) + 2 / N / (N + 1))
    result = CF * gamma
    return result


@nb.njit
def gamma_qg_0(N, nf: int, CA: float, CF: float):  # pylint: disable=unused-argument
    """
      Computes the leading-order quark-gluon anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.

      Parameters
      ----------
        N : complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : float
          Casimir constant of adjoint representation
        CF : float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_qg_0 : complex
          Leading-order quark-gluon anomalous dimension :math:`\\gamma_{qg}^{(0)}(N)`
    """
    gamma = -(N ** 2 + N + 2.0) / (N * (N + 1) * (N + 2))
    result = 2.0 * nf * gamma
    return result


@nb.njit
def gamma_gq_0(N, nf: int, CA: float, CF: float):  # pylint: disable=unused-argument
    """
      Computes the leading-order gluon-quark anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.

      Parameters
      ----------
        N : complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : float
          Casimir constant of adjoint representation
        CF : float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_gq_0 : complex
          Leading-order gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(0)}(N)`
    """
    gamma = -(N ** 2 + N + 2.0) / (N * (N + 1.0) * (N - 1.0))
    result = 2.0 * CF * gamma
    return result


@nb.njit
def gamma_gg_0(N, nf: int, CA: float, CF: float):  # pylint: disable=unused-argument
    """
      Computes the leading-order gluon-gluon anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.

      Parameters
      ----------
        N : complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : float
          Casimir constant of adjoint representation
        CF : float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_gg_0 : complex
          Leading-order gluon-gluon anomalous dimension :math:`\\gamma_{gg}^{(0)}(N)`
    """
    gamma = S1(N) - 1 / N / (N - 1) - 1 / (N + 1) / (N + 2)
    result = CA * (4.0 * gamma - 11.0 / 3.0) + 2.0 / 3.0 * nf
    return result


@nb.njit
def gamma_singlet_0(N, nf: int, CA: float, CF: float):
    r"""
      Computes the leading-order singlet anomalous dimension matrix

      .. math::
          \gamma_S^{(0)} = \left(\begin{array}{cc}
            \gamma_{qq}^{(0)} & \gamma_{qg}^{(0)}\\
            \gamma_{gq}^{(0)} & \gamma_{gg}^{(0)}
          \end{array}\right)

      Parameters
      ----------
        N : complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : float
          Casimir constant of adjoint representation
        CF : float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_S_0 : numpy.ndarray
          Leading-order singlet anomalous dimension matrix :math:`\gamma_{S}^{(0)}(N)`

      See Also
      --------
        gamma_ns_0 : :math:`\gamma_{qq}^{(0)}`
        gamma_qg_0 : :math:`\gamma_{qg}^{(0)}`
        gamma_gq_0 : :math:`\gamma_{gq}^{(0)}`
        gamma_gg_0 : :math:`\gamma_{gg}^{(0)}`
    """
    gamma_qq = gamma_ns_0(N, nf, CA, CF)
    gamma_qg = gamma_qg_0(N, nf, CA, CF)
    gamma_gq = gamma_gq_0(N, nf, CA, CF)
    gamma_gg = gamma_gg_0(N, nf, CA, CF)
    gamma_S_0 = np.array([[gamma_qq, gamma_qg], [gamma_gq, gamma_gg]], np.complex_)
    return gamma_S_0
