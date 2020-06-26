# -*- coding: utf-8 -*-
r"""
  This file contains the leading-order Altarelli-Parisi splitting kernels.
"""

import numpy as np
import numba as nb

from eko import t_float, t_complex
from eko.ekomath import harmonic_S1 as S1


@nb.njit
def gamma_ns_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """
      Computes the leading-order non-singlet anomalous dimension.

      Implements Eq. (3.4) of :cite:`Moch:2004pa`.
      For the sake of unification we keep a unique function signature for *all* coefficients.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours (which is actually not used here)
        CA : t_float
          Casimir constant of adjoint representation (which is actually not used here)
        CF : t_float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_ns_0 : t_complex
          Leading-order non-singlet anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    gamma = 2 * (S1(N - 1) + S1(N + 1)) - 3
    result = CF * gamma
    return result


@nb.njit
def gamma_ps_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """
      Computes the leading-order pure-singlet anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
      For the sake of unification we keep a unique function signature for *all* coefficients.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours (which is actually not used here)
        CA : t_float
          Casimir constant of adjoint representation (which is actually not used here)
        CF : t_float
          Casimir constant of fundamental representation (which is actually not used here)

      Returns
      -------
        gamma_ps_0 : t_complex
          Leading-order pure-singlet anomalous dimension :math:`\\gamma_{ps}^{(0)}(N)`
    """
    return 0.0


@nb.njit
def gamma_qg_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """
      Computes the leading-order quark-gluon anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
      For the sake of unification we keep a unique function signature for *all* coefficients.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : t_float
          Casimir constant of adjoint representation (which is actually not used here)
        CF : t_float
          Casimir constant of fundamental representation (which is actually not used here)

      Returns
      -------
        gamma_qg_0 : t_complex
          Leading-order quark-gluon anomalous dimension :math:`\\gamma_{qg}^{(0)}(N)`
    """
    gamma = S1(N - 1) + 4.0 * S1(N + 1) - 2.0 * S1(N + 2) - 3.0 * S1(N)
    result = 2.0 * nf * gamma
    return result


@nb.njit
def gamma_gq_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """
      Computes the leading-order gluon-quark anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
      For the sake of unification we keep a unique function signature for *all* coefficients.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours (which is actually not used here)
        CA : t_float
          Casimir constant of adjoint representation (which is actually not used here)
        CF : t_float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_qg_0 : t_complex
          Leading-order gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(0)}(N)`
    """
    gamma = 2.0 * S1(N - 2) - 4.0 * S1(N - 1) - S1(N + 1) + 3.0 * S1(N)
    result = 2.0 * CF * gamma
    return result


@nb.njit
def gamma_gg_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """
      Computes the leading-order gluon-gluon anomalous dimension

      Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
      For the sake of unification we keep a unique function signature for *all* coefficients.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : t_float
          Casimir constant of adjoint representation
        CF : t_float
          Casimir constant of fundamental representation (which is actually not used here)

      Returns
      -------
        gamma_qg_0 : t_complex
          Leading-order gluon-gluon anomalous dimension :math:`\\gamma_{gg}^{(0)}(N)`
    """
    gamma = S1(N - 2) - 2.0 * S1(N - 1) - 2.0 * S1(N + 1) + S1(N + 2) + 3.0 * S1(N)
    result = CA * (4.0 * gamma - 11.0 / 3.0) + 2.0 / 3.0 * nf
    return result


@nb.njit
def get_gamma_singlet_0(N: t_complex, nf: int, CA: t_float, CF: t_float):
    r"""
      Computes the leading-order singlet anomalous dimension matrix

      .. math::
          \gamma_S^{(0)} = \left(\begin{array}
            \gamma_{qq}^{(0)} & \gamma_{qg}^{(0)}\\
            \gamma_{gq}^{(0)} & \gamma_{gg}^{(0)}
          \end{array}\right)

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : t_float
          Casimir constant of adjoint representation
        CF : t_float
          Casimir constant of fundamental representation (which is actually not used here)

      Returns
      -------
        gamma_S_0 : np.array
          Leading-order singlet anomalous dimension matrix :math:`\gamma_{S}^{(0)}(N)`

      See Also
      --------
        - gamma_ns_0
        - gamma_ps_0
        - gamma_qg_0
        - gamma_gq_0
        - gamma_gg_0
    """
    gamma_qq = gamma_ns_0(N, nf, CA, CF) + gamma_ps_0(N, nf, CA, CF)
    gamma_qg = gamma_qg_0(N, nf, CA, CF)
    gamma_gq = gamma_gq_0(N, nf, CA, CF)
    gamma_gg = gamma_gg_0(N, nf, CA, CF)
    gamma_S_0 = np.array([[gamma_qq, gamma_qg], [gamma_gq, gamma_gg]])
    return gamma_S_0


@nb.njit
def get_Eigensystem_gamma_singlet_0(N: t_complex, nf: int, CA: t_float, CF: t_float):
    r"""
      Computes the Eigensystem of the leading-order singlet anomalous dimension matrix

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : t_float
          Casimir constant of adjoint representation
        CF : t_float
          Casimir constant of fundamental representation (which is actually not used here)

      Returns
      -------
        lambda_p : t_complex
          positive eigenvalue of the Leading-order singlet anomalous dimension matrix
          :math:`\gamma_{S}^{(0)}(N)`
        lambda_m : t_complex
          negative eigenvalue of the Leading-order singlet anomalous dimension matrix
          :math:`\gamma_{S}^{(0)}(N)`
        e_p : np.array
          projector for the positive eigenvalue of the Leading-order singlet anomalous
          dimension matrix :math:`\gamma_{S}^{(0)}(N)`
        e_m : np.array
          projector for the negative eigenvalue of the Leading-order singlet anomalous
          dimension matrix :math:`\gamma_{S}^{(0)}(N)`
    """
    gamma_S_0 = get_gamma_singlet_0(N, nf, CA, CF)
    # compute eigenvalues
    lambda_m, lambda_p = np.linalg.eigvals(gamma_S_0)
    # compute projectors
    identity = np.identity(2)
    c = 1.0 / (lambda_p - lambda_m)
    e_p = +c * (gamma_S_0 - lambda_m * identity)
    e_m = -c * (gamma_S_0 - lambda_p * identity)
    return lambda_p, lambda_m, e_p, e_m
