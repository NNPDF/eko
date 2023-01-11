"""This file contains the leading-order Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants


@nb.njit(cache=True)
def gamma_ns(N, s1):
    """
    Computes the leading-order non-singlet anomalous dimension.

    Implements Eq. (3.4) of :cite:`Moch:2004pa`.

    Parameters
    ----------
      N : complex
        Mellin moment
      s1 : complex
        harmonic sum :math:`S_{1}`

    Returns
    -------
      gamma_ns : complex
        Leading-order non-singlet anomalous dimension :math:`\\gamma_{ns}^{(0)}(N)`
    """
    gamma = -(3.0 - 4.0 * s1 + 2.0 / N / (N + 1.0))
    result = constants.CF * gamma
    return result


@nb.njit(cache=True)
def gamma_qg(N, nf):
    """
    Computes the leading-order quark-gluon anomalous dimension

    Implements Eq. (3.5) of :cite:`Vogt:2004mw`.

    Parameters
    ----------
      N : complex
        Mellin moment
      nf : int
        Number of active flavors

    Returns
    -------
      gamma_qg : complex
        Leading-order quark-gluon anomalous dimension :math:`\\gamma_{qg}^{(0)}(N)`
    """
    gamma = -(N**2 + N + 2.0) / (N * (N + 1.0) * (N + 2.0))
    result = 2.0 * constants.TR * 2.0 * nf * gamma
    return result


@nb.njit(cache=True)
def gamma_gq(N):
    """
    Computes the leading-order gluon-quark anomalous dimension

    Implements Eq. (3.5) of :cite:`Vogt:2004mw`.

    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_gq : complex
        Leading-order gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(0)}(N)`
    """
    gamma = -(N**2 + N + 2.0) / (N * (N + 1.0) * (N - 1.0))
    result = 2.0 * constants.CF * gamma
    return result


@nb.njit(cache=True)
def gamma_gg(N, s1, nf):
    """
    Computes the leading-order gluon-gluon anomalous dimension

    Implements Eq. (3.5) of :cite:`Vogt:2004mw`.

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
    gamma = s1 - 1.0 / N / (N - 1.0) - 1.0 / (N + 1.0) / (N + 2.0)
    result = constants.CA * (4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * constants.TR * nf
    return result


@nb.njit(cache=True)
def gamma_singlet(N, s1, nf):
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
    gamma_qq = gamma_ns(N, s1)
    gamma_S_0 = np.array(
        [[gamma_qq, gamma_qg(N, nf)], [gamma_gq(N), gamma_gg(N, s1, nf)]], np.complex_
    )
    return gamma_S_0
