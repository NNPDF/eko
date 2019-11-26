# -*- coding: utf-8 -*-
r"""
This file contains the Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
  \mathbf{P}(x) = - \sum\limits_{n=0} a_s^{n+1} \mathbf P^{(n)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
The 3-loop references are given for the non-singlet :cite:`Moch:2004pa`
and singlet :cite:`Vogt:2004mw` case, which contain also the lower
order results. The results are also determined in Mellin space in
terms of the anomalous dimensions (note the additional sign!)

.. math::
  \gamma(N) = - \mathcal{M}[\mathbf{P}(x)](N)

"""
import numpy as np
import numba as nb
from numba import cffi_support
import _gsl_digamma
from eko import t_float, t_complex
# Prepare the cffi functions to be used within numba
cffi_support.register_module(_gsl_digamma)
c_digamma = _gsl_digamma.lib.digamma #pylint: disable=c-extension-no-member

@nb.njit
def gsl_digamma(N: t_complex):
    r = np.real(N)
    i = np.imag(N)
    out = np.empty(2)
    c_digamma(r, i, _gsl_digamma.ffi.from_buffer(out)) #pylint: disable=c-extension-no-member
    result = np.complex(out[0], out[1])
    return result

@nb.njit
def _S1(N: t_complex):
    r"""Computes the simple harmonic sum

    .. math::
      S_1(N) = \sum\limits_{j=0}^N \frac 1 j = \psi(N+1)+\gamma_E

    with :math:`\psi(M)` the digamma function and :math:`\gamma_E` the Euler-Mascheroni constant

    Parameters
    ----------
      N : t_complex
        Mellin moment

    Returns
    -------
      S_1 : t_complex
        (simple) Harmonic sum up to N :math:`S_1(N)`
    """
    result = gsl_digamma(N + 1)
    return result + np.euler_gamma


@nb.njit
def gamma_ns_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """Computes the leading-order non-singlet anomalous dimension.

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
    gamma = 2 * (_S1(N - 1) + _S1(N + 1)) - 3
    result = CF * gamma
    return result


@nb.njit
def gamma_ps_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """Computes the leading-order pure-singlet anomalous dimension

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
    """Computes the leading-order quark-gluon anomalous dimension

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
    gamma = _S1(N - 1) + 4.0 * _S1(N + 1) - 2.0 * _S1(N + 2) - 3.0 * _S1(N)
    result = 2.0 * nf * gamma
    return result


@nb.njit
def gamma_gq_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """Computes the leading-order gluon-quark anomalous dimension

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
    gamma = 2.0 * _S1(N - 2) - 4.0 * _S1(N - 1) - _S1(N + 1) + 3.0 * _S1(N)
    result = 2.0 * CF * gamma
    return result


@nb.njit
def gamma_gg_0(
    N: t_complex, nf: int, CA: t_float, CF: t_float
):  # pylint: disable=unused-argument
    """Computes the leading-order gluon-gluon anomalous dimension

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
    gamma = _S1(N - 2) - 2.0 * _S1(N - 1) - 2.0 * _S1(N + 1) + _S1(N + 2) + 3.0 * _S1(N)
    result = CA * (4.0 * gamma - 11.0 / 3.0) + 2.0 / 3.0 * nf
    return result

@nb.njit
def get_gamma_singlet_0(N : t_complex, nf: int, CA: t_float, CF: t_float):
    r"""Computes the leading-order singlet anomalous dimension matrix

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
    gamma_qq_0 = gamma_ns_0(N,nf,CA,CF) + gamma_ps_0(N,nf,CA,CF)
    gamma_S_0 = np.array([
      [gamma_qq_0, gamma_qg_0(N,nf,CA,CF)],
      [gamma_gq_0(N,nf,CA,CF), gamma_gg_0(N,nf,CA,CF)]
    ])
    return gamma_S_0

@nb.njit
def get_Eigensystem_gamma_singlet_0(N : t_complex, nf: int, CA: t_float, CF: t_float):
    r"""Computes the Eigensystem of the leading-order singlet anomalous dimension matrix

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
    gamma_S_0 = get_gamma_singlet_0(N,nf,CA,CF)
    # compute eigenvalues
    gamma_qq = gamma_S_0[0][0]
    gamma_qg = gamma_S_0[0][1]
    gamma_gq = gamma_S_0[1][0]
    gamma_gg = gamma_S_0[1][1]
    b = (gamma_qq + gamma_gg) / 2.0
    det = np.sqrt((gamma_qq - gamma_gg)**2 + 4*gamma_qg*gamma_gq) / 2.0
    lambda_p = b + det
    lambda_m = b - det
    # compute projectors
    identity = np.identity(2,dtype=t_complex)
    c = 1.0 / (lambda_p - lambda_m)
    e_p = c * (gamma_S_0 - lambda_m * identity)
    e_m = - c * (gamma_S_0 - lambda_p * identity)
    return lambda_p,lambda_m,e_p,e_m

if __name__ == "__main__":
    from scipy.special import digamma
    test_number = complex(0.4, 0.3)
    my_s1 = _S1(test_number)
    def pyt(c):
        n = c+1
        c_res = digamma(n)
        return c_res + np.euler_gamma
    np.testing.assert_almost_equal(my_s1, pyt(test_number))
