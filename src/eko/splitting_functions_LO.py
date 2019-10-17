# -*- coding: utf-8 -*-
r"""
This file contains the Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
      \frac{dq_{ns}^i}{d\ln\mu^2}
      = - \sum\limits_{n=0} a^{n+1} P_{ns}^{(n),i} q_{ns}^i \
          \quad \text{with}~ a = \frac{\alpha_S1(\mu^2)}{4\pi}


References
----------
  The 3-loop references are for the non-singlet :cite:`Moch:2004pa`
  and singlet :cite:`Vogt:2004mw` case.
  They also contain the lower order results

"""
from numpy import euler_gamma
from scipy.special import digamma
from eko import t_float, t_complex

def _S1(N : t_complex):
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
      S : t_complex
        (simple) Harmonic sum up to N :math:`S_1(N)`
    """
    return digamma(N + 1) + euler_gamma

def gamma_ns_0(N : t_complex, nf : int, CA : t_float, CF : t_float): # pylint: disable=unused-argument
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
    f = 2*(_S1(N-1) + _S1(N+1)) - 3
    return CF * f

def gamma_ps_0(N : t_complex, nf : int, CA : t_float, CF : t_float): # pylint: disable=unused-argument
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
    return 0.

def gamma_qg_0(N : t_complex, nf : int, CA : t_float, CF : t_float): # pylint: disable=unused-argument
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
    f = _S1(N-1)+ 4.*_S1(N+1) - 2.*_S1(N+2) - 3.*_S1(N)
    return 2.*nf*f

def gamma_gq_0(N : t_complex, nf : int, CA : t_float, CF : t_float): # pylint: disable=unused-argument
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
    f = 2.*_S1(N-2) - 4.*_S1(N-1) - _S1(N+1) + 3.*_S1(N)
    return 2.*CF*f

def gamma_gg_0(N : t_complex, nf : int, CA : t_float, CF : t_float): # pylint: disable=unused-argument
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
    f = _S1(N-2) - 2.*_S1(N-1) - 2.*_S1(N+1) + _S1(N+2) + 3.*_S1(N)
    return CA * (4. * f - 11./3.) + 2./3. * nf
