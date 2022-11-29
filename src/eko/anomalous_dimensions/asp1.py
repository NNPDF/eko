# -*- coding: utf-8 -*-
"""This file contains the leading-order Altarelli-Parisi splitting kernels."""

import numba as nb
import numpy as np

from eko import constants
from eko.anomalous_dimensions.as1 import gamma_ns as gamma_pns 


#@nb.njit(cache=True)
#def gamma_pns(N, s1):
    #Computes the leading-order non-singlet anomalous dimension for the polarised case. 
    #This is going to be the same expression as the one for the unpolarised case.

   # gamma = -(3.0 - 4.0 * s1 + 2.0 / N / (N + 1.0))
   # result = constants.CF * gamma
   # return result



@nb.njit(cache=True)
def gamma_pqg(N, nf):
    """
    Computes the leading-order polarised quark-gluon anomalous dimension
    #requires citation 

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
    gamma = (N-1) / N / (N+1)
    result = 2.0 * constants.TR * 2.0 * nf * gamma
    return result


@nb.njit(cache=True)
def gamma_pgq(N):
    """
    Computes the leading-order polarised gluon-quark anomalous dimension


    Parameters
    ----------
      N : complex
        Mellin moment

    Returns
    -------
      gamma_gq : complex
        Leading-order gluon-quark anomalous dimension :math:`\\gamma_{gq}^{(0)}(N)`
    """
    gamma =  (N+2) / N / (N+1)
    result = 2.0 * constants.CF * gamma
    return result



@nb.njit(cache=True)
def gamma_pgg(N, s1, nf):
    """
    Computes the leading-order polarised gluon-gluon anomalous dimension


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
    gamma = s1- 2 / N / (N+1)
    result = constants.CA * (4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * constants.TR * nf
    return result
#I think that there is a problem with the sign here (maybe the constants have the opp. sign)

@nb.njit(cache=True)
def gamma_psinglet(N, s1, nf):
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

      See Also
      --------
        gamma_ns : :math:`\gamma_{qq}^{(0)}`
        gamma_qg : :math:`\gamma_{qg}^{(0)}`
        gamma_gq : :math:`\gamma_{gq}^{(0)}`
        gamma_gg : :math:`\gamma_{gg}^{(0)}`
    """
    gamma_pqq = gamma_pns(N, s1)
    gamma_pS_0 = np.array(
        [[gamma_pqq, gamma_pqg(N, nf)], [gamma_pgq(N), gamma_pgg(N, s1, nf)]], np.complex_
    )
    return gamma_pS_0



   
       
       
