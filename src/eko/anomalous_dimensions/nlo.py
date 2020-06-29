# -*- coding: utf-8 -*-
r"""
  This file contains the next-to-leading-order Altarelli-Parisi splitting kernels.
"""

#import numpy as np
import numba as nb

from eko import t_float, t_complex
#from eko.ekomath import harmonic_S1 as S1


@nb.njit
def gamma_nsp_1(N: t_complex, nf: int, CA: t_float, CF: t_float):
    """
      Computes the next-to-leading-order non-singlet singlet-like anomalous dimension.

      Implements Eq. (3.5) of :cite:`Moch:2004pa`.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : t_float
          Casimir constant of adjoint representation
        CF : t_float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_nsp_1 : t_complex
          Leading-order non-singlet anomalous dimension :math:`\\gamma_{ns}^{(1)+}(N)`
    """
    # TODO
    raise NotImplementedError("TODO")


@nb.njit
def gamma_nsm_1(N: t_complex, nf: int, CA: t_float, CF: t_float):
    """
      Computes the next-to-leading-order non-singlet valence-like anomalous dimension.

      Implements Eq. (3.6) of :cite:`Moch:2004pa`.

      Parameters
      ----------
        N : t_complex
          Mellin moment
        nf : int
          Number of active flavours
        CA : t_float
          Casimir constant of adjoint representation
        CF : t_float
          Casimir constant of fundamental representation

      Returns
      -------
        gamma_nsp_1 : t_complex
          Leading-order non-singlet anomalous dimension :math:`\\gamma_{ns}^{(1)-}(N)`
    """
    # TODO
    raise NotImplementedError("TODO")
