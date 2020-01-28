"""
    Implements mathematical functions
"""

import numpy as np
import numba as nb
import mpmath as mp
from numba import cffi_support
import _gsl_digamma
from eko import t_complex

# Prepare the cffi functions to be used within numba
cffi_support.register_module(_gsl_digamma)
c_digamma = _gsl_digamma.lib.digamma  # pylint: disable=c-extension-no-member


#@nb.njit
def gsl_digamma(N: t_complex):
    """
      Wrapper around the cffi implementation of the digamma function.

      So it can take a complex both as input and output.

      Parameters
      ----------
        N : t_complex
          input

      Returns
      -------
        psi : t_complex
          psi(N)
    """
    r = np.real(N)
    i = np.imag(N)
    out = np.empty(2)
    c_digamma(
        r, i, _gsl_digamma.ffi.from_buffer(out)  # pylint: disable=c-extension-no-member
    )
    result = np.complex(out[0], out[1])
    return result


#@nb.njit
def harmonic_S1(N: t_complex):
    r"""
      Computes the simple harmonic sum

      .. math::
        S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi(N+1)+\gamma_E

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
    return mp.harmonic(N)
    #result = gsl_digamma(N + 1)
    #return result + np.euler_gamma
