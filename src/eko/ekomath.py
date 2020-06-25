"""
    Implements the cffi ports from gsl for higher mathematical functions.

    Used libraries:

      - `cffi <https://cffi.readthedocs.io/en/latest/>`_
      - `gsl <https://www.gnu.org/software/gsl/doc/html/index.html>`_
"""

import numpy as np
import numba as nb
from numba.core.typing import cffi_utils
import _gsl_digamma
from eko import t_complex

# Prepare the cffi functions to be used within numba
cffi_utils.register_module(_gsl_digamma)
c_digamma = _gsl_digamma.lib.digamma  # pylint: disable=no-member


@nb.njit
def gsl_digamma(N: t_complex):
    """
      Wrapper around the gsl implementation via cffi of the
      `digamma function <https://en.wikipedia.org/wiki/Digamma_function>`_.

      The wrapper allows both input and output to be complex.
      Note that the `SciPy implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.digamma.html>`_
      does not allow for complex inputs.

      Parameters
      ----------
        N : t_complex
          input

      Returns
      -------
        psi : t_complex
          digamma function :math:`\\psi_0(N)`

      Notes
      -----
        `GSL documentation <https://www.gnu.org/software/gsl/doc/html/specfunc.html#psi-digamma-function>`_

        Note that, although not listed in the documentation, there **is** a
        `complex version <http://git.savannah.gnu.org/cgit/gsl.git/tree/specfunc/gsl_sf_psi.h#n76>`_
    """  # pylint: disable=line-too-long
    r = np.real(N)
    i = np.imag(N)
    out = np.empty(2)
    c_digamma(r, i, _gsl_digamma.ffi.from_buffer(out))
    result = np.complex(out[0], out[1])
    return result


@nb.njit
def harmonic_S1(N: t_complex):
    r"""
      Computes the simple harmonic sum.

      .. math::
        S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi_0(N+1)+\gamma_E

      with :math:`\psi_0(N)` the digamma function and :math:`\gamma_E` the
      Euler-Mascheroni constant.

      Parameters
      ----------
        N : t_complex
          Mellin moment

      Returns
      -------
        S_1 : t_complex
          (simple) Harmonic sum :math:`S_1(N)`

      See Also
      --------
        gsl_digamma :
    """
    result = gsl_digamma(N + 1)
    return result + np.euler_gamma
