# -*- coding: utf-8 -*-
r"""
This file contains the implementation of the inverse Mellin transformation

.. math::
      \mathcal{M}^{-1}[f(N)](x) = \frac{1}{2\pi i} \int\limits_{c-i\infty}^{c+i\infty} x^{-N} f(N)

References
----------
"""

import numpy as np
import scipy.integrate as integrate

from eko import t_float

# TODO make this module numba/C-save

def inverse_Mellin_transform(f, path, jac, x, cut : t_float = 0.):
    """Inverse Mellin transformation

    Paramters
    ---------
      f : function
        Integration kernel
      path : function
        Integration path as a function :math:`p(t) : [0,1] \\to \\mathcal C : t \\to p(t)`
      jac : function
        Jacobian of integration path :math:`j(t) = \\frac{dp(t)}{dt}`
      x : float
        requested point in x-space
      cut : t_float
        Numeric cut-off parameter to the integration, the actual integration borders are
        determied by :math:`t\\in [c : 1-c]`

    Returns
    -------
      res : float
    """
    # integrate.quad can only do float, as it links to QUADPACK
    return integrate.quad(lambda u,f=f,path=path,jac=jac,x=x:
                # cast to real to allow integrate.quad
                np.real(
                    # prefactor                   * x^-N                       * f(N)
                    np.complex(0.,-1./(2.*np.pi)) * np.exp(-path(u)*np.log(x)) * f(path(u)) * jac(u)
                ),
                cut, 1.-cut)

def get_path_Talbot():
    """get Talbot path

    Returns
    -------
      path : function
        Talbot path function
        :math:`p_{\\text{Talbot}}(t) = \\pi*(2t-1) * cot(\\pi*(2t-1)) + i\\cdot\\pi*(2t-1)`
      jac : function
        derivative of Talbot path :math:`j(t) = \\frac{dp_{\\text{Talbot}}(t)}{dt}`
    """
    return (
        lambda t: np.complex(
                    1 if 0.5 == t else np.pi*(2.*t-1.)/np.tan(np.pi*(2.*t-1.)),
                    np.pi*(2.*t-1.)),
        lambda t: np.pi*2.*np.complex(
                    0 if 0.5 == t else 1./np.tan(np.pi*(2.*t-1.))
                                        - np.pi*(2.*t-1.)/(np.sin(np.pi*(2.*t-1.)))**2,
                    1.)
    )
