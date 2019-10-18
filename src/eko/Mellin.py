# -*- coding: utf-8 -*-
r"""
This file contains the implementation of the inverse Mellin transformation

It contains the actual transformation itself, as well as the necessary tools
such as the definition of paths.

The discussion in https://arxiv.org/pdf/1910.07049.pdf on p.5 below eq. 15
might be interesting.
"""

import numpy as np
import scipy.integrate as integrate

from eko import t_float,t_complex

# citing https://arxiv.org/pdf/1910.07049.pdf p.5 below eq. 15
# Thanks to the analytic continuation of Eq. (15) in theregion of the complex plane with Re(N)<0,
# when PDFs are expressed with this form, the integration contour in Eq. (12) can be optimised
# by bending towards negative values of Re(N), as depicted schematically in Figure 2, allowing
# for a faster convergence of the Mellin inversion integral. Such a strategy is adopted in
# DYRes and in Refs. [94,95]. As a drawback, PDFs need to be parameterised as in Eq. (14), or an
# approximation of the PDFs that follows this form has to be evaluated, which is significantly
# time consuming. In DYTurbo,the Mellin moments of PDFs are evaluated numerically,by using
# Gauss-Legendre quadrature to calculate the integrals of Eq. (13). However these integrals can
# be evaluated numerically only for Re(N)>0. As a consequence the integration contour of the
# inverse Mellin transform cannot be bent towards negative values of Re(N), and a standard contour
# along the straight line [c−i∞,c+i∞] is used (see Figure 2). This procedure results in a slower
# convergence of the integration in Eq. (12), for which about twice as many function evaluations
# are required, but it has the great advantage of allowing usage of PDFs with arbitrary
# parameterisation, without requiring knowledge of their functional form, and without requiring any
# time consuming evaluation of an approximation of PDFs in the form of Eq. (14).

# TODO make this module numba/C-save

# TODO replace inversion with something better? (t_float!)
def inverse_Mellin_transform(f, path, jac, x, cut : t_float = 0.):
    """Inverse Mellin transformation

    Parameters
    ----------
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

def get_path_Talbot(r : t_float = 1.):
    """get Talbot path

    Parameters
    ----------
      r : t_float
        scaling parameter - effectivly corresponds to the intersection of the path with the
        real axis

    Returns
    -------
      path : function
        Talbot path function
        :math:`p_{\\text{Talbot}}(t) = \\pi*(2t-1) * cot(\\pi*(2t-1)) + i\\cdot\\pi*(2t-1)`
      jac : function
        derivative of Talbot path :math:`j_{\\text{Talbot}}(t) = \\frac{dp_{\\text{Talbot}}(t)}{dt}`
    """
    return (
        lambda t: r*t_complex(np.complex(
                    1 if t == 0.5 else np.pi*(2.*t-1.)/np.tan(np.pi*(2.*t-1.)),
                    np.pi*(2.*t-1.))),
        lambda t: r*np.pi*2.*t_complex(np.complex(
                    0 if t == 0.5 else 1./np.tan(np.pi*(2.*t-1.))
                                        - np.pi*(2.*t-1.)/(np.sin(np.pi*(2.*t-1.)))**2,
                    1.))
    )

def get_path_line(m : t_float, c : t_float =1):
    """Get textbook path, i.e. a straight line parallel to imaginary axis

    Parameters
    ----------
      m : t_float
        half length of the path
      c : t_float
        intersection of path with real axis

    Returns
    -------
      path : function
        textbook path :math:`p_{\\text{line}}(t) = c + m \\cdot (2t - 1)`
      jac : function
        derivative of textbook path
        :math:`j_{\\text{line}}(t) = \\frac{dp_{\\text{line}}(t)}{dt} = 2m`
    """
    return (
      lambda t,max=m,c=c: t_complex(np.complex(c,max*(2*t-1))),
      lambda t,max=m:     t_complex(np.complex(0,max* 2     ))
    )
