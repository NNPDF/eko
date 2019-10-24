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

# TODO make this module numba/C-save

# TODO replace inversion with something better? (t_float!)
def inverse_mellin_transform(f, path, jac, x, cut : t_float = 0.):
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
    def integrand(u):
        pathu = path(u)
        prefactor = t_complex(complex(0.0, -1/2/np.pi))
        xexp = np.exp(- pathu*np.log(x))
        fofn = f(pathu)*jac(u)
        # integrate.quad can only do float, as it links to QUADPACK
        result = np.real(prefactor*xexp*fofn)
        return result
    result = integrate.quad(integrand, cut, 1.0-cut)
    return result

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

def get_path_edge(m:t_float,c:t_float=1.0):
    """Get edged path with an angle of 45°

    Parameters
    ----------
      m : t_float
        half length of the path
      c : t_float
        intersection of path with real axis

    Returns
    -------
      path : function
        edged path :math:`p_{\\text{edge}}(t)`
      jac : function
        derivative of edged path
        :math:`j_{\\text{edge}}(t) = \\frac{dp_{\\text{edge}}(t)}{dt}`
    """
    return (lambda t,max=m,c=c: c + (.5-t)*max*np.exp(np.complex(0,-np.pi*2./3.)) if t < .5 else
                                c + (t-.5)*max*np.exp(np.complex(0, np.pi*2./3.)),
            lambda t,max=m:            -max*np.exp(np.complex(0,-np.pi*2./3.)) if t < .5 else
                                        max*np.exp(np.complex(0, np.pi*2./3.))
            )
