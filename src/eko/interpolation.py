# -*- coding: utf-8 -*-
"""
This file provides all necessary tools for PDF interpolation.

"""
import numpy as np
from numpy.polynomial import Polynomial as P
import scipy.special as special
from eko import t_float

# rebuild beta function as default implementation in scipy does not allow complex arguments
#def _special_beta(alpha, beta):
#    return special.gamma(alpha)*special.gamma(beta)/special.gamma(alpha+beta)

def get_xgrid_linear_at_id(grid_size : int, xmin : t_float = 0., xmax : t_float = 1.):
    """Computes a linear grid on true x - corresponds to the flag `linear@id`

    This function is mainly for testing purpuse, as it is not physically relevant.

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value. Default is 0.
      xmax : t_float
        The maximum x value. Default is 1.

    Returns
    -------
      xgrid : array
        List of grid points in x-space
    """
    return np.linspace(xmin,xmax,num=grid_size,dtype=t_float)

def get_xgrid_Chebyshev_at_id(grid_size : int, xmin : t_float = 0., xmax : t_float = 1.):
    """Computes a Chebyshev-like spaced grid on true x - corresponds to the flag `Chebyshev@id`

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value. Default is 0.
      xmax : t_float
        The maximum x value. Default is 1.

    Returns
    -------
      xgrid : array
        List of grid points in x-space
    """
    return np.array([t_float(.5)*(xmax + xmin)
                      - .5*(xmax - xmin)*np.cos((2.*j+1)/(2.*grid_size)*np.pi)
            for j in range(grid_size)],dtype=t_float)

def get_xgrid_linear_at_log10(grid_size : int, xmin : t_float, xmax : t_float = 1.):
    """Computes a linear grid on log10(x) - corresponds to the flag `linear@log10`

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value.
      xmax : t_float
        The maximum x value. Default is 1.

    Returns
    -------
      xgrid : array
        List of grid points in x-space
    """
    return np.logspace(np.log10(xmin),np.log10(xmax),num=grid_size,dtype=t_float)

def get_xgrid_Chebyshev_at_log10(grid_size : int, xmin : t_float, xmax : t_float = 1.):
    """Computes a Chebyshev-like spaced grid on log10(x) - corresponds to the flag `Chebyshev@log10`

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value.
      xmax : t_float
        The maximum x value. Default is 1.

    Returns
    -------
      xgrid : array
        List of grid points in x-space
    """
    l = get_xgrid_Chebyshev_at_id(grid_size)
    r = [10**(np.log10(xmin) + j * (np.log10(xmax) - np.log10(xmin))) for j in l]
    return r

def get_Lagrange_iterpolators_x(x : t_float, xgrid, j : int):
    """Get a single Lagrange interpolator in x-space

    Lagragrange interpolation polynoms are defined by

    .. math::
      P_j(x) = \\prod_{k=1,k\\neq j}^{N_{grid}} \\frac{x - x_k}{x_j - x_k}

    Parameters
    ----------
      x : t_float
        Evaluated point in x-space
      xgrid : array
        Grid in x-space from which the interpolater is constructed
      j : int
        Number of chosen interpolater

    Returns
    -------
      p_j(x) : t_float
        Evaluated jth-polynom at x
    """
    l = len(xgrid)
    if l < 2 :
        raise "need at least 2 points"
    d = np.prod([1 if j == k else xgrid[j] - xgrid[k] for k in range(l)])
    n = np.prod([1 if j == k else x - xgrid[k] for k in range(l)])
    return n/d

def get_Lagrange_iterpolators_N(N,xgrid,j):
    """Get a single Lagrange interpolator in N-space

    Parameters
    ----------
      N : t_float
        Evaluated point in N-space
      xgrid : array
        Grid in x-space from which the interpolater is constructed
      j : int
        Number of chosen interpolater

    Returns
    -------
      p_j(N) : t_float
        Evaluated jth-polynom at N"""
    l = len(xgrid)
    if l < 2 :
        raise "need at least 2 points"
    d = np.prod([1 if j == k else xgrid[j] - xgrid[k] for k in range(l)])
    nx = np.prod([P([1]) if j == k else P([- xgrid[k],1]) for k in range(l)])
    n = np.sum([nx.coef[k]/(N+k) for k in range(l)])
    return n/d

def get_Lagrange_iterpolators_log_x(x,xgrid,j):
    """get j-th Lagrange interpolator of log10(grid) in x"""
    l = len(xgrid)
    x = np.log(x)
    xgrid = np.log(np.array(xgrid))
    if l < 2 : raise "need at least 2 points"
    d = np.prod([1 if j == k else xgrid[j] - xgrid[k] for k in range(l)])
    n = np.prod([1 if j == k else x - xgrid[k] for k in range(l)])
    return n/d

def get_Lagrange_iterpolators_log_N(N,xgrid,j):
    """get j-th Lagrange interpolator of log10(grid) in N"""
    l = len(xgrid)
    xgrid = np.log(np.array(xgrid))
    if l < 2 : raise "need at least 2 points"
    d = np.prod([1 if j == k else xgrid[j] - xgrid[k] for k in range(l)])
    from numpy.polynomial import Polynomial as P
    nx = np.prod([P([1]) if j == k else P([- xgrid[k],1]) for k in range(l)])
    n = np.sum([nx.coef[k]*(np.math.factorial(k))/N*(-1./N)**k for k in range(l)])
    return n/d
