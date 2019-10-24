# -*- coding: utf-8 -*-
"""
This file provides all necessary tools for PDF interpolation.

The files provides the tools for generating grids and the
Lagrange interpolation polynomials

"""
import inspect
import numpy as np
from numpy.polynomial import Polynomial as P
from eko import t_float

# Decorators
# can they be used with numba?? We'll see in the future
def check_xgrid(function_in):
    """ Check whether the argument xgrid is valud """

    def decorated_fun(*args, **kwargs):
        get_all_args = inspect.getcallargs(
            function_in, *args, **kwargs
        )  # TODO use signature
        xgrid = get_all_args["xgrid"]
        if len(xgrid) < 2:
            raise ValueError("The xgrid argument needs at least two values")
        return function_in(*args, **kwargs)

    return decorated_fun


def get_xgrid_linear_at_id(grid_size: int, xmin: t_float = 0.0, xmax: t_float = 1.0):
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
    return np.linspace(xmin, xmax, num=grid_size, dtype=t_float)


def get_xgrid_Chebyshev_at_id(grid_size: int, xmin: t_float = 0.0, xmax: t_float = 1.0):
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
    avgx = (xmax + xmin) / 2.0
    deltax = (xmax - xmin) / 2.0
    grid_points = []
    for j in range(grid_size):
        cos_arg = (2.0 * j + 1) / (2.0 * grid_size) * np.pi
        new_point = avgx - deltax * np.cos(cos_arg)
        grid_points.append(new_point)
    xgrid = np.array(grid_points, dtype=t_float)
    return xgrid


def get_xgrid_linear_at_log(grid_size: int, xmin: t_float, xmax: t_float = 1.0):
    """Computes a linear grid on log(x) - corresponds to the flag `linear@log`

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
    return np.logspace(np.log10(xmin), np.log10(xmax), num=grid_size, dtype=t_float)


def get_xgrid_Chebyshev_at_log(grid_size: int, xmin: t_float, xmax: t_float = 1.0):
    """Computes a Chebyshev-like spaced grid on log(x) - corresponds to the flag `Chebyshev@log`

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
    cheb_grid = get_xgrid_Chebyshev_at_id(grid_size)
    exp_arg = np.log(xmin) + cheb_grid * (np.log(xmax) - np.log(xmin))
    xgrid = np.exp(exp_arg)
    return xgrid


@check_xgrid
def get_Lagrange_interpolators_x(x: t_float, xgrid, j: int):
    """Get a single Lagrange interpolator in true x-space  - corresponds to the flag `Lagrange@id`

    Lagragrange interpolation polynoms are defined by

    .. math::
      P_j^{\\text{id}}(x) = \\prod_{k=1,k\\neq j}^{N_{grid}} \\frac{x - x_k}{x_j - x_k}

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
    xj = xgrid[j]
    result = 1.0
    for k, xk in enumerate(xgrid):
        if k == j:
            continue
        num = x - xk
        den = xj - xk
        result *= num / den
    return result


@check_xgrid
def get_Lagrange_interpolators_N(N, xgrid, j):
    """Get a single Lagrange interpolator in N-space - corresponds to the flag `Lagrange@id`

    .. math::
      \\tilde P_j^{\\text{id}}(N)

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
        Evaluated jth-polynom at N
    """
    xj = xgrid[j]
    num = 1.0
    den = 1.0
    for k, xk in enumerate(xgrid):
        if k != j:
            den *= xj - xk
            num *= P([-xk, 1.0])
    n = 0.0
    for k in range(len(xgrid)):
        n += num.coef[k] / (N + k)
    return n / den


@check_xgrid
def get_Lagrange_interpolators_log_x(x, xgrid, j):
    """Get a single Lagrange interpolator in logarithmic x-space\
       - corresponds to the flag `Lagrange@log`

    Lagragrange interpolation polynoms are defined by

    .. math::
      P_j^{\\ln}(x) = \\prod_{k=1,k\\neq j}^{N_{grid}} \\frac{\\ln(x) - \\ln(x_k)}
                                                             {\\ln(x_j) - \\ln(x_k)}

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
    log_xgrid = np.log(xgrid)
    logx = np.log(x)
    return get_Lagrange_interpolators_x(logx, log_xgrid, j)


def get_Lagrange_interpolators_log_N(N, xgrid, j):
    """Get a single, logarithmic Lagrange interpolator in N-space\
       - corresponds to the flag `Lagrange@log`

    .. math::
      \\tilde P_j^{\\ln}(N)

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
        Evaluated jth-polynom at N
    """
    log_xgrid = np.log(xgrid)
    xj = log_xgrid[j]
    num = 1.0
    den = 1.0
    for k, xk in enumerate(log_xgrid):
        if k != j:
            den *= xj - xk
            num *= P([-xk, 1])
    n = 0.0
    for k in range(len(log_xgrid)):
        ifac = np.math.factorial(k) / N
        powi = pow(-1.0 / N, k)
        n += num.coef[k] * ifac * powi
    return n / den
