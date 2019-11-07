# -*- coding: utf-8 -*-
"""
This file provides all necessary tools for PDF interpolation.

The files provides the tools for generating grids and the
Lagrange interpolation polynomials

"""
import inspect
import numpy as np
import numba as nb
from numpy.polynomial import Polynomial as P
from eko import t_float

# Decorators
# can they be used with numba?? We'll see in the future
def check_xgrid(function_in):
    """ Check whether the argument xgrid is valud """
    #TODO Testing is needed to learn whether this makes numba useless
    def decorated_fun(*args, **kwargs):
        try:
            get_all_args = inspect.getcallargs(function_in, *args, **kwargs)
            function_out = function_in
            # TODO use signature
        except TypeError:
            get_all_args = inspect.getcallargs(function_in.py_func, *args, **kwargs)
            function_out = function_in.py_func
        xgrid = get_all_args["xgrid"]
        if len(xgrid) < 2:
            raise ValueError("The xgrid argument needs at least two values")
        return function_out(*args, **kwargs)

    return decorated_fun


@nb.njit
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
    # numba does not accept to set the type of the grid
    xgrid = np.linspace(xmin, xmax, grid_size)
    return xgrid

@nb.njit(parallel = True)
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
    for j in nb.prange(grid_size):
        cos_arg = (2.0 * j + 1) / (2.0 * grid_size) * np.pi
        new_point = avgx - deltax * np.cos(cos_arg)
        grid_points.append(new_point)
    xgrid = np.array(grid_points, dtype=t_float)
    return xgrid

@nb.jit(forceobj=True) # due to np.logspace
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

@nb.jit
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
@nb.njit
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
@nb.jit(forceobj=True) # Due to the usage of polynomial
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
    for k in nb.prange(len(xgrid)):
        n += num.coef[k] / (N + k)
    return n / den

@check_xgrid
@nb.njit
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

@check_xgrid
@nb.jit(forceobj=True)
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
    for k in nb.prange(len(log_xgrid)):
        ifac = np.math.factorial(k) / N
        powi = pow(-1.0 / N, k)
        n += num.coef[k] * ifac * powi
    return n / den

def get_Lagrange_basis_functions(xgrid_in, polynom_rank : int):
    # setup params
    xgrid = np.unique(xgrid_in)
    xgrid_size = len(xgrid)
    if not len(xgrid_in)  == xgrid_size:
        raise ValueError("xgrid is not unique")
    if xgrid_size < 2:
        raise ValueError("xgrid needs at least 2 points")
    if polynom_rank < 1:
        raise ValueError("need at least linear interpolation")
    if xgrid_size < polynom_rank:
        raise ValueError(f"to interpolate with rank {polynom_rank} we need at least that much points")

    # create blocks
    list_of_blocks = []
    for j in range(xgrid_size-1):
        kmin = max(0,j-polynom_rank//2) # borders are (]
        kmax = kmin + polynom_rank
        if kmax >= xgrid_size:
            kmax = xgrid_size - 1
            kmin = kmax - polynom_rank
        list_of_blocks.append((kmin,kmax))

    # setup basis functions
    list_of_basis_functions = [{"polynom_number": j, "areas": [{"lower_index": k, "reference_indices": None} for k in range(xgrid_size-1)]} for j in range(xgrid_size)]
    for j,current_block in enumerate(list_of_blocks):
        for k in range(current_block[0], current_block[1] + 1):
            list_of_basis_functions[k]["areas"][j]["reference_indices"] = current_block
    # compute
    def is_not_zero_sector(e):
        return not None == e["reference_indices"]
    for j,current_polynom in enumerate(list_of_basis_functions):
        # clean up zero sectors
        current_polynom["areas"] = list(filter(is_not_zero_sector,current_polynom["areas"]))
        # precompute coefficients
        xj = xgrid[j]
        for current_area in current_polynom["areas"]:
            denominator = 1.0
            coeffs = np.array([1])
            for k in range(current_area["reference_indices"][0],current_area["reference_indices"][1]+1):
                if k == j:
                    continue
                xk = xgrid[k]
                # Lagrange interpolation formula
                denominator *= (xj - xk)
                x_coeffs = np.insert(coeffs,0,0)
                Mxk_coeffs = -xk * coeffs
                Mxk_coeffs = np.append(Mxk_coeffs,np.zeros(len(x_coeffs)-len(Mxk_coeffs)))
                coeffs = x_coeffs + Mxk_coeffs
            # apply common denominator
            coeffs = coeffs / denominator
            current_area["coeffs"] = coeffs
            current_area["xmin"] = xgrid[current_area["lower_index"]]
            current_area["xmax"] = xgrid[current_area["lower_index"]+1]
            # clean up
            #del current_area["reference_indices"]
            #del current_area["lower_index"]
    # return
    return list_of_basis_functions

def evaluate_Lagrange_basis_function_x(x,conf):
    if not "areas" in conf or len(conf["areas"]) <= 0:
        raise ValueError("need some areas to explore")
    for current_area in conf["areas"]:
        if x <= current_area["xmin"] or x > current_area["xmax"]: # borders are (]
            continue
        polynom_rank = len(current_area["coeffs"])
        powers_in_x = np.array([x**k for k in range(polynom_rank)])
        return np.dot(current_area["coeffs"],powers_in_x)
    return 0.0

def evaluate_Lagrange_basis_function_N(N,conf):
    if not "areas" in conf or len(conf["areas"]) <= 0:
        raise ValueError("need some areas to explore")
    res = 0.0
    polynom_rank = len(conf["areas"][0]["coeffs"])
    for current_area in conf["areas"]:
        powers_x = []
        for j in range(polynom_rank):
            low = 0.0 if 0.0 == current_area["xmin"] else current_area["xmin"]**(N + j)
            up = current_area["xmax"]**(N + j)
            powers_x.append((up - low)/(N+j))
        res += np.dot(current_area["coeffs"],powers_x)
    return res