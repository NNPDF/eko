# -*- coding: utf-8 -*-
"""
This file provides all necessary tools for PDF interpolation.

"""
import numpy as np
from eko import t_float

def get_xgrid_linear_at_id(grid_size : int, xmin : t_float, xmax : t_float = 1.):
    """Computes a linear grid on true x - corresponds to the flag `linear@id`

    This function is mainly for testing purpuse, as it is not physically relevant.

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value.
      xmax : t_float
        The maximum x value. Default is 1.
    """
    return np.linspace(xmin,xmax,num=grid_size,dtype=t_float)

def get_xgrid_linear_at_log(grid_size : int, xmin : t_float, xmax : t_float = 1.):
    """Computes a linear grid on log(x) - corresponds to the flag `linear@log`

    Here log refers to the decimal logarithm `np.log10`.

    Parameters
    ----------
      grid_size : int
        The total size of the grid.
      xmin : t_float
        The minimum x value.
      xmax : t_float
        The maximum x value. Default is 1.
    """
    return np.logspace(np.log10(xmin),np.log10(xmax),num=grid_size,dtype=t_float)
