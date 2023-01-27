"""Some utility functions"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def geomspace(start, end, num):
    """
    Numba port of :func:`numpy.geomspace`.

    Parameters
    ----------
        start : float
            initial value
        end : float
            final value
        num : int
            steps

    Returns
    -------
        geomspace : numpy.ndarray
            logarithmic spaced list between `start` and `end`
    """
    return np.exp(np.linspace(np.log(start), np.log(end), num))
