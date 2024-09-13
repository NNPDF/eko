"""Mocking tools."""

import numpy as np


def eko_identity(shape):
    """Generate an identity operator with the given shape.

    The operator has to be squared in PIDs and x grid, thus the last two
    elements of `shape` are not read, and they are considered to be equal to the
    second and third element respectively.

    Parameters
    ----------
    shape: 5-tuple
        specifies the required output tuple (last two elements are ignored)

    Returns
    -------
    np.array
        rank 5 array, representing an identity operator of the requested shape
    """
    i, k = np.ogrid[: shape[1], : shape[2]]
    eko_identity = np.zeros(shape[1:], int)
    eko_identity[i, k, i, k] = 1
    return np.broadcast_to(eko_identity[np.newaxis, :, :, :, :], shape)
