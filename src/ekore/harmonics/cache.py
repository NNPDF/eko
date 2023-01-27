"""Caching (complicated) harmonic sums across :mod:`ekore`."""
import numba as nb
import numpy as np
import numpy.typing as npt

from . import w1, w2

# here a register of all possible functions
S1 = 0  # = S_1(N)
S2 = 1
# this could also be S1h = S1(N/2)
# the only requirement is that they are insubsequent order
# and reset knows the maximum size (to fit them all)

# this is a plain list holding the values
@nb.njit(cache=True)
def reset():
    """Return the cache placeholder array."""
    return np.full(2, np.nan, np.complex_)


@nb.njit(cache=True)
def get(key: int, cache: npt.ArrayLike, n: complex) -> complex:
    """Retrieve an element of the cache.

    Parameters
    ----------
    key :
        harmonic sum key
    cache :
        cache list holding all elements
    n :
        complex evaluation point

    Returns
    -------
    complex :
        requested harmonic sum evaluated at n

    """
    # Maybe improve error
    if key < 0 or key > len(cache):
        raise RuntimeError
    # load the thing
    s = cache[key]
    # found? i.e. not NaN?
    if not np.isnan(s):
        return s
    # compute it now ...
    if key == S1:
        s = w1.S1(n)
    elif key == S2:
        s = w2.S2(n)
    # store and return
    cache[key] = s
    return s
