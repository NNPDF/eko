"""Caching (complicated) harmonic sums across :mod:`ekore`."""
import numba as nb
import numpy as np
import numpy.typing as npt

from . import w1, w2, w3, w4, w5

# here a register of all possible functions
S1 = 0  # = S_1(N)
S2 = 1
S3 = 2
S4 = 3
S5 = 4
Sm1 = 5
Sm2 = 6
Sm3 = 7
Sm4 = 8
Sm5 = 9
S21 = 10
S2m1 = 11
Sm21 = 12
Sm2m1 = 13
S31 = 14
Sm31 = 15
Sm22 = 16
S211 = 17
Sm211 = 18

# this could also be S1h = S1(N/2)
# the only requirement is that they are insubsequent order
# and reset knows the maximum size (to fit them all)

# this is a plain list holding the values
@nb.njit(cache=True)
def reset():
    """Return the cache placeholder array."""
    return np.full(19, np.nan, np.complex_)


@nb.njit(cache=True)
def get(key: int, cache: npt.ArrayLike, 
        n: complex, is_singlet: bool) -> complex:
    """Retrieve an element of the cache.

    Parameters
    ----------
    key :
        harmonic sum key
    cache :
        cache list holding all elements
    n :
        complex evaluation point
    is_singlet :
        symmetry factor: True for singlet like quantities 
        (:math:`\eta=(-1)^N = 1`),
        False for non-singlet like quantities 
        (:math:`\eta=(-1)^N=-1`)

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
    elif key == S3:
        s = w3.S3(n)
    elif key == S4:
        s = w4.S4(n)
    elif key == S5:
        s = w5.S5(n)
    elif key == Sm1:
        s = w1.Sm1(n, get(S1, cache, n), is_singlet)
    elif key == Sm2:
        s = w2.Sm2(n, get(S2, cache, n), is_singlet)
    elif key == Sm3:
        s = w3.Sm3(n, get(S3, cache, n), is_singlet)
    elif key == Sm4:
        s = w4.Sm4(n, get(S4, cache, n), is_singlet)
    elif key == Sm5:
        s = w5.Sm5(n, get(S5, cache, n), is_singlet)
    elif key == S21:
        s = w3.S21(n, get(S1, cache, n), get(S2, cache, n))
    elif key == S2m1:
        s = w3.S2m1(n, get(S2, cache, n), get(Sm1, cache, n), 
            get(Sm2, cache, n), is_singlet)
    elif key == Sm21:
        s = w3.Sm21(n, get(S1, cache, n), get(Sm1, cache, n), 
            is_singlet)
    elif key == Sm2m1:
        s = w3.Sm2m1(n, get(S1, cache, n), get(S2, cache, n), 
            get(Sm2, cache, n))
    elif key == S31:
        s = w4.S31(n, get(S1, cache, n), get(S2, cache, n), 
            get(S3, cache, n), get(S4, cache, n))
    elif key == Sm31:
        s = w4.Sm31(n, get(S1, cache, n), get(Sm1, cache, n), 
            get(Sm2, cache, n), is_singlet)
    elif key == Sm22:
        s = w4.Sm22(n, get(S1, cache, n), get(S2, cache, n), 
            get(Sm2, cache, n), get(Sm31, cache, n), is_singlet)
    elif key == S211:
        s = w4.S211(n, get(S1, cache, n), get(S2, cache, n), 
            get(S3, cache, n))
    elif key == Sm211:
        s = Sm211(n, get(S1, cache, n), get(S2, cache, n), 
            get(Sm1, cache, n), is_singlet)
    # store and return
    cache[key] = s
    return s