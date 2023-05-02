"""Caching harmonic sums across :mod:`ekore`."""
import numba as nb
import numpy as np
import numpy.typing as npt

from . import w1, w2, w3, w4, w5
from .g_functions import mellin_g3
from .polygamma import recursive_harmonic_sum

# here a register of all possible functions
CACHE_SIZE = 37
_index = iter(range(CACHE_SIZE))
S1 = next(_index)  # = S_1(N)
S2 = next(_index)
S3 = next(_index)
S4 = next(_index)
S5 = next(_index)
Sm1 = next(_index)
Sm2 = next(_index)
Sm3 = next(_index)
Sm4 = next(_index)
Sm5 = next(_index)
S21 = next(_index)
S2m1 = next(_index)
Sm21 = next(_index)
Sm2m1 = next(_index)
S31 = next(_index)
Sm31 = next(_index)
Sm22 = next(_index)
S211 = next(_index)
Sm211 = next(_index)
S1h = next(_index)
S2h = next(_index)
S3h = next(_index)
S4h = next(_index)
S5h = next(_index)
S1mh = next(_index)
S2mh = next(_index)
S3mh = next(_index)
S4mh = next(_index)
S5mh = next(_index)
S1ph = next(_index)
S2ph = next(_index)
S3ph = next(_index)
S4ph = next(_index)
S5ph = next(_index)
g3 = next(_index)
S1p2 = next(_index)
g3p2 = next(_index)


@nb.njit(cache=True)
def reset():
    """Return the cache placeholder array."""
    return np.full(CACHE_SIZE, np.nan, np.complex_)


@nb.njit(cache=True)
def get(key: int, cache: npt.ArrayLike, n: complex, is_singlet=None) -> complex:
    r"""Retrieve an element of the cache.

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
        s = w1.Sm1(
            n,
            get(S1, cache, n),
            get(S1mh, cache, n),
            get(S1h, cache, n),
            is_singlet,
        )
    elif key == Sm2:
        s = w2.Sm2(
            n,
            get(S2, cache, n),
            get(S2mh, cache, n),
            get(S2h, cache, n),
            is_singlet,
        )
    elif key == Sm3:
        s = w3.Sm3(
            n,
            get(S3, cache, n),
            get(S3mh, cache, n),
            get(S3h, cache, n),
            is_singlet,
        )
    elif key == Sm4:
        s = w4.Sm4(
            n,
            get(S4, cache, n),
            get(S4mh, cache, n),
            get(S4h, cache, n),
            is_singlet,
        )
    elif key == Sm5:
        s = w5.Sm5(
            n,
            get(S5, cache, n),
            get(S5mh, cache, n),
            get(S5h, cache, n),
            is_singlet,
        )
    elif key == S21:
        s = w3.S21(n, get(S1, cache, n), get(S2, cache, n))
    elif key == S2m1:
        s = w3.S2m1(
            n,
            get(S2, cache, n),
            get(Sm1, cache, n, is_singlet),
            get(Sm2, cache, n, is_singlet),
            is_singlet,
        )
    elif key == Sm21:
        s = w3.Sm21(n, get(S1, cache, n), get(Sm1, cache, n, is_singlet), is_singlet)
    elif key == Sm2m1:
        s = w3.Sm2m1(
            n,
            get(S1, cache, n),
            get(S2, cache, n),
            get(Sm2, cache, n, is_singlet),
        )
    elif key == S31:
        s = w4.S31(
            n,
            get(S1, cache, n),
            get(S2, cache, n),
            get(S3, cache, n),
            get(S4, cache, n),
        )
    elif key == Sm31:
        s = w4.Sm31(
            n,
            get(S1, cache, n),
            get(Sm1, cache, n, is_singlet),
            get(Sm2, cache, n, is_singlet),
            is_singlet,
        )
    elif key == Sm22:
        s = w4.Sm22(
            n,
            get(S1, cache, n),
            get(S2, cache, n),
            get(Sm2, cache, n, is_singlet),
            get(Sm31, cache, n, is_singlet),
            is_singlet,
        )
    elif key == S211:
        s = w4.S211(
            n,
            get(S1, cache, n),
            get(S2, cache, n),
            get(S3, cache, n),
        )
    elif key == Sm211:
        s = w4.Sm211(
            n,
            get(S1, cache, n),
            get(S2, cache, n),
            get(Sm1, cache, n, is_singlet),
            is_singlet,
        )
    elif key == S1h:
        s = w1.S1(n / 2)
    elif key == S2h:
        s = w2.S2(n / 2)
    elif key == S3h:
        s = w3.S3(n / 2)
    elif key == S4h:
        s = w4.S4(n / 2)
    elif key == S5h:
        s = w5.S5(n / 2)
    elif key == S1mh:
        s = w1.S1((n - 1) / 2)
    elif key == S2mh:
        s = w2.S2((n - 1) / 2)
    elif key == S3mh:
        s = w3.S3((n - 1) / 2)
    elif key == S4mh:
        s = w4.S4((n - 1) / 2)
    elif key == S5mh:
        s = w5.S5((n - 1) / 2)
    elif key == S1ph:
        s = recursive_harmonic_sum(get(S1mh, cache, n), (n - 1) / 2, 1, 1)
    elif key == S2ph:
        s = recursive_harmonic_sum(get(S2mh, cache, n), (n - 1) / 2, 1, 2)
    elif key == S3ph:
        s = recursive_harmonic_sum(get(S3mh, cache, n), (n - 1) / 2, 1, 3)
    elif key == S4ph:
        s = recursive_harmonic_sum(get(S4mh, cache, n), (n - 1) / 2, 1, 4)
    elif key == S5ph:
        s = recursive_harmonic_sum(get(S5mh, cache, n), (n - 1) / 2, 1, 5)
    elif key == g3:
        s = mellin_g3(n, get(S1, cache, n))
    elif key == S1p2:
        s = recursive_harmonic_sum(get(S1, cache, n), n, 2, 1)
    elif key == g3p2:
        s = mellin_g3(n + 2, get(S1p2, cache, n))
    # store and return
    cache[key] = s
    return s
