"""Caching harmonic sums across :mod:`ekore`."""

from typing import Optional

import numba as nb
import numpy as np
import numpy.typing as npt

from . import w1, w2, w3, w4, w5
from .g_functions import mellin_g3
from .polygamma import recursive_harmonic_sum

# here a register of all possible functions
CACHE_SIZE = 31
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
S1mh = next(_index)
S2mh = next(_index)
S3mh = next(_index)
S1ph = next(_index)
S2ph = next(_index)
S3ph = next(_index)
g3 = next(_index)
S1p2 = next(_index)
g3p2 = next(_index)


@nb.njit(cache=True)
def reset():
    """Return the cache placeholder array."""
    return np.full(CACHE_SIZE, np.nan, np.complex128)


@nb.njit(cache=True)
def update(func, key, cache, n):
    """Compute simple harmonics if not yet in cache."""
    if np.isnan(cache[key]):
        cache[key] = func(n)
    return cache


@nb.njit(cache=True)
def update_Sm1(cache, n, is_singlet):
    """Compute Sm1 if not yet in cache."""
    if np.isnan(cache[Sm1]):
        cache = update(w1.S1, S1, cache, n)
        cache = update(w1.S1, S1mh, cache, (n - 1) / 2)
        cache = update(w1.S1, S1h, cache, n / 2)
        cache[Sm1] = w1.Sm1(n, cache[S1], cache[S1mh], cache[S1h], is_singlet)
    return cache


@nb.njit(cache=True)
def update_Sm2(cache, n, is_singlet):
    """Compute Sm2 if not yet in cache."""
    if np.isnan(cache[Sm2]):
        cache = update(w2.S2, S2, cache, n)
        cache = update(w2.S2, S2mh, cache, (n - 1) / 2)
        cache = update(w2.S2, S2h, cache, n / 2)
        cache[Sm2] = w2.Sm2(n, cache[S2], cache[S2mh], cache[S2h], is_singlet)
    return cache


@nb.njit(cache=True)
def get(
    key: int, cache: npt.ArrayLike, n: complex, is_singlet: Optional[bool] = None
) -> complex:
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
    if key < 0 or key >= len(cache):
        raise RuntimeError
    # load the thing
    s = cache[key]
    # found? i.e. not NaN?
    if not np.isnan(s):
        return s
    # compute it now ...
    # weight 1
    if key == S1:
        s = w1.S1(n)
    elif key == S1h:
        s = w1.S1(n / 2)
    elif key == S1mh:
        s = w1.S1((n - 1) / 2)
    elif key == S1ph:
        cache = update(w1.S1, S1mh, cache, (n - 1) / 2)
        s = recursive_harmonic_sum(cache[S1mh], (n - 1) / 2, 1, 1)
    elif key == Sm1:
        cache = update_Sm1(cache, n, is_singlet)
        s = cache[key]
    elif key == S1p2:
        cache = update(w1.S1, S1, cache, n)
        s = recursive_harmonic_sum(cache[S1], n, 2, 1)
    # weight 2
    elif key == S2:
        s = w2.S2(n)
    elif key == S2h:
        s = w2.S2(n / 2)
    elif key == S2mh:
        s = w2.S2((n - 1) / 2)
    elif key == S2ph:
        cache = update(w2.S2, S2mh, cache, (n - 1) / 2)
        s = recursive_harmonic_sum(cache[S2mh], (n - 1) / 2, 1, 2)
    elif key == Sm2:
        cache = update_Sm2(cache, n, is_singlet)
        s = cache[key]
    # weight 3
    elif key == S3:
        s = w3.S3(n)
    elif key == S3h:
        s = w3.S3(n / 2)
    elif key == S3mh:
        s = w3.S3((n - 1) / 2)
    elif key == S3ph:
        cache = update(w3.S3, S3mh, cache, (n - 1) / 2)
        s = recursive_harmonic_sum(cache[S3mh], (n - 1) / 2, 1, 3)
    elif key == Sm3:
        cache = update(w3.S3, S3, cache, n)
        cache = update(w3.S3, S3mh, cache, (n - 1) / 2)
        cache = update(w3.S3, S3h, cache, n / 2)
        s = w3.Sm3(n, cache[S3], cache[S3mh], cache[S3h], is_singlet)
    # weight 4
    elif key == S4:
        s = w4.S4(n)
    elif key == Sm4:
        cache = update(w4.S4, S4, cache, n)
        S4mh = w4.S4((n - 1) / 2)
        S4h = w4.S4(n / 2)
        s = w4.Sm4(n, cache[S4], S4mh, S4h, is_singlet)
    # weight 5
    elif key == S5:
        s = w5.S5(n)
    elif key == Sm5:
        cache = update(w5.S5, S5, cache, n)
        S5mh = w5.S5((n - 1) / 2)
        S5h = w5.S5(n / 2)
        s = w5.Sm5(n, cache[S5], S5mh, S5h, is_singlet)
    # mellin g3 and related
    elif key == g3:
        cache = update(w1.S1, S1, cache, n)
        s = mellin_g3(n, cache[S1])
    elif key == g3p2:
        cache = update(w1.S1, S1p2, cache, n + 2)
        s = mellin_g3(n + 2, cache[S1p2])
    else:
        # Multi index harmonics which do not require Sxm
        cache = update(w1.S1, S1, cache, n)
        cache = update(w2.S2, S2, cache, n)
        if key == S21:
            s = w3.S21(n, cache[S1], cache[S2])
        elif key == S31:
            cache = update(w3.S3, S3, cache, n)
            cache = update(w4.S4, S4, cache, n)
            s = w4.S31(n, cache[S1], cache[S2], cache[S3], cache[S4])
        elif key == S211:
            cache = update(w3.S3, S3, cache, n)
            s = w4.S211(n, cache[S1], cache[S2], cache[S3])
        else:
            # Multi index harmonics which require Sm1
            cache = update_Sm1(cache, n, is_singlet)
            if key == Sm21:
                s = w3.Sm21(n, cache[S1], cache[Sm1], is_singlet)
            elif key == Sm211:
                s = w4.Sm211(n, cache[S1], cache[S2], cache[Sm1], is_singlet)
            else:
                #  Multi index harmonics which require also Sm2
                cache = update_Sm2(cache, n, is_singlet)
                if key == S2m1:
                    s = w3.S2m1(n, cache[S2], cache[Sm1], cache[Sm2], is_singlet)
                elif key == Sm2m1:
                    s = w3.Sm2m1(n, cache[S1], cache[S2], cache[Sm2])
                elif key == Sm31:
                    s = w4.Sm31(n, cache[S1], cache[Sm1], cache[Sm2], is_singlet)
                elif key == Sm22:
                    if np.isnan(cache[Sm31]):
                        cache[Sm31] = w4.Sm31(
                            n, cache[S1], cache[Sm1], cache[Sm2], is_singlet
                        )
                    s = w4.Sm22(
                        n, cache[S1], cache[S2], cache[Sm2], cache[Sm31], is_singlet
                    )
    # store and return
    cache[key] = s
    return s
