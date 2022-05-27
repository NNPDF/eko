# -*- coding: utf-8 -*-
"""The modules provides a hack to automatically generate separate numba
callables for each perturbative order."""

import inspect


def parse(fnk, max_order, locs):
    """Remove higher perturbative order function calls.

    - the string `__nklo` is replaced with an explicit version (i.e. sufficient ns).
      This can be used to chain the scope to lower laying funciton calls.
    - the code between `#>> start/NLO` and `#<< end/NLO` is cut out for a LO cut
      (and similar for higher orders). This can be used to cut out expensive calls
      needed at higher orders.


    Parameters
    ----------
    fnk : callable
        all order callable
    max_order : int
        allowed perturbative order
    locs : mapping
        local call to `globals()` to pass the variables around
    """
    cnt = inspect.getsource(fnk)
    # remove code
    for order in range(max_order, 3 + 1):
        k = "N" * order + "LO"
        s = cnt.find(f"#>> start/{k}")
        e = cnt.find(f"#<< end/{k}")
        cnt = cnt[:s] + cnt[e:]
    # replace place holder tags with their actual incarnation
    tag = "n" * max_order + "lo"
    new_name = fnk.__name__.replace("__nklo", f"__{tag}")
    cnt = cnt.replace("__nklo", f"__{tag}")
    exec(cnt, locs)
    return locs[new_name]
