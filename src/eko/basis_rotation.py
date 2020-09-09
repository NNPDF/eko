# -*- coding: utf-8 -*-
"""
This module contains the definitions of the different basis used, i.e.:

- flavor basis::

    22  -6  -5  -4  -3  -2  -1   21  1   2   3   4   5   6
    gm  tb  bb  cb  sb  ub  db   g   d   u   s   c   b   t

- QCD Evolution basis::

    0   1   2   3   4   5   6   7   8   9  10  11  12  13
    gm  S   g   V   V3  V8 V15 V24 V35  T3 T8  T15 T24 T35

"""

import numpy as np

flavor_basis_pids = [22] + list(range(-6, -1 + 1)) + [21] + list(range(1, 6 + 1))

evol_basis = [
    "ph",
    "S",
    "g",
    "V",
    "V3",
    "V8",
    "V15",
    "V24",
    "V35",
    "T3",
    "T8",
    "T15",
    "T24",
    "T35",
]

# Tranformation from physical basis to QCD evolution basis
rotate_flavor_to_evolution = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, -0, -1, 1, 0, -1, 1, 0, -0, 0, 0],
        [0, 0, 0, 0, 2, -1, -1, 0, 1, 1, -2, -0, 0, 0],
        [0, 0, 0, 3, -1, -1, -1, 0, 1, 1, 1, -3, 0, 0],
        [0, 0, 4, -1, -1, -1, -1, 0, 1, 1, 1, 1, -4, 0],
        [0, 5, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, -5],
        [0, 0, 0, -0, 0, 1, -1, 0, -1, 1, 0, -0, 0, 0],
        [0, 0, 0, -0, -2, 1, 1, 0, 1, 1, -2, -0, 0, 0],
        [0, 0, 0, -3, 1, 1, 1, 0, 1, 1, 1, -3, 0, 0],
        [0, 0, -4, 1, 1, 1, 1, 0, 1, 1, 1, 1, -4, 0],
        [0, -5, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, -5],
    ]
)

# inverse transformation
rotate_evolution_to_flavor = np.linalg.inv(rotate_flavor_to_evolution)


def generate_input_from_lhapdf(lhapdf, Q2init):
    """
    Rotate lhapdf-like input object from flavor space to evolution space

    Parameters
    ----------
        lhapdf : object
            lhapdf-like input object (i.e. an object with xfxQ2)
        Q2init : float
            input scale

    Returns
    -------
        input_dict : dict
            a mapping evolution element to callable
    """
    input_dict = {}
    for evol in evol_basis:

        def f(x, evol=evol):
            evol_map = rotate_flavor_to_evolution[evol_basis.index(evol)]
            flavor_list = np.array(
                [lhapdf.xfxQ2(pid, x, Q2init) / x for pid in flavor_basis_pids]
            )
            return evol_map @ flavor_list

        input_dict[evol] = f
    return input_dict
