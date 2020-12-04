# -*- coding: utf-8 -*-
"""
This module contains the definitions of the
:doc:`Flavor Basis and Evolution Basis </theory/FlavorSpace>`.
"""

import numpy as np

flavor_basis_pids = tuple([22] + list(range(-6, -1 + 1)) + [21] + list(range(1, 6 + 1)))
r"""
Sorted elements in flavor basis as |pid|.

Definition: `here <https://pdg.lbl.gov/2019/reviews/rpp2019-rev-monte-carlo-numbering.pdf>`_.

corresponding |PDF| : :math:`\gamma, \bar t, \bar b, \bar c, \bar s, \bar u, \bar d, g, d, u, s, c, b, t`.
"""

flavor_basis_names = (
    "ph",
    "tbar",
    "bbar",
    "cbar",
    "sbar",
    "ubar",
    "dbar",
    "g",
    "d",
    "u",
    "s",
    "c",
    "b",
    "t",
)
"""String representation of :data:`flavor_basis_pids`."""

evol_basis = (
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
)

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
