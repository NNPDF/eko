# -*- coding: utf-8 -*-
"""
This module contains the definitions of the
:doc:`Flavor Basis and Evolution Basis </theory/FlavorSpace>`.
"""

import numpy as np

flavor_basis_pids = tuple([22] + list(range(-6, -1 + 1)) + [21] + list(range(1, 6 + 1)))
r"""
Sorted elements in Flavor Basis as |pid|.

Definition: `here <https://pdg.lbl.gov/2019/reviews/rpp2019-rev-monte-carlo-numbering.pdf>`_

corresponding |PDF| : :math:`\gamma, \bar t, \bar b, \bar c, \bar s, \bar u, \bar d, g,
d, u, s, c, b, t`
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
r"""
Sorted elements in Evolution Basis as :obj:`str`.

Definition: :ref:`here <theory/FlavorSpace:flavor basis>`.

corresponding |PDF| : :math:`\gamma, \Sigma, g, V, V_{3}, V_{8}, V_{15}, V_{24},
V_{35}, T_{3}, T_{8}, T_{15}, T_{24}, T_{35}`
"""

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
        [0, 0, 0, 0, 0, 1, -1, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -2, 1, 1, 0, 1, 1, -2, 0, 0, 0],
        [0, 0, 0, -3, 1, 1, 1, 0, 1, 1, 1, -3, 0, 0],
        [0, 0, -4, 1, 1, 1, 1, 0, 1, 1, 1, 1, -4, 0],
        [0, -5, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, -5],
    ]
)
"""
Basis rotation matrix between :doc:`Flavor Basis and Evolution Basis </theory/FlavorSpace>`.
"""
