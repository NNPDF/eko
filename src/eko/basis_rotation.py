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

singlet_labels = ("S_qq", "S_qg", "S_gq", "S_gg")
non_singlet_labels = ("NS_m", "NS_p", "NS_v")
full_labels = (*singlet_labels, *non_singlet_labels)
anomalous_dimensions_basis = full_labels
r"""
Sorted elements in Anomalous Dimensions Basis as :obj:`str`.
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

map_ad_to_evolution = {
    "S_qq": ["S.S"],
    "S_qg": ["S.g"],
    "S_gq": ["g.S"],
    "S_gg": ["g.g"],
    "NS_v": ["V.V"],
    "NS_p": ["T3.T3", "T8.T8", "T15.T15", "T24.T24", "T35.T35"],
    "NS_m": ["V3.V3", "V8.V8", "V15.V15", "V24.V24", "V35.V35"],
}
"""
Map anomalous dimension sectors' names to their members
"""


def ad_projector(ad_lab, nf):
    """
    Build a projector (as a numpy array) for the given anomalous dimension
    sector.

    Parameters
    ----------
    ad_lab : str
        name of anomalous dimension sector
    nf : int
        number of light flavors

    Returns
    -------
    proj : np.ndarray
        projector over the specified sector
    """
    proj = np.zeros_like(rotate_flavor_to_evolution, dtype=float)
    l = map_ad_to_evolution[ad_lab]
    # restrict the evolution basis to light flavors
    # NOTE: the cut is only needed for "NS_p" and "NS_m", but the other lists
    # are 1-long so they are unaffected
    l = l[: (nf - 1)]

    for el in l:
        out_name, in_name = el.split(".")
        out_idx = evol_basis.index(out_name)
        in_idx = evol_basis.index(in_name)
        out = rotate_flavor_to_evolution[out_idx].copy()
        in_ = rotate_flavor_to_evolution[in_idx].copy()
        out[: 1 + (6 - nf)] = out[len(out) - (6 - nf) :] = 0
        in_[: 1 + (6 - nf)] = in_[len(in_) - (6 - nf) :] = 0
        proj += (out[:, np.newaxis] * in_) / (out @ out)

    return proj


def ad_projectors(nf):
    """
    Build projectors tensor (as a numpy array), collecting all the individual
    sector projectors.

    Parameters
    ----------
    nf : int
        number of light flavors

    Returns
    -------
    projs : np.ndarray
        projectors tensor
    """
    projs = []
    for ad in anomalous_dimensions_basis:
        projs.append(ad_projector(ad, nf))

    return np.array(projs)
