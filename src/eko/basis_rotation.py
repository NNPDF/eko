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


quark_names = "".join(flavor_basis_names[-6:])

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

evol_basis_pids = tuple(
    [22, 100, 21, 200]
    + [200 + n**2 - 1 for n in range(2, 6 + 1)]
    + [100 + n**2 - 1 for n in range(2, 6 + 1)]
)
"""|pid| representation of :data:`evol_basis`."""

non_singlet_pids_map = {
    "ns-": 10201,
    "ns+": 10101,
    "nsV": 10200,
    "ns-u": 10202,
    "ns-d": 10203,
    "ns+u": 10102,
    "ns+d": 10103,
}

singlet_labels = ((100, 100), (100, 21), (21, 100), (21, 21))
non_singlet_labels = (
    (non_singlet_pids_map["ns-"], 0),
    (non_singlet_pids_map["ns+"], 0),
    (non_singlet_pids_map["nsV"], 0),
)
# Sdelta = 101
singlet_unified_labels = (
    (21, 21),
    (21, 20),
    (21, 100),
    (21, 101),
    (20, 21),
    (20, 20),
    (20, 100),
    (20, 101),
    (100, 21),
    (100, 20),
    (100, 100),
    (100, 101),
    (101, 21),
    (101, 20),
    (101, 100),
    (101, 101),
)
# Vdelta = 10204
valence_unified_labels = (
    (10200, 10200),
    (10200, 10204),
    (10204, 10200),
    (10204, 10204),
)
non_singlet_unified_labels = (
    (non_singlet_pids_map["ns-u"], 0),
    (non_singlet_pids_map["ns-d"], 0),
    (non_singlet_pids_map["ns+u"], 0),
    (non_singlet_pids_map["ns+d"], 0),
)
full_labels = (*singlet_labels, *non_singlet_labels)
anomalous_dimensions_basis = full_labels
r"""
Sorted elements in Anomalous Dimensions Basis as :obj:`str`.
"""
matching_hplus_pid = 90
matching_hminus_pid = 91

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
    (100, 100): ["S.S"],
    (100, 21): ["S.g"],
    (21, 100): ["g.S"],
    (21, 21): ["g.g"],
    (non_singlet_pids_map["nsV"], 0): ["V.V"],
    (non_singlet_pids_map["ns+"], 0): [
        "T3.T3",
        "T8.T8",
        "T15.T15",
        "T24.T24",
        "T35.T35",
    ],
    (non_singlet_pids_map["ns-"], 0): [
        "V3.V3",
        "V8.V8",
        "V15.V15",
        "V24.V24",
        "V35.V35",
    ],
}
"""
Map anomalous dimension sectors' names to their members
"""

map_ad_to_intrinsic_evolution = {
    (21, 21): ["g.g"],
    (21, 20): ["g.ph"],
    (21, 100): ["g.S"],
    (21, 101): ["g.Sdelta"],
    (20, 21): ["ph.g"],
    (20, 20): ["ph.ph"],
    (20, 100): ["ph.S"],
    (20, 101): ["ph.Sdelta"],
    (100, 21): ["S.g"],
    (100, 20): ["S.ph"],
    (100, 100): ["S.S"],
    (100, 101): ["S.Sdelta"],
    (101, 21): ["Sdelta.Sdelta"],
    (101, 20): ["Sdelta.ph"],
    (101, 100): ["Sdelta.S"],
    (101, 101): ["Sdelta.Sdelta"],
    (10200, 10200): ["V.V"],
    (10200, 10204): ["V.Vdelta"],
    (10204, 10200): ["Vdelta.V"],
    (10204, 10204): ["Vdelta.Vdelta"],
    (non_singlet_pids_map["ns+u"], 0): [
        "Tu3.Tu3",
        "Tu8.Tu8",
    ],
    (non_singlet_pids_map["ns+d"], 0): [
        "Td3.Td3",
        "Td8.Td8",
    ],
    (non_singlet_pids_map["ns-u"], 0): [
        "Vu3.Vu3",
        "Vu8.Vu8",
    ],
    (non_singlet_pids_map["ns-d"], 0): [
        "Vd3.Vd3",
        "Vd8.Vd8",
    ],
}


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


def intrinsic_unified_evol_labels(nf):
    """
    Collect all labels in the intrinsic unified evolution basis.

    Parameters
    ----------
    nf : int
        number of light flavors

    Returns
    -------
    labels : list(str)
        active distributions
    """
    labels = [
        "ph",
        "g",
        "S",
        "V",
        "Sdelta",
        "Vdelta",
        "Td3",
        "Vd3",
    ]
    news = ["u3", "d8", "u8"]
    for j in range(nf, 6):
        q = quark_names[j]
        labels.extend([f"{q}+", f"{q}-"])
    for k in range(3, nf):
        new = news[k - 3]
        labels.extend([f"T{new}", f"V{new}"])

    return labels
