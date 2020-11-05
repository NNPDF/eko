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

import copy

import numpy as np

flavor_basis_pids = tuple([22] + list(range(-6, -1 + 1)) + [21] + list(range(1, 6 + 1)))

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

# inverse transformation
rotate_evolution_to_flavor = np.linalg.inv(rotate_flavor_to_evolution)


def rotate_pm_to_flavor(label):
    """
    Rotate from +- basis to flavor basis.

    Parameters
    ----------
        label : str
            label

    Returns
    -------
        l : list(float)
            list of weights
    """
    # g and ph are unaltered
    if label in ["g", "ph"]:
        return rotate_flavor_to_evolution[evol_basis.index(label)].copy()
    # no it has to be a quark with + or - appended
    if label[0] not in "duscbt" or label[1] not in ["+", "-"]:
        raise ValueError(f"Invalid pm label: {label}")
    l = np.zeros(len(flavor_basis_pids))
    idx = flavor_basis_names.index(label[0])
    pid = flavor_basis_pids[idx]
    l[idx] = 1
    # + is +, - is -
    if label[1] == "+":
        l[flavor_basis_pids.index(-pid)] = 1
    else:
        l[flavor_basis_pids.index(-pid)] = -1
    return l


def generate_input_from_lhapdf(lhapdf, xs, Q2init):
    """
    Rotate lhapdf-like input object from flavor space to evolution space

    Parameters
    ----------
        lhapdf : object
            lhapdf-like input object (i.e. an object with xfxQ2)
        xs : list
            input x values
        Q2init : float
            input scale

    Returns
    -------
        input_dict : dict
            a mapping evolution element to callable
    """
    # query all pdfs
    flavor_list = []
    empty_pids = []
    for pid in flavor_basis_pids:
        if lhapdf.hasFlavor(pid):
            ls = [lhapdf.xfxQ2(pid, x, Q2init) / x for x in xs]
        else:
            ls = np.zeros(len(xs))
            empty_pids.append(pid)
        flavor_list.append(ls)
    # iterate all pdfs
    input_dict = {}
    for j, evol in enumerate(evol_basis):
        # is it a quark pdf and is it trivial? then skip
        if len(evol) > 1 and evol[0] in ["V", "T"]:
            q = int(np.sqrt(int(evol[1:]) + 1))
            if q in empty_pids and -q in empty_pids:
                continue
        # rotate
        evol_map = rotate_flavor_to_evolution[j].copy()
        input_dict[evol] = evol_map @ flavor_list
    return input_dict


def fill_trivial_dists(old_evols):
    """
    Insert trivial evolutions basis elements.

    Parameters
    ----------
        old_evols : dict
            :class:`eko.evolution_operator.PhysicalOperator`-like dictionary

    Returns
    -------
        evols : dict
            updated dictionary
        trivial_dists : list
            inserted elements
    """
    evols = copy.deepcopy(old_evols)
    # check dependencies beforehand
    if "S" not in evols:
        raise KeyError("No S distribution available")
    if "V" not in evols:
        raise KeyError("No V distribution available")
    trivial_dists = []
    for evol in evol_basis:
        # only rotate in quark distributions
        if evol == "g":
            continue
        # are the target distributions there?
        if evol in ["S", "V"]:
            continue
        # PDF is set?
        if evol in evols:
            continue
        # insert empty photon
        if evol == "ph":
            evols[evol] = np.zeros(len(evols["S"]))
        # insert trivial value
        elif evol[0] == "V":
            evols[evol] = evols["V"]
        elif evol[0] == "T":
            evols[evol] = evols["S"]
        # register for later use
        trivial_dists.append(evol)
    return evols, trivial_dists


def rotate_output(in_evols):
    """
    Rotate lists in evolution basis back to flavor basis.

    Parameters
    ----------
        in_evols : dict
            :class:`eko.evolution_operator.PhysicalOperator`-like dictionary

    Returns
    -------
        out : dict
            rotated dictionary
    """
    # prepare
    evols, trivial_dists = fill_trivial_dists(in_evols)
    evol_list = np.array([evols[evol] for evol in evol_basis])
    final_quark_pid = 6
    final_valence_pid = 6
    for q in range(6, 3, -1):
        if f"T{q*q-1}" in trivial_dists:
            final_quark_pid -= 1
            # assume Vxx too vanish
            final_valence_pid -= 1
            continue
        if f"V{q*q-1}" in trivial_dists:
            final_valence_pid -= 1
    # rotate
    out = {}
    for j, pid in enumerate(flavor_basis_pids):
        # do we actually have to do smth?
        is_non_trivial_ph = pid == 22 and "ph" not in trivial_dists
        is_non_trivial_q = -final_valence_pid <= pid <= final_quark_pid
        if is_non_trivial_ph or is_non_trivial_q or pid == 21:
            flavor_map = rotate_evolution_to_flavor[j]
            out[pid] = flavor_map @ evol_list
    # recover the bared distributions, e.g. tbar = t
    for q in range(-final_quark_pid, -final_valence_pid):
        out[q] = out[-q]
    return out
