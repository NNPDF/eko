# -*- coding: utf-8 -*-
r"""
The write-up of the matching conditions is given in
:doc:`Matching Conditions </theory/Matching>`.

"""

import numpy as np

from .. import basis_rotation as br

quark_names = "duscbt"


def pids_from_intrinsic_evol(label, nlf, normalize):
    r"""
    Obtain the list of pids with their corresponding weight, that are contributing to ``evol``

    The normalization of the weights is only needed for the output rotation:

    - if we want to build e.g. the singlet in the initial state we simply have to sum
      to obtain :math:`S = u + \bar u + d + \bar d + \ldots`
    - if we want to rotate back in the output we have to *normalize* the weights:
      e.g. in nf=3 :math:`u = \frac 1 6 S + \frac 1 6 V + \ldots`

    The normalization can only happen here since we're actively cutting out some
    flavor (according to ``nlf``).

    Parameters
    ----------
        evol : str
            evolution label
        nlf : int
            maximum number of light flavors
        normalize : bool
            normalize output

    Returns
    -------
        m : list
    """
    try:
        evol_idx = br.evol_basis.index(label)
        is_evol = True
    except ValueError:
        is_evol = False
    if is_evol:
        weights = br.rotate_flavor_to_evolution[evol_idx].copy()
        for j, pid in enumerate(br.flavor_basis_pids):
            if nlf < abs(pid) <= 6:
                weights[j] = 0
    else:
        weights = rotate_pm_to_flavor(label)
    # normalize?
    if normalize:
        norm = weights @ weights
        weights = weights / norm
    return weights


def get_range(evol_labels):
    """
    Determine the number of light and heavy flavors participating in the input and output.

    Here, we assume that the T distributions (e.g. T15) appears *always*
    before the corresponding V distribution (e.g. V15).

    Returns
    -------
        nf_in : int
            number of light flavors in the input
        nf_out : int
            number of light flavors in the output
    """
    nf_in = 3
    nf_out = 3

    def update(label):
        nf = 3
        if label[0] == "T":
            nf = round(np.sqrt(int(label[1:]) + 1))
        return nf

    for op in evol_labels:
        nf_in = max(update(op.input), nf_in)
        nf_out = max(update(op.target), nf_out)

    return nf_in, nf_out


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
        return br.rotate_flavor_to_evolution[br.evol_basis.index(label)].copy()
    # no it has to be a quark with + or - appended
    if label[0] not in quark_names or label[1] not in ["+", "-"]:
        raise ValueError(f"Invalid pm label: {label}")
    l = np.zeros(len(br.flavor_basis_pids))
    idx = br.flavor_basis_names.index(label[0])
    pid = br.flavor_basis_pids[idx]
    l[idx] = 1
    # + is +, - is -
    if label[1] == "+":
        l[br.flavor_basis_pids.index(-pid)] = 1
    else:
        l[br.flavor_basis_pids.index(-pid)] = -1
    return l


def rotate_matching(nf, inverse=False):
    """
    Rotation between matching basis (with e.g. S,g,...V8 and c+,c-) and new true evolution basis
    (with S,g,...V8,T15,V15).

    Parameters
    ----------
        nf : int
            number of active flavors in the higher patch: to activate T15, nf=4
        inverse : bool
            use inverse conditions?

    Returns
    -------
        l : dict
            mapping in dot notation between the bases
    """
    # the gluon and the photon do not care about new quarks
    l = {"g.g": 1.0, "ph.ph": 1.0}
    # already active distributions
    for k in range(2, nf):  # nf is the upper, so excluded
        n = k ** 2 - 1
        l[f"V{n}.V{n}"] = 1.0
        l[f"T{n}.T{n}"] = 1.0
    # the new contributions
    n = nf ** 2 - 1  # nf is pointing upwards
    q = quark_names[nf - 1]
    for (tot, oth, qpm) in (("S", f"T{n}", f"{q}+"), ("V", f"V{n}", f"{q}-")):
        if inverse:
            l[f"{tot}.{tot}"] = (nf - 1.0) / nf
            l[f"{tot}.{oth}"] = 1.0 / nf
            l[f"{qpm}.{tot}"] = 1.0 / nf
            l[f"{qpm}.{oth}"] = -1.0 / nf
        else:
            l[f"{tot}.{tot}"] = 1.0
            l[f"{tot}.{qpm}"] = 1.0
            l[f"{oth}.{tot}"] = 1.0
            l[f"{oth}.{qpm}"] = -(nf - 1.0)
    # also higher quarks do not care
    for k in range(nf + 1, 6 + 1):
        q = quark_names[k - 1]
        for sgn in "+-":
            l[f"{q}{sgn}.{q}{sgn}"] = 1.0
    return l


def rotate_matching_inverse(nf):
    return rotate_matching(nf, True)
