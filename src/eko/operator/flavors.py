# -*- coding: utf-8 -*-
r"""
The write-up of the matching conditions is given in
:doc:`Matching Conditions </theory/Matching>`.

"""

import numpy as np

from .. import basis_rotation as br


singlet_labels = ("S_qq", "S_qg", "S_gq", "S_gg")
non_singlet_labels = ("NS_m", "NS_p", "NS_v")
full_labels = (*singlet_labels, *non_singlet_labels)


class MemberName:
    """
    Operator member name

    Parameters
    ----------
        name : str
            operator name
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    def _split_name(self):
        """Splits the name according to target.input"""
        # we need to do this late, as in raw mode the name to not follow this principle
        name_spl = self.name.split(".")
        if len(name_spl) != 2:
            raise ValueError("The operator name has no valid format: target.input")
        for k in [0, 1]:
            name_spl[k] = name_spl[k].strip()
            if len(name_spl[k]) <= 0:
                raise ValueError("The operator name has no valid format: target.input")
        return name_spl

    @property
    def target(self):
        """Returns target flavor name (given by the first part of the name)"""
        return self._split_name()[0]

    @property
    def input(self):
        """Returns input flavor name (given by the second part of the name)"""
        return self._split_name()[1]


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
    if label[0] not in "duscbt" or label[1] not in ["+", "-"]:
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
