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

    def __str__(self):
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
        """Returns target flavour name (given by the first part of the name)"""
        return self._split_name()[0]

    @property
    def input(self):
        """Returns input flavour name (given by the second part of the name)"""
        return self._split_name()[1]

    @property
    def is_physical(self):
        """Lives inside a :class:`PhysicalOperator`? determined by name"""
        return self.name not in full_labels


def pids_from_intrinsic_evol(label, nlf):
    """
    Obtain the list of pids with their corresponding weight, that are contributing to ``evol``

    Parameters
    ----------
        evol : str
            evolution label
        nlf : int
            maximum number of light flavors

    Returns
    -------
        m : dict
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
        weights = br.rotate_pm_to_flavor(label)
    return dict(zip(br.flavor_basis_pids, weights))


def get_range(evol_labels):
    """
    Determine the number of light and heavy flavors participating in the input and output

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
