# -*- coding: utf-8 -*-
r"""
The write-up of the matching conditions is given in
:doc:`Matching Conditions </theory/Matching>`.

"""

import numpy as np


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

def get_range(evol_labels):
    """
    Determine the number of light and heavy flavors participating in the input and output

    Returns
    -------
        nf_in : int
            number of light flavors in the input
        nf_out : int
            number of light flavors in the output
        intrinsic_range_in : list(int)
            list of heavy flavors in the input
        intrinsic_range_out : list(int)
            list of heavy flavors in the output
    """
    nf_in = 3
    nf_out = 3
    intrinsic_range_in = []
    intrinsic_range_out = []
    def update(label):
        nf = 3
        intrinsic_range = []
        if label[0] == "T":
            nf = round(np.sqrt(int(label[1:]) + 1))
        elif label[1] in ["+", "-"]:
            intrinsic_range.append(4+hqfl.index(label[0]))
        return nf, intrinsic_range
    hqfl = "cbt"
    for op in evol_labels:
        nf, intr = update(op.input)
        nf_in = max(nf, nf_in)
        intrinsic_range_in.extend(intr)
        nf, intr = update(op.target)
        nf_out = max(nf, nf_out)
        intrinsic_range_out.extend(intr)

    intrinsic_range_in.sort()
    intrinsic_range_out.sort()
    return nf_in, nf_out, intrinsic_range_in, intrinsic_range_out
