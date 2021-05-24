# -*- coding: utf-8 -*-
"""
This module defines the matching conditions for the |VFNS| evolution.
"""

import numpy as np

from .. import member


class MatchingCondition(member.OperatorBase):
    """
    Matching conditions for |PDF| at threshold.

    The continuation of the (formally) heavy non-singlet distributions
    with either the full singlet :math:`S` or the full valence :math:`V`
    is considered "trivial". Instead at |NNLO| additional terms ("non-trivial")
    enter.
    """

    @classmethod
    def split_ad_to_evol_map(
        cls, ome_members, nf, q2_thr, intrinsic_range, is_backward
    ):
        """
        Create the instance from the operator matrix elements.

        Parameters
        ----------
            ome_members : eko.operator_matrix_element.OperatorMatrixElement.ome_member
                Attribute of :class:`~eko.operator_matrix_element.OperatorMatrixElement`
                containing the operator matrix elements
            nf : int
                number of active flavors *below* the threshold
            q2_thr: float
                threshold value
            intrinsic_range : list
                list of intrinsic quark pids
            is_backward: bool
                True for backward evolution
        """
        len_xgrid = ome_members["NS_qq"].value.shape[0]
        op_id = member.OpMember(np.eye(len_xgrid), np.zeros((len_xgrid, len_xgrid)))
        hqfl = "cbt"
        hq = hqfl[nf - 3]

        m = {
            "S.S": ome_members["S_qq"],
            "S.g": ome_members["S_qg"],
            "g.S": ome_members["S_gq"],
            "g.g": ome_members["S_gg"],
            "V.V": ome_members["NS_qq"],
        }

        # activate one higher element, i.e. where the next heavy quark could participate,
        # without this new heavy quark Vn = V and Tn = S, done separately for intrinsic
        if is_backward is False and len(intrinsic_range) == 0:
            m.update(
                {
                    f"{hq}-.V": ome_members["NS_Hq"],
                    f"{hq}+.S": ome_members["S_Hq"],
                    f"{hq}+.g": ome_members["S_Hg"],
                }
            )

        # add elements which are already active
        for f in range(2, nf + 1):
            n = f ** 2 - 1
            m[f"V{n}.V{n}"] = m["V.V"]
            m[f"T{n}.T{n}"] = m["V.V"]

        # intrinsic matching
        if len(intrinsic_range) != 0:
            for intr_fl in intrinsic_range:
                hq = hqfl[intr_fl - 4]  # find name
                if intr_fl > nf + 1 and is_backward is False:
                    # keep the higher quarks as they are
                    m[f"{hq}+.{hq}+"] = op_id
                    m[f"{hq}-.{hq}-"] = op_id
                elif intr_fl == nf + 1:
                    # match the missing contibution form h+ and h-
                    m.update(
                        {
                            f"{hq}+.{hq}+": ome_members["S_HH"],
                            f"S.{hq}+": ome_members["S_qH"],
                            f"g.{hq}+": ome_members["S_gH"],
                            f"{hq}-.{hq}-": ome_members["NS_HH"],
                            f"V.{hq}-": ome_members["NS_qH"],
                            f"{hq}+.S": ome_members["S_Hq"],
                            f"{hq}+.g": ome_members["S_Hg"],
                            f"{hq}-.V": ome_members["NS_Hq"],
                        }
                    )
        return cls.promote_names(m, q2_thr)
