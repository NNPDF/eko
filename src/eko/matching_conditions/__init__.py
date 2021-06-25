# -*- coding: utf-8 -*-
"""
This module defines the matching conditions for the |VFNS| evolution.
"""

from .. import member
from ..evolution_operator import flavors


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
        cls,
        ome_members,
        nf,
        q2_thr,
        intrinsic_range,
    ):
        """
        Create the instance from the |OME|.

        Parameters
        ----------
            ome_members : eko.operator_matrix_element.OperatorMatrixElement.ome_member
                Attribute of :class:`~eko.operator_matrix_element.OperatorMatrixElement`
                containing the |OME|
            nf : int
                number of active flavors *below* the threshold
            q2_thr: float
                threshold value
            intrinsic_range : list
                list of intrinsic quark pids
            is_backward: bool
                True for backward evolution
        """

        m = {
            "S.S": ome_members["S_qq"],
            "S.g": ome_members["S_qg"],  # This is always zero for the time being
            "g.S": ome_members["S_gq"],
            "g.g": ome_members["S_gg"],
            "V.V": ome_members["NS_qq"],
        }

        # add elements which are already active
        for f in range(2, nf + 1):
            n = f ** 2 - 1
            m[f"V{n}.V{n}"] = m["V.V"]
            m[f"T{n}.T{n}"] = m["V.V"]

        # activate the next heavy quark
        hq = flavors.quark_names[nf]
        m.update(
            {
                f"{hq}-.V": ome_members["NS_Hq"],
                f"{hq}+.S": ome_members["S_Hq"],
                f"{hq}+.g": ome_members["S_Hg"],
            }
        )

        # intrinsic matching
        if len(intrinsic_range) != 0:
            op_id = member.OpMember.id_like(ome_members["NS_qq"])
            for intr_fl in intrinsic_range:
                ihq = flavors.quark_names[intr_fl - 1]  # find name
                if intr_fl > nf + 1:
                    # keep the higher quarks as they are
                    m[f"{ihq}+.{ihq}+"] = op_id.copy()
                    m[f"{ihq}-.{ihq}-"] = op_id.copy()
                elif intr_fl == nf + 1:
                    # match the missing contibution from h+ and h-
                    m.update(
                        {
                            f"{ihq}+.{ihq}+": ome_members["S_HH"],
                            # f"S.{ihq}+": ome_members["S_qH"],
                            f"g.{ihq}+": ome_members["S_gH"],
                            f"{ihq}-.{ihq}-": ome_members["NS_HH"],
                            # f"V.{ihq}-": ome_members["NS_qH"],
                        }
                    )
        return cls.promote_names(m, q2_thr)
