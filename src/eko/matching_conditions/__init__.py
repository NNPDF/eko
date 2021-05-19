# -*- coding: utf-8 -*-
"""
This module defines the matching conditions for the |VFNS| evolution.
"""

import numpy as np

from .. import member


# def invert_matching(op_id, ome_members, a_s, method):
#     """Compute the backward matching condition matrix.
#     Both expanded inversion and exact inversion can be called

#     Parameters
#     ----------
#         op_id : eko.evolution_operator.Operator.op_member
#             Attribute of :class:`~eko.evolution_operator.Operator` corresponding to the identity operator
#         ome_members : eko.operator_matrix_element.OperatorMatrixElement.ome_member
#             Attribute of :class:`~eko.operator_matrix_element.OperatorMatrixElement`
#             containing the operator matrix elements
#         a_s : float
#             value of the strong coupling at the threshold
#         method: ["expanded", "exact"]
#             Method for inverting the matching condition (exact or expanded)

#     Returns
#     -------
#         m : dict
#             matching conditions dictionary
#     """
#     if method == "expanded":
#         m = {
#             "S.S": op_id - a_s ** 2 * (ome_members["NS"] + ome_members["S_qq"]),
#             "S.g": -(a_s ** 2) * ome_members["S_qg"],
#             "g.S": -(a_s ** 2) * ome_members["S_gq"],
#             "g.g": op_id - a_s ** 2 * ome_members["S_gg"],
#             "V.V": op_id - a_s ** 2 * ome_members["NS"],
#         }
#     # This in not going to work!
#     elif method == "exact":
#         det_inv = 1. / (
#             - ome_members["S_gq"] * ome_members["S_qg"] * a_s ** 4 +
#             (1 + ome_members["S_gg"] * a_s ** 2)
#             (1 + (ome_members["NS"] + ome_members["S_qq"]) * a_s ** 2)
#         )
#         m = {
#             "S.S": det_inv * ( op_id + a_s ** 2 * ome_members["S_gg"] ) ,
#             "S.g": det_inv * ( - a_s ** 2 * ome_members["S_qg"],
#             "g.S": det_inv * ( - a_s ** 2 * ome_members["S_gq"],
#             "g.g": det_inv * ( op_id + a_s ** 2 * ( ome_members["NS"] +  ome_members["S_qq"]),
#             "V.V": 1. / ( 1 + a_s ** 2 * ome_members["NS"]),
#         }

# #     return m


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
        cls, ome_members, nf, q2_thr, a_s, intrinsic_range=None, backward_inversion=None
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
            a_s : float
                value of the strong coupling at the threshold
            intrinsic_range : list
                list of intrinsic quark pids
            backward_inversion: str
                None or method for inverting the matching contidtion (exact or expanded)
        """
        len_xgrid = ome_members["NS_qq"].value.shape[0]
        op_id = member.OpMember(np.eye(len_xgrid), np.zeros((len_xgrid, len_xgrid)))
        m = {}

        if backward_inversion == "exact":
            # inversion is alrady done before the integration
            m = {
                "S.S": ome_members["S_qq"],
                "S.g": ome_members["S_qg"],
                "g.S": ome_members["S_gq"],
                "g.g": ome_members["S_gg"],
                "V.V": ome_members["NS_qq"],
            }
        else:
            # backawrd expanded or forward mathcing
            m = {
                "S.S": op_id + a_s ** 2 * (ome_members["NS_qq"] + ome_members["S_qq"]),
                "S.g": a_s ** 2 * ome_members["S_qg"],
                "g.S": a_s ** 2 * ome_members["S_gq"],
                "g.g": op_id + a_s ** 2 * ome_members["S_gg"],
                "V.V": op_id + a_s ** 2 * ome_members["NS_qq"],
            }

        # activate one higher element, i.e. where the next heavy quark could participate,
        # without this new heavy quark Vn = V and Tn = S
        if backward_inversion is None or (
            intrinsic_range is not None and backward_inversion == "expanded"
        ):
            n = (nf + 1) ** 2 - 1
            m.update(
                {
                    f"V{n}.V": op_id + a_s ** 2 * ome_members["NS_qq"],
                    f"T{n}.S": op_id
                    + a_s ** 2 * (ome_members["NS_qq"] - nf * ome_members["S_qq"]),
                    f"T{n}.g": -nf * a_s ** 2 * ome_members["S_qg"],
                }
            )
        # add elements which are already active
        for f in range(2, nf + 1):
            n = f ** 2 - 1
            m[f"V{n}.V{n}"] = m["V.V"]
            m[f"T{n}.T{n}"] = m["V.V"]

        # intrinsic matching
        if intrinsic_range is not None:
            hqfl = "cbt"
            for intr_fl in intrinsic_range:
                hq = hqfl[intr_fl - 4]  # find name
                if backward_inversion is None:
                    if intr_fl > nf + 1:  # keep the higher quarks as they are
                        m[f"{hq}+.{hq}+"] = op_id
                        m[f"{hq}-.{hq}-"] = op_id
                    elif intr_fl == nf + 1:  # next is comming hq?
                        n = intr_fl ** 2 - 1
                        # add hq to S and V
                        m[f"V.{hq}-"] = op_id
                        m[f"S.{hq}+"] = op_id
                        # e.g. T15 = (u+ + d+ + s+) - 3c+
                        m[f"V{n}.{hq}-"] = -(intr_fl - 1) * op_id
                        m[f"T{n}.{hq}+"] = -(intr_fl - 1) * op_id
                else:
                    # backward matching
                    # one flavor is not evolving anymore need to match
                    n = intr_fl ** 2 - 1
                    if intr_fl == nf:
                        # TODO: check if this is correct
                        if backward_inversion == "exact":
                            # inversion is alrady done before the integration
                            m.update(
                                {
                                    f"{hq}+.S": ome_members["S_Tq"],
                                    f"{hq}+.g": ome_members["S_Tg"],
                                    f"{hq}+.T{n}": ome_members["S_TT"],
                                    f"{hq}-.V": ome_members["NS_Vq"],
                                    f"{hq}-.V{n}": ome_members["NS_VV"],
                                    # add the new contribution to V, S and g
                                    f"V.V{n}": ome_members["NS_qV"],
                                    f"S.T{n}": ome_members["S_qT"],
                                    f"g.T{n}": ome_members["S_gT"],
                                }
                            )

                        elif backward_inversion == "expanded":
                            m.update(
                                {
                                    # build q+ and q-
                                    f"{hq}+.S": 1.0 / nf * m["S.S"],
                                    f"{hq}+.g": 1.0 / nf * m["S.g"],
                                    f"{hq}+.T{n}": -1.0 / nf * m[f"T{n}.T{n}"],
                                    f"{hq}-.V": 1.0 / nf * m["V.V"],
                                    f"{hq}-.V{n}": -1.0 / nf * m[f"V{n}.V{n}"],
                                    # add the new contribution to V, S and g
                                    f"V.V{n}": 1.0 / nf * m[f"V{n}.V"],
                                    f"S.T{n}": 1.0 / nf * m[f"T{n}.S"],
                                    f"g.T{n}": 1.0 / nf * m[f"T{n}.g"],
                                    "V.V": -(nf - 1) / nf * m["V.V"],
                                    "S.S": -(nf - 1) / nf * m["S.S"],
                                    "S.g": -(nf - 1) / nf * m["S.g"],
                                }
                            )
        # map key to MemberName
        return cls.promote_names(m, q2_thr)
