# -*- coding: utf-8 -*-
"""
This module defines the matching conditions for the |VFNS| evolution.
"""

import numpy as np

from .. import member


def invert_matching(op_id, ome_members, a_s, method = "perturbative"):
    """Compute the backward matching condition matrix.
    Both expanded inversion and exact inversion can be called
    
    Parameters
    ----------
        op_id : eko.operator.Operator.op_member
            Attribute of :class:`~eko.operator.Operator` corresponding to the identity operator
        ome_members : eko.operator_matrix_element.OperatorMatrixElement.ome_member
            Attribute of :class:`~eko.operator_matrix_element.OperatorMatrixElement`
            containing the operator matrix elements
        a_s : float
            value of the strong coupling at the threshold
        method: ["expanded", "exact"]
            Method for inverting the matching contidtion (exact or perturbative)

    Returns
    -------
        m : dict
            matching conditions dictionary
    """
    if method == "expanded":
        m = {
            "S.S": op_id - a_s ** 2 * (ome_members["NS"] + ome_members["S_qq"]),
            "S.g": - a_s ** 2 *ome_members["S_qg"],
            "g.S": - a_s ** 2 * ome_members["S_gq"],
            "g.g": op_id - a_s ** 2 * ome_members["S_gg"],
            "V.V": op_id - a_s ** 2 * ome_members["NS"],
        }
    elif method == "exact":
        len_xgrid = ome_members["NS"].value.shape[0]
        import pdb; pdb.set_trace()
        m = {
            "S.S": np.linalg.inv( (op_id + a_s ** 2 * (ome_members["NS"] + ome_members["S_qq"])).value ) ,
            "S.g": np.linalg.inv( (a_s ** 2 * ome_members["S_qg"]).value ),
            "g.S": np.linalg.inv( (a_s ** 2 * ome_members["S_gq"]).value ),
            "g.g": np.linalg.inv((op_id - a_s ** 2 * ome_members["S_gg"]).value ),
            "V.V": np.linalg.inv((op_id - a_s ** 2 * ome_members["NS"]).value ),
        }
        for key, op in m.items():
            m[key] = member.OpMember(op, np.zeros((len_xgrid, len_xgrid)))
    else:
        raise NotImplementedError( f"Called inversion method {method} is not implemented.")
    return m



class MatchingCondition(member.OperatorBase):
    """
    Matching conditions for |PDF| at threshold.

    The continuation of the (formally) heavy non-singlet distributions
    with either the full singlet :math:`S` or the full valence :math:`V`
    is considered "trivial". Instead at |NNLO| additional terms ("non-trivial")
    enter.
    """

    @classmethod
    def split_ad_to_evol_map(cls, ome_members, nf, q2_thr, a_s, intrinsic_range=None, backward_method=None):
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
            backward_method: str
                None or method for inverting the matching contidtion (exact or expanded)
        """
        len_xgrid = ome_members["NS"].value.shape[0]
        op_id = member.OpMember(np.eye(len_xgrid), np.zeros((len_xgrid, len_xgrid)))
        # activate one higher element, i.e. where the next heavy quark could participate,
        # without this new heavy quark Vn = V and Tn = S
        m = {}
        n = (nf + 1) ** 2 - 1
        if backward_method is not None:
            m = invert_matching(op_id, ome_members, a_s, backward_method)
            m.update({
                f"V{n}.V": member.OpMember(np.zeros(len_xgrid), np.zeros((len_xgrid, len_xgrid))),
                f"T{n}.S": member.OpMember(np.zeros(len_xgrid), np.zeros((len_xgrid, len_xgrid))),
                f"T{n}.g": member.OpMember(np.zeros(len_xgrid), np.zeros((len_xgrid, len_xgrid))),
            })
        else:
            m = {
                f"V{n}.V": op_id + a_s ** 2 * ome_members["NS"],
                f"T{n}.S": op_id
                + a_s ** 2 * (ome_members["NS"] - nf * ome_members["S_qq"]),
                f"T{n}.g": -nf * a_s ** 2 * ome_members["S_qg"],
                "S.S": op_id + a_s ** 2 * (ome_members["NS"] + ome_members["S_qq"]),
                "S.g": a_s ** 2 * ome_members["S_qg"],
                "g.S": a_s ** 2 * ome_members["S_gq"],
                "g.g": op_id + a_s ** 2 * ome_members["S_gg"],
                "V.V": op_id + a_s ** 2 * ome_members["NS"],
            }
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
                if intr_fl > nf + 1:  # keep the higher quarks as they are
                    m[f"{hq}+.{hq}+"] = op_id
                    m[f"{hq}-.{hq}-"] = op_id
                elif intr_fl == nf + 1:  # next is comming hq?
                    n = intr_fl ** 2 - 1
                    # e.g. T15 = (u+ + d+ + s+) - 3c+
                    m[f"V{n}.{hq}-"] = -(intr_fl - 1) * op_id
                    m[f"T{n}.{hq}+"] = -(intr_fl - 1) * op_id
        # map key to MemberName
        opms = {}
        for k, v in m.items():
            opms[member.MemberName(k)] = v.copy()

        return cls(opms, q2_thr)
