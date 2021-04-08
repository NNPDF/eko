# -*- coding: utf-8 -*-
"""
This module defines the non-trivial matching conditions for the |VFNS| evolution.
"""

import numpy as np

from .. import member


class MatchingCondition(member.OperatorBase):
    @classmethod
    def split_ad_to_evol_map(cls, ome_members, nf, q2_thr, a_s):
        len_xgrid = ome_members["NS"].value.shape[0]
        op_id = member.OpMember(np.eye(len_xgrid), np.zeros((len_xgrid, len_xgrid)))
        # activate one higher element, i.e. where the next heavy quark could participate,
        # without this new heavy quark Vn = V and Tn = S
        n = (nf + 1) ** 2 - 1
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
        # TODO add intrinsic matching
        # add elements which are already active
        for f in range(2, nf + 1):
            n = f ** 2 - 1
            m[f"V{n}.V{n}"] = op_id + a_s ** 2 * ome_members["NS"]
            m[f"T{n}.T{n}"] = op_id + a_s ** 2 * ome_members["NS"]
        # map key to MemberName
        opms = {}
        for k, v in m.items():
            opms[member.MemberName(k)] = v.copy()
        return cls(opms, q2_thr)
