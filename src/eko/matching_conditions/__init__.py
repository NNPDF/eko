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
        n = nf ** 2 - 1
        m = {
            f"V{n}.V": op_id,
            f"T{n}.S": op_id,
            "S.S": op_id,
            "g.g": op_id,
            "V.V": op_id,
        }
        # TODO add intrinsic matching
        # add elements which are already active
        for f in range(2, nf):
            n = f ** 2 - 1
            m[f"V{n}.V{n}"] = op_id
            m[f"T{n}.T{n}"] = op_id
        # map key to MemberName
        opms = {}
        for k, v in m.items():
            opms[member.MemberName(k)] = v.copy()
        return cls(opms, q2_thr)
