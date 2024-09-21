"""Defines the matching conditions for the |VFNS| evolution."""

from .. import basis_rotation as br
from .. import member


class MatchingCondition(member.OperatorBase):
    """Matching conditions for |PDF| at threshold.

    The continuation of the (formally) heavy non-singlet distributions
    with either the full singlet :math:`S` or the full valence :math:`V`
    is considered "trivial". Instead at |NNLO| additional terms ("non-trivial")
    enter.
    """

    @classmethod
    def split_ad_to_evol_map(
        cls,
        op_members,
        nf,
        q2_thr,
        qed=False,
    ):
        """Create the instance from the |OME|.

        Parameters
        ----------
            op_members : eko.operator_matrix_element.OperatorMatrixElement.op_members
                Attribute of :class:`~eko.operator_matrix_element.OperatorMatrixElement`
                containing the |OME|
            nf : int
                number of active flavors *below* the threshold
            q2_thr: float
                threshold value
            qed : bool
                activate qed
        """
        m = {
            "S.S": op_members[(100, 100)],
            "S.g": op_members[(100, 21)],
            "g.S": op_members[(21, 100)],
            "g.g": op_members[(21, 21)],
            "V.V": op_members[(200, 200)],
        }

        # add elements which are already active
        if not qed:
            for f in range(2, nf + 1):
                n = f**2 - 1
                m[f"V{n}.V{n}"] = m["V.V"]
                m[f"T{n}.T{n}"] = m["V.V"]
        else:
            m.update(
                {
                    "Sdelta.Sdelta": op_members[(200, 200)],
                    "Vdelta.Vdelta": op_members[(200, 200)],
                    "ph.ph": member.OpMember.id_like(op_members[(200, 200)]).copy(),
                }
            )
            names = {3: "d3", 4: "u3", 5: "d8", 6: "u8"}
            for k in range(3, nf + 1):
                m[f"V{names[k]}.V{names[k]}"] = m["V.V"]
                m[f"T{names[k]}.T{names[k]}"] = m["V.V"]

        # activate the next heavy quark
        hq = br.quark_names[nf]
        m.update(
            {
                # f"{hq}-.V": op_members[(br.matching_hminus_pid, 200)],
                f"{hq}+.S": op_members[(br.matching_hplus_pid, 100)],
                f"{hq}+.g": op_members[(br.matching_hplus_pid, 21)],
            }
        )

        # intrinsic matching
        op_id = member.OpMember.id_like(op_members[(200, 200)])
        for intr_fl in [4, 5, 6]:
            ihq = br.quark_names[intr_fl - 1]  # find name
            if intr_fl > nf + 1:
                # keep the higher quarks as they are
                m[f"{ihq}+.{ihq}+"] = op_id.copy()
                m[f"{ihq}-.{ihq}-"] = op_id.copy()
            elif intr_fl == nf + 1:
                # match the missing contribution from h+ and h-
                m.update(
                    {
                        f"{ihq}+.{ihq}+": op_members[
                            (br.matching_hplus_pid, br.matching_hplus_pid)
                        ],
                        f"S.{ihq}+": op_members[(100, br.matching_hplus_pid)],
                        f"g.{ihq}+": op_members[(21, br.matching_hplus_pid)],
                        f"{ihq}-.{ihq}-": op_members[
                            (br.matching_hminus_pid, br.matching_hminus_pid)
                        ],
                        # f"V.{ihq}-": op_members[(200, br.matching_hminus_pid)],
                    }
                )
        return cls.promote_names(m, q2_thr)
