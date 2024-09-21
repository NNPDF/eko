"""Contains the :class:`PhysicalOperator` class."""

from .. import basis_rotation as br
from .. import member


class PhysicalOperator(member.OperatorBase):
    """Join several fixed flavor scheme operators together.

    - provides the connection between the 7-dimensional anomalous dimension
      basis and the 15-dimensional evolution basis
    - provides the connection between the 15-dimensional evolution basis
      and the 169-dimensional flavor basis


    Parameters
    ----------
        op_members : dict
            list of all members
        q2_final : float
            final scale
    """

    @classmethod
    def ad_to_evol_map(cls, op_members, nf, q2_final, qed=False):
        """Obtain map between the 3-dimensional anomalous dimension basis and
        the 4-dimensional evolution basis.

        .. todo:: in VFNS sometimes IC is irrelevant if nf>=4

        Parameters
        ----------
            op_members : dict
                operator members in anomalous dimension basis
            nf : int
                number of active light flavors
            qed : bool
                activate qed

        Returns
        -------
            m : dict
                map
        """
        # constant elements
        m = {
            "S.S": op_members[(100, 100)],
            "S.g": op_members[(100, 21)],
            "g.g": op_members[(21, 21)],
            "g.S": op_members[(21, 100)],
        }
        if not qed:
            m.update({"V.V": op_members[(br.non_singlet_pids_map["nsV"], 0)]})
            # add elements which are already active
            for f in range(2, nf + 1):
                n = f**2 - 1
                m[f"V{n}.V{n}"] = op_members[(br.non_singlet_pids_map["ns-"], 0)]
                m[f"T{n}.T{n}"] = op_members[(br.non_singlet_pids_map["ns+"], 0)]
        else:
            m.update(
                {
                    "g.ph": op_members[(21, 22)],
                    "g.Sdelta": op_members[(21, 101)],
                    "ph.g": op_members[(22, 21)],
                    "ph.ph": op_members[(22, 22)],
                    "ph.S": op_members[(22, 100)],
                    "ph.Sdelta": op_members[(22, 101)],
                    "S.ph": op_members[(100, 22)],
                    "S.Sdelta": op_members[(100, 101)],
                    "Sdelta.g": op_members[(101, 21)],
                    "Sdelta.ph": op_members[(101, 22)],
                    "Sdelta.S": op_members[(101, 100)],
                    "Sdelta.Sdelta": op_members[(101, 101)],
                    "V.V": op_members[(10200, 10200)],
                    "V.Vdelta": op_members[(10200, 10204)],
                    "Vdelta.V": op_members[(10204, 10200)],
                    "Vdelta.Vdelta": op_members[(10204, 10204)],
                }
            )
            # add elements which are already active
            if nf >= 3:
                m["Td3.Td3"] = op_members[(br.non_singlet_pids_map["ns+d"], 0)]
                m["Vd3.Vd3"] = op_members[(br.non_singlet_pids_map["ns-d"], 0)]
            if nf >= 4:
                m["Tu3.Tu3"] = op_members[(br.non_singlet_pids_map["ns+u"], 0)]
                m["Vu3.Vu3"] = op_members[(br.non_singlet_pids_map["ns-u"], 0)]
            if nf >= 5:
                m["Td8.Td8"] = op_members[(br.non_singlet_pids_map["ns+d"], 0)]
                m["Vd8.Vd8"] = op_members[(br.non_singlet_pids_map["ns-d"], 0)]
            if nf >= 6:
                m["Tu8.Tu8"] = op_members[(br.non_singlet_pids_map["ns+u"], 0)]
                m["Vu8.Vu8"] = op_members[(br.non_singlet_pids_map["ns-u"], 0)]
        # deal with intrinsic heavy quark PDFs
        hqfl = "cbt"
        op_id = member.OpMember.id_like(op_members[(21, 21)])
        for intr_fl in [4, 5, 6]:
            if intr_fl <= nf:  # light quarks are not intrinsic
                continue
            hq = hqfl[intr_fl - 4]  # find name
            # intrinsic means no evolution, i.e. they are evolving with the identity
            m[f"{hq}+.{hq}+"] = op_id.copy()
            m[f"{hq}-.{hq}-"] = op_id.copy()
        # map key to MemberName
        return cls.promote_names(m, q2_final)
