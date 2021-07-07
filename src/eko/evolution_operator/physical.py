# -*- coding: utf-8 -*-
import numpy as np

from .. import basis_rotation as br
from .. import member
from . import flavors


class PhysicalOperator(member.OperatorBase):
    """
    This joins several fixed flavor scheme operators together.

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
    def ad_to_evol_map(cls, op_members, nf, q2_final, intrinsic_range):
        """
        Obtain map between the 3-dimensional anomalous dimension basis and the
        4-dimensional evolution basis.

        .. todo:: in VFNS sometimes IC is irrelevant if nf>=4

        Parameters
        ----------
            op_members : dict
                operator members in anomalous dimension basis
            nf : int
                number of active light flavors
            intrinsic_range : sequence
                intrinsic heavy flavors

        Returns
        -------
            m : dict
                map
        """
        # constant elements
        m = {
            "S.S": op_members["S_qq"],
            "S.g": op_members["S_qg"],
            "g.g": op_members["S_gg"],
            "g.S": op_members["S_gq"],
            "V.V": op_members["NS_v"],
        }
        # add elements which are already active
        for f in range(2, nf + 1):
            n = f ** 2 - 1
            m[f"V{n}.V{n}"] = op_members["NS_m"]
            m[f"T{n}.T{n}"] = op_members["NS_p"]
        # deal with intrinsic heavy quark pdfs
        if intrinsic_range is not None:
            hqfl = "cbt"
            op_id = member.OpMember.id_like(op_members["NS_v"])
            for intr_fl in intrinsic_range:
                if intr_fl <= nf:  # light quarks are not intrinsic
                    continue
                hq = hqfl[intr_fl - 4]  # find name
                # intrinsic means no evolution, i.e. they are evolving with the identity
                m[f"{hq}+.{hq}+"] = op_id.copy()
                m[f"{hq}-.{hq}-"] = op_id.copy()
        # map key to MemberName
        return cls.promote_names(m, q2_final)

    def to_flavor_basis_tensor(self):
        """
        Convert the computations into an rank 4 tensor over flavor operator space and
        momentum fraction operator space.

        Returns
        -------
            tensor : numpy.ndarray
                EKO
        """
        nf_in, nf_out = flavors.get_range(self.op_members.keys())
        len_pids = len(br.flavor_basis_pids)
        len_xgrid = list(self.op_members.values())[0].value.shape[0]
        # dimension will be pids^2 * xgrid^2
        value_tensor = np.zeros((len_pids, len_xgrid, len_pids, len_xgrid))
        error_tensor = value_tensor.copy()
        for name, op in self.op_members.items():
            in_pids = flavors.pids_from_intrinsic_evol(name.input, nf_in, False)
            out_pids = flavors.pids_from_intrinsic_evol(name.target, nf_out, True)
            for out_idx, out_weight in enumerate(out_pids):
                for in_idx, in_weight in enumerate(in_pids):
                    # keep the outer index to the left as we're mulitplying from the right
                    value_tensor[
                        out_idx,  # output pid (position)
                        :,  # output momentum fraction
                        in_idx,  # input pid (position)
                        :,  # input momentum fraction
                    ] += out_weight * (op.value * in_weight)
                    error_tensor[
                        out_idx,  # output pid (position)
                        :,  # output momentum fraction
                        in_idx,  # input pid (position)
                        :,  # input momentum fraction
                    ] += out_weight * (op.error * in_weight)
        return value_tensor, error_tensor
