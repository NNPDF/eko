"""Defines the SCET kernel."""

from .. import basis_rotation as br
from .. import member


class ScetKernel(member.OperatorBase):
    """
    Scet kernel for |PDF|.

    """

    @classmethod
    def split_ad_to_evol_map(
        cls,
        op_members,
        q2_thr,
    ):
        """
        Create the instance from the |OME|.

        Parameters
        ----------
            op_members : eko.beam_fuctions.SCET_I.op_members
                Attribute of :class:`~eko.beam_fuctions.SCET_I` containing the scet kernels
            nf : int
                number of active flavors *below* the threshold
            q2_thr: float
                dummy value
        """
        
        quark_names = br.flavor_basis_names
        quarks = quark_names[8:]
        
        m = {}

        for q1 in quarks:
            for q2 in quarks:
                if q1 == q2:
                    m[f"{q1}.{q1}"] = op_members[(1, 1)]
                    m[f"{q1}bar.{q1}bar"] = op_members[(1, 1)]
                    m[f"{q1}.{q1}bar"] = op_members[(1, -1)]
                    m[f"{q1}bar.{q1}"] = op_members[(1, -1)]         
                else:
                    m[f"{q1}.{q2}"] = op_members[(1, 2)]
                    m[f"{q2}.{q1}"] = op_members[(1, 2)]
                    m[f"{q1}bar.{q2}bar"] = op_members[(1, 2)]
                    m[f"{q2}bar.{q1}bar"] = op_members[(1, 2)]
                    m[f"{q1}.{q2}bar"] = op_members[(1, -2)]
                    m[f"{q2}bar.{q1}"] = op_members[(1, -2)]
    
        for q in quarks:
            m[f"{q}.g"] = op_members[(1, 21)]
            m[f"{q}bar.g"] = op_members[(1, 21)]  
            m[f"g.{q}"] = op_members[(21, 1)]
            m[f"g.{q}bar"] = op_members[(21, 1)]

        m[f"g.g"] = op_members[(21,21)]
        
        return cls.promote_names(m, q2_thr)
