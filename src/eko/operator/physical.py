# -*- coding: utf-8 -*-
import numpy as np

from . import flavors

class PhysicalOperator:
    """
    This is exposed to the outside world.

    This operator is computed via the composition method of the
    :class:`Operator` class.


    Parameters
    ----------
        op_members : dict
            list of all members
        q2_final : float
            final scale
    """

    def __init__(self, op_members, q2_final):
        self.op_members = op_members
        self.q2_final = q2_final

    @classmethod
    def ad_to_evol_map(cls, op_members, nf, q2_final, is_vfns, intrinsic_range=None):
        """
        Obtain map between the 3-dimensional anomalous dimension basis and the
        4-dimensional evolution basis.

        .. todo:: update docs, in VFNS sometimes IC is irrelevant if nf>=4
                explain which to keep and which to hide

        Parameters
        ----------
            op_members : dict
                operator members in anomalous dimension basis
            nf : int
                number of active light flavors
            is_vfns : bool
                is |VFNS|?
            intrinsic_range : sequence
                intrinsic heavy flavours

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
        # activate one higher element, i.e. where the next heavy quark could participate,
        # but actually it is trivial
        if is_vfns:
            n = (nf + 1) ** 2 - 1
            # without this new heavy quark Vn = V and Tn = S
            m[f"V{n}.V"] = op_members["NS_v"]
            m[f"T{n}.S"] = op_members["S_qq"]
            m[f"T{n}.g"] = op_members["S_qg"]
        # deal with intrinsic heavy quark pdfs
        if intrinsic_range is not None:
            hqfl = "cbt"
            for f in intrinsic_range:
                hq = hqfl[f - 4]  # find name
                # intrinsic means no evolution, i.e. they are evolving with the identity
                op_id = np.eye(op_members["NS_v"].shape[0])
                if f > nf + 1:  # keep the higher quarks as they are
                    m[f"{hq}+.{hq}+"] = op_id
                    m[f"{hq}-.{hq}-"] = op_id
                elif f == nf + 1:  # next is comming hq?
                    n = f ** 2 - 1
                    if is_vfns:
                        # e.g. T15 = (u+ + d+ + s+) - 3c+
                        m[f"V{n}.{hq}-"] = -(f - 1) * op_id
                        m[f"T{n}.{hq}+"] = -(f - 1) * op_id
                    else:
                        m[f"{hq}+.{hq}+"] = op_id
                        m[f"{hq}-.{hq}-"] = op_id
                else:  # f <= nf
                    if not is_vfns:
                        raise ValueError(
                            f"{hq} is perturbative inside FFNS{nf} so can NOT be intrinsic"
                        )
        opms = {}
        for k,v in m.items():
            opms[flavors.MemberName(k)] = v.copy()
        return cls(m, q2_final)

    def to_flavor_basis(self):
        nf_in, nf_out, intrinsic_range_in, intrinsic_range_out = flavors.get_range(self.op_members.keys())
        # TODO

#        br.rotate_flavor_to_evolution

    def __matmul__(self, other):
        """
        Multiply ``other`` to self.

        Parameters
        ----------
            other : PhysicalOperator
                second factor with a lower initial scale

        Returns
        -------
            p : PhysicalOperator
                self * other
        """
        if not isinstance(other, PhysicalOperator):
            raise ValueError("Can only multiply with another PhysicalOperator")
        # prepare paths
        new_oms = {}
        for my_op in self.op_members.values():
            for other_op in other.op_members.values():
                # ops match?
                if my_op.input != other_op.target:
                    continue
                new_key = my_op.target + "." + other_op.input
                # new?
                if not new_key in new_oms:
                    new_oms[new_key] = my_op @ other_op
                else:  # add element
                    new_oms[new_key] += my_op @ other_op
        return self.__class__(new_oms, self.q2_final)

    def to_raw(self):
        """
        Returns serializable matrix representation of all members and their errors

        Returns
        -------
            ret : dict
                the members are stored under the ``operators`` key and their
                errors under the ``operator_errors`` key. They are labeled as
                ``{outputPDF}.{inputPDF}``.
        """
        # map matrices
        ret = {"operators": {}, "operator_errors": {}}
        for name, op in self.op_members.items():
            ret["operators"][name] = op.value.tolist()
            ret["operator_errors"][name] = op.error.tolist()
        return ret

    def apply_pdf(self, pdf_lists):
        """
        Apply PDFs to the EKOs.

        It assumes as input the PDFs as dictionary in evolution basis with:

        .. code-block:: python

            pdf_lists = {
                'V' : list,
                'g' : list,
                # ...
            }

        Each member has to be evaluated on the corresponding xgrid (which
        is tracked by :class:`~eko.operator_grid.OperatorGrid` and not
        :class:`PhysicalOperator`)

        Parameters
        ----------
            ret : dict
                operator matrices of :class:`PhysicalOperator`
            pdf_lists : dict
                PDFs in evolution basis as list on the corresponding xgrid

        Returns
        -------
            out : dict
                evolved PDFs
            out_errors : dict
                associated errors of the evolved PDFs
        """
        # build output
        outs = {}
        out_errors = {}
        for op in self.op_members.values():
            target, input_pdf = op.target, op.input
            # basis vector available?
            if input_pdf not in pdf_lists:
                # thus can I not complete the calculation for this target
                outs[target] = None
                continue
            # is target new?
            if target not in outs:
                # set output
                outs[target], out_errors[target] = op.apply_pdf(pdf_lists[input_pdf])
            else:
                # is target already blocked?
                if outs[target] is None:
                    continue
                # else add to it
                out, err = op.apply_pdf(pdf_lists[input_pdf])
                outs[target] += out
                out_errors[target] += err
        # remove uncompleted
        outs = {k: outs[k] for k in outs if not outs[k] is None}
        return outs, out_errors
