# -*- coding: utf-8 -*-
from .member import OpMember


class PhysicalOperator:
    """
    This is exposed to the outside world.

    This operator is computed via the composition method of the
    :class:`Operator` class.

    This operator can act on PDFs through the `__call__` method.


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

    def __mul__(self, other):
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
        instructions = {}
        for my_op in self.op_members.values():
            for other_op in other.op_members.values():
                # ops match?
                if my_op.input != other_op.target:
                    continue
                new_key = my_op.target + "." + other_op.input
                path = [my_op.name, other_op.name]
                # new?
                if not new_key in instructions:
                    instructions[new_key] = [path]
                else:  # add element
                    instructions[new_key].append(path)

        # prepare operators
        op_to_compose = [self.op_members] + [other.op_members]
        # iterate operators
        new_ops = {}
        for key, paths in instructions.items():
            new_ops[key] = OpMember.join(op_to_compose, paths)
        return self.__class__(new_ops, self.q2_final)

    def get_raw_operators(self):
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
