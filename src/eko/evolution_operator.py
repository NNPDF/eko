# -*- coding: utf-8 -*-
r"""
    This module contains all evolution operator classes.

    See :doc:`Operator overview </Code/Operators>`.
"""

import logging
from numbers import Number

import numpy as np
import numba as nb

import eko.mellin as mellin

logger = logging.getLogger(__name__)


def _get_kernel_integrands(
    singlet_integrands, nonsinglet_integrands, delta_t, xgrid, cut=1e-2
):
    """
        Return actual integration kernels.

        Parameters
        ----------
            singlet_integrands : list(list(callable))
                kernels for singlet integrations
            nonsinglet_integrands : list(callable)
                kernels for non-singlet integrations
            delta_t : float
                evolution distance
            xgrid : np.array
                basis grid

        Returns
        -------
            run_singlet : function
                singlet integration routine
            run_nonsinglet : function
                non-singlet integration routine
    """
    # Generic parameters
    grid_size = len(xgrid)
    grid_logx = np.log(xgrid)

    def run_singlet():
        logger.info("Starting singlet")
        all_output = []
        # iterate output grid
        for k, logx in enumerate(grid_logx):
            extra_args = nb.typed.List()
            extra_args.append(logx)
            extra_args.append(delta_t)
            # Path parameters
            extra_args.append(0.4 * 16 / (1.0 - logx))
            extra_args.append(1.0)
            # iterate (nested) kernels
            results = []
            for integrand_set in singlet_integrands:
                all_res = []
                for integrand in integrand_set:
                    result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
                    all_res.append(result)
                results.append(all_res)
            # out = [[[0,0]]*4]*grid_size
            all_output.append(results)
            logger.info("computing Singlet operator - %d/%d", k + 1, grid_size)
        output_array = np.array(all_output)

        # resort result: key -> op
        singlet_names = ["S_qq", "S_qg", "S_gq", "S_gg"]
        op_dict = {}
        for i, name in enumerate(singlet_names):
            op = output_array[:, :, i, 0]
            er = output_array[:, :, i, 1]
            new_op = OperatorMember(op, er, name)
            op_dict[name] = new_op

        return op_dict

    def run_nonsinglet():
        logger.info("Starting non-singlet")
        operators = []
        operator_errors = []
        # iterate output grid
        for k, logx in enumerate(grid_logx):
            extra_args = nb.typed.List()
            extra_args.append(logx)
            extra_args.append(delta_t)
            # Path parameters
            extra_args.append(0.5)
            extra_args.append(0.0)
            # iterate kernels
            results = []
            for integrand in nonsinglet_integrands:
                result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
                results.append(result)
            operators.append(np.array(results)[:, 0])
            operator_errors.append(np.array(results)[:, 1])
            logger.info("computing NS operator - %d/%d", k + 1, grid_size)

        # resort result: key -> op
        # in LO v=+=-
        ns_names = ["NS_p", "NS_m", "NS_v"]
        op_dict = {}
        for _, name in enumerate(ns_names):
            op = np.array(operators)
            op_err = np.array(operator_errors)
            new_op = OperatorMember(op, op_err, name)
            op_dict[name] = new_op

        return op_dict

    return run_singlet, run_nonsinglet


class OperatorMember:
    """
        A single operator for a specific element in evolution basis.

        The :class:`OperatorMember` provide some basic mathematical operations such as products.
        It can also be applied to a pdf vector by the `__call__` method.
        This class will never be exposed to the outside, but will be an internal member
        of the :class:`Operator` and :class:`PhysicalOperator` instances.

        Parameters
        ----------
            value : np.array
                operator matrix
            error : np.array
                operator error matrix
            name : str
                operator name
    """

    def __init__(self, value, error, name):
        self.value = np.array(value)
        self.error = np.array(error)
        self.name = name

    def _split_name(self):
        """Splits the name according to target.input"""
        # we need to do this late, as in raw mode the name to not follow this principle
        name_spl = self.name.split(".")
        if len(name_spl) != 2:
            raise ValueError("The operator name has no valid format: target.input")
        for k in [0, 1]:
            name_spl[k] = name_spl[k].strip()
            if len(name_spl[k]) <= 0:
                raise ValueError("The operator name has no valid format: target.input")
        return name_spl

    @property
    def target(self):
        """Returns target flavour name (given by the first part of the name)"""
        return self._split_name()[0]

    @property
    def input(self):
        """Returns input flavour name (given by the second part of the name)"""
        return self._split_name()[1]

    @property
    def is_physical(self):
        """Lives inside a :class:`PhysicalOperator`? determined by name"""
        for n in ["NS_p", "NS_m", "NS_v", "S_qq", "S_qg", "S_gq", "S_gg"]:
            if self.name.find(n) >= 0:
                return False
        return True

    def apply_pdf(self, pdf_member):
        """
            The operator member can act on a pdf member.

            Parameters
            ----------
                pdf_member : np.array
                    pdf vector

            Returns
            -------
                result : float
                    higher scale pdf
                error : float
                    evolution uncertainty to pdf at higher scale
        """
        result = np.dot(self.value, pdf_member)
        error = np.dot(self.error, pdf_member)
        return result, error

    def __str__(self):
        return self.name

    def __mul__(self, operator_member):
        # scalar multiplication
        if isinstance(operator_member, Number):
            one = np.identity(len(self.value))
            rval = operator_member * one
            rerror = 0.0 * one
            new_name = self.name
        # matrix multiplication
        elif isinstance(operator_member, OperatorMember):
            # check compatibility
            if self.is_physical != operator_member.is_physical:
                raise ValueError(f"Operators do not live in the same space!")
            if self.is_physical:
                if self.input != operator_member.target:
                    raise ValueError(
                        f"Can not sum {operator_member.name} and {self.name} OperatorMembers!"
                    )
                new_name = f"{self.target}.{operator_member.input}"
            else:
                new_name = f"{self.name}.{operator_member.name}"
            rval = operator_member.value
            rerror = operator_member.error
        else:
            raise NotImplementedError(
                f"Can't multiply OperatorMember and {type(operator_member)}"
            )
        lval = self.value
        ler = self.error
        new_val = np.matmul(lval, rval)
        # TODO check error propagation
        new_err = np.abs(np.matmul(lval, rerror)) + np.abs(np.matmul(ler, rval))
        return OperatorMember(new_val, new_err, new_name)

    def __add__(self, operator_member):
        if isinstance(operator_member, Number):
            # we only allow the integer 0 as alias for the true zero operator
            if operator_member != 0:
                raise ValueError(
                    "The only integer we can sum to is 0 (as alias for the zero operator)"
                )
            rval = operator_member
            rerror = 0.0
            new_name = self.name
        elif isinstance(operator_member, OperatorMember):
            # check compatibility
            if self.is_physical != operator_member.is_physical:
                raise ValueError(f"Operators do not live in the same space!")
            if self.is_physical and operator_member.name != self.name:
                raise ValueError(
                    f"Can not sum {operator_member.name} and {self.name} OperatorMembers!"
                )
            rval = operator_member.value
            rerror = operator_member.error
            new_name = self.name
        else:
            raise NotImplementedError(
                f"Can't sum OperatorMember and {type(operator_member)}"
            )
        new_val = self.value + rval
        new_err = self.error + rerror
        return OperatorMember(new_val, new_err, new_name)

    def __neg__(self):
        return self.__mul__(-1)

    def __eq__(self, operator_member):
        return np.allclose(self.value, operator_member.value)

    def __sub__(self, operator_member):
        return self.__add__(-operator_member)

    def __radd__(self, operator_member):
        return self.__add__(operator_member)

    def __rsub__(self, operator_member):
        return self.__radd__(-operator_member)

    def __rmul__(self, operator_member):
        return self.__mul__(operator_member)

    @staticmethod
    def join(steps, list_of_paths):
        """
            Multiply a list of :class:`OperatorMember` using the given paths.

            Parameters
            ----------
                steps : list(list(OperatorMember))
                    list of raw operators, with the lowest scale to the right
                list_of_paths : list(list(str))
                    list of paths

            Returns
            -------
                final_op : OperatorMember
                    joined operator
        """
        final_op = 0
        for path in list_of_paths:
            cur_op = None
            for step, member in zip(steps, path):
                new_op = step[member]
                if cur_op is None:
                    cur_op = new_op
                else:
                    cur_op = cur_op * new_op
            final_op += cur_op
        return final_op


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

    def __call__(self, pdf_lists):
        """
            Apply PDFs to the EKOs.

            Parameters
            ----------
                pdf_lists : dict
                    PDFs in evolution basis as list on the corresponding xgrid

            Returns
            -------
                out : dict
                    evolved PDFs
                out_errors : dict
                    associated errors of the evolved PDFs
        """
        return self.apply_pdf(pdf_lists)

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
            new_ops[key] = OperatorMember.join(op_to_compose, paths)
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


class Operator:
    """
        Internal representation of a single EKO.

        The actual matrices are computed only upon calling :meth:`compute`.
        :meth:`compose` will generate the :class:`PhysicalOperator` for the outside world.
        If not computed yet, :meth:`compose` will call :meth:`compute`.

        Parameters
        ----------
            delta_t : float
                Evolution distance
            xgrid : np.array
                basis interpolation grid
            integrands_ns : list(callable)
                list of non-singlet kernels
            integrands_s : list(list(callable))
                list of singlet kernels
            metadata : dict
                metadata with keys `nf`, `q2ref` and `q2`
            mellin_cut : float
                cut to the upper limit in the mellin inversion
    """

    def __init__(
        self, delta_t, xgrid, integrands_ns, integrands_s, metadata, mellin_cut=1e-2
    ):
        # Save the metadata
        self._metadata = metadata
        self._xgrid = xgrid
        # Get ready for the computation
        singlet, nons = _get_kernel_integrands(
            integrands_s, integrands_ns, delta_t, xgrid, cut=mellin_cut
        )
        # TODO make 'cut' external parameter?
        self._compute_singlet = singlet
        self._compute_nonsinglet = nons
        self._computed = False
        self.op_members = {}

    @property
    def nf(self):
        """ number of active flavours """
        return self._metadata["nf"]

    @property
    def q2ref(self):
        """ scale reference point """
        return self._metadata["q2ref"]

    @property
    def q2(self):
        """ actual scale """
        return self._metadata["q2"]

    @property
    def xgrid(self):
        """ underlying basis grid """
        return self._xgrid

    def compose(self, op_list, instruction_set, q2_final):
        """
            Compose all :class:`Operator` together.

            Calls :meth:`compute`, if necessary.

            Parameters
            ----------
                op_list : list(Operator)
                    list of operators to merge
                instruction_set : dict
                    list of instructions (generated by :class:`eko.thresholds.FlavourTarget`)
                q2_final : float
                    final scale

            Returns
            -------
                op : PhysicalOperator
                    final operator
        """
        # compute?
        if not self._computed:
            self.compute()
        # prepare operators
        op_to_compose = [self.op_members] + [i.op_members for i in reversed(op_list)]
        # iterate operators
        new_ops = {}
        for name, instructions in instruction_set:
            for origin, paths in instructions.items():
                key = f"{name}.{origin}"
                op = OperatorMember.join(op_to_compose, paths)
                # enforce new name
                op.name = key
                new_ops[key] = op
        return PhysicalOperator(new_ops, q2_final)

    def compute(self):
        """ compute the actual operators (i.e. run the integrations) """
        op_members_ns = self._compute_nonsinglet()
        op_members_s = self._compute_singlet()
        self._computed = True
        self.op_members.update(op_members_s)
        self.op_members.update(op_members_ns)
