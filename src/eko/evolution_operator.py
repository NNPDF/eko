# -*- coding: utf-8 -*-
r"""
    This module contains all evolution operator classes.

    See :doc:`Operator overview </Code/Operators>`.
"""

import logging
import numpy as np
import numba as nb
import eko.mellin as mellin

logger = logging.getLogger(__name__)

def _get_kernel_integrands(singlet_integrands, nonsinglet_integrands, delta_t, xgrid):
    """
        Return actual integration kernels.

        Parameters
        ----------
            singlet_integrands : list
                kernels for singlet integrations
            nonsinglet_integrands : list
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
    cut = 1e-2 # TODO make 'cut' external parameter?
    grid_size = len(xgrid)
    grid_logx = np.log(xgrid)

    def run_singlet():
        print("Starting singlet") # TODO delegate to logger?
        all_output = []
        log_prefix = "computing Singlet operator - %s"
        # iterate output grid
        for k, logx in enumerate(grid_logx):
            extra_args = nb.typed.List()
            extra_args.append(logx)
            extra_args.append(delta_t)
            # Path parameters
            extra_args.append(0.4 * 16 / (1.0 - logx))
            extra_args.append(1.0)
            results = []
            for integrand_set in singlet_integrands:
                all_res = []
                for integrand in integrand_set:
                    result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
                    all_res.append(result)
                results.append(all_res)
            #out = [[[0,0]]*4]*grid_size
            all_output.append(results)
            log_text = f"{k+1}/{grid_size}"
            logger.info(log_prefix, log_text)
        logger.info(log_prefix, "done.")
        output_array = np.array(all_output)

        # resort
        singlet_names = ["S_qq", "S_qg", "S_gq", "S_gg"]
        op_dict = {}
        for i, name in enumerate(singlet_names):
            op = output_array[:, :, i,0]
            er = output_array[:, :, i,1]
            new_op = OperatorMember(op, er, name)
            op_dict[name] = new_op

        return op_dict

    def run_nonsinglet():
        print("Starting non-singlet") # TODO delegate to logger?
        operators = []
        operator_errors = []
        log_prefix = "computing NS operator - %s"
        # iterate output grid
        for k, logx in enumerate(grid_logx):
            extra_args = nb.typed.List()
            extra_args.append(logx)
            extra_args.append(delta_t)
            # Path parameters
            extra_args.append(0.5)
            extra_args.append(0.0)

            results = []
            for integrand in nonsinglet_integrands:
                result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
                results.append(result)
            operators.append(np.array(results)[:, 0])
            operator_errors.append(np.array(results)[:, 1])
            log_text = f"{k+1}/{grid_size}"
            logger.info(log_prefix, log_text)

        # resort
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
        self.value = value
        self.error = error
        self._name = name

    @property
    def name(self):
        """ full operator name """
        return self._name
    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def target(self):
        """ target flavour name (given by the second part of the name) """
        name_spl = self._name.split(".")
        if len(name_spl) != 2:
            raise TypeError("This operator is not defining any targets")
        return name_spl[1]

    @property
    def input(self):
        """ input flavour name (given by the first part of the name) """
        name_spl = self._name.split(".")
        if len(name_spl) != 2:
            raise TypeError("This operator is not defining any input")
        return name_spl[0]

    def __call__(self, pdf_member):
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
        if isinstance(operator_member, (np.int, np.float, np.integer)):
            rval = operator_member
            rerror = 0.0
            new_name = self.name
        # matrix multiplication
        elif isinstance(operator_member, OperatorMember):
            rval = operator_member.value
            rerror = operator_member.error
            new_name = f"{self.name}.{operator_member.name}"
        else:
            raise NotImplementedError(f"Can't multiply OperatorMember and {type(operator_member)}")
        lval = self.value
        ler = self.error
        new_val = np.matmul(lval, rval)
        # TODO check error propagation
        new_err = np.abs(np.matmul(lval, rerror) + np.matmul(rval, ler))
        return OperatorMember(new_val, new_err, new_name)

    def __add__(self, operator_member):
        if isinstance(operator_member, (np.int, np.float, np.integer)):
            rval = operator_member
            rerror = 0.0
            new_name = self.name
        elif isinstance(operator_member, OperatorMember):
            rval = operator_member.value
            rerror = operator_member.error
            new_name = f"{self.name}+{operator_member.name}"
        else:
            raise NotImplementedError(f"Can't sum OperatorMember and {type(operator_member)}")
        new_val = self.value + rval
        # TODO check error propagation
        new_err = np.sqrt(pow(self.error,2)+pow(rerror,2))
        return OperatorMember(new_val, new_err, new_name)

    def __sub__(self, operator_member):
        self.__add__(-operator_member)

    # These are necessary to deal with python operators such as sum
    def __radd__(self, operator_member):
        if isinstance(operator_member, OperatorMember):
            return operator_member.__add__(self)
        else:
            return self.__add__(operator_member)

    def __rsub__(self, operator_member):
        return self.__radd__(-operator_member)

    def __eq__(self, operator_member):
        return np.allclose(self.value, operator_member.value)

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
            xgrid : np.array
                list of basis x points
            q2_final : float
                final scale
    """
    def __init__(self, op_members, xgrid, q2_final):
        self.op_members = op_members
        self.xgrid = xgrid
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
        ops = self.get_operator_matrices()
        return apply_PDF_to_operator(ops,pdf_lists)

    def get_operator_matrices(self):
        """
            Returns the matrix representation of all members and their errors

            Returns
            -------
                ret : dict
                    the members are stored under the ``operators`` key and their
                    errors under the ``operator_errors`` key. They are labeled as
                    ``{outputPDF}.{inputPDF}``.
        """
        # add matrices
        ret = { "operators" : {}, "operator_errors" : {} }
        for key, new_op in self.op_members.items():
            ret["operators"][key] = new_op.value
            ret["operator_errors"][key] = new_op.error
        return ret

def apply_PDF_to_operator(ret, pdf_lists):
    """
        Apply PDFs to the EKOs provided by :meth:`PhysicalOperator.get_operator_matrices`.

        It assumes as input the PDFs as dictionary in evolution basis with:

        .. code-block:: python

            pdf = {
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
    for k in ret["operators"]:
        out_key, in_key = k.split(".")
        # basis vector available?
        if in_key not in pdf_lists:
            # thus can I not complete the calculation for this out_key?
            if out_key in outs:
                outs[out_key] = None
            continue
        op = ret["operators"][k]
        op_err = ret["operator_errors"][k]
        # is out_key new?
        if out_key not in outs:
            # set output
            outs[out_key] = np.matmul(op, pdf_lists[in_key])
            out_errors[out_key] = np.matmul(op_err, pdf_lists[in_key])
        else:
            # is out_key already blocked?
            if outs[out_key] is None:
                continue
            # else add to it
            outs[out_key] += np.matmul(op, pdf_lists[in_key])
            out_errors[out_key] += np.matmul(op_err, pdf_lists[in_key])
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
            integrands_ns : list(function)
                list of non-singlet kernels
            integrands_s : list(function)
                list of singlet kernels
            metadata : dict
                metadata with keys `nf`, `q2ref` and `q2`
    """
    def __init__(self, delta_t, xgrid, integrands_ns, integrands_s, metadata):
        # Save the metadata
        self._metadata = metadata
        self._xgrid = xgrid
        # Get ready for the computation
        singlet, nons = _get_kernel_integrands(integrands_s, integrands_ns, delta_t, xgrid)
        self._compute_singlet = singlet
        self._compute_nonsinglet = nons
        self._computed = False
        self.op_members = {}

    @property
    def nf(self):
        """ number of active flavours """
        return self._metadata['nf']

    @property
    def q2ref(self):
        """ scale reference point """
        return self._metadata['q2ref']

    @property
    def q2(self):
        """ actual scale """
        return self._metadata['q2']

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
                key = f'{name}.{origin}'
                new_ops[key] = self.join_members(op_to_compose, paths, key)
        return PhysicalOperator(new_ops, self.xgrid, q2_final)

    def join_members(self, steps, list_of_paths, name):
        """
            Multiply a list of :class:`OperatorMember` using the given paths.

            Parameters
            ----------
                steps : list(OperatorMember)
                    list of raw operators, with the lowest scale to the right
                list_of_paths : list(list(str))
                    list of paths
                name : str
                    final name

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
                    cur_op = cur_op*new_op
            final_op += cur_op
        final_op.name = name
        return final_op

    def compute(self):
        """ compute the actual operators (i.e. run the integrations) """
        op_members_ns = self._compute_nonsinglet()
        op_members_s = self._compute_singlet()
        self._computed = True
        self.op_members.update(op_members_s)
        self.op_members.update(op_members_ns)
