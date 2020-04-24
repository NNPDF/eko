# -*- coding: utf-8 -*-
r"""
    This module contains all evolution operator classes.

    The classes are nested as follows:

    .. graphviz::
        :name: operators
        :caption: nesting of the operator classes: solid lines means "has many",
                  dashed lines means "creates"
        :align: center

        digraph G {
            bgcolor = transparent

            node [shape=box]
            OperatorGrid [label="OperatorGrid"];
            OperatorMaster [label="OperatorMaster" ];
            test, test1 [style=invis];
            Operator [label="Operator" ];
            PhysicalOperator [label="PhysicalOperator"];
            OperatorMember [label="OperatorMember"];

            {rank=same OperatorMaster test test1}
            {rank=same Operator PhysicalOperator}

            OperatorGrid -> OperatorMaster;
            OperatorMaster -> Operator  [weight=1000];
            Operator -> OperatorMember;
            Operator -> PhysicalOperator [style=dashed len=10];
            PhysicalOperator -> OperatorMember
            OperatorMaster -> test -> test1 [style=invis];
            test1 -> PhysicalOperator [weight=1000 style=invis];
        }

    - :class:`~eko.operator_grid.OperatorGrid`:

        *  this is the master class which administrates all evolution kernel operator tasks
        *  it is instantiated once for each run
        *  it divides the given range of :math:`Q^2` into the necessary threshold crossings and
           creates a :class:`~eko.operator_grid.OperatorMaster` for each
        *  it recollects all necessary operators in the end to create the
           :class:`~eko.evolution_operator.PhysicalOperator`

    - :class:`~eko.operator_grid.OperatorMaster`

        * this represents a configuration for a fixed number of flavours
        * it creates an :class:`~eko.evolution_operator.Operator` for each final scale :math:`Q^2`

    - :class:`~eko.evolution_operator.Operator`

        * this represents a configuration for a fixed final scale :math:`Q^2`
        * this class is only used *internally*
        * its :class:`~eko.evolution_operator.OperatorMember` are only valid in the current
          threshold area

    - :class:`~eko.evolution_operator.PhysicalOperator`

        * this is the exposed equivalent of :class:`~eko.evolution_operator.Operator`,
          i.e. it also lives at at fixed final scale
        * its :class:`~eko.evolution_operator.OperatorMember` are valid from the starting scale
          to the final scale

    - :class:`~eko.evolution_operator.OperatorMember`

        * this represents a single evolution kernel operator
        * inside :class:`~eko.evolution_operator.Operator` they are in "raw" evolution basis, i.e.
          :math:`\tilde{\mathbf{E}}_S, \tilde{E}_{ns}^{\pm,v}`, and they never cross a threshold
        * inside :class:`~eko.evolution_operator.PhysicalOperator` they are in "true" evolution
          basis, i.e. they evolve e.g. :math:`\tilde V, \tilde T_3` etc., so they are a evetually
          a product of the "raw" basis (see :doc:`Matching Conditions </Physics/Matching>`)
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
        PhysicalOperator is exposed to the outside world.

        This operator is computed via the composition method of the
        Operator class.

        This operator can act on PDFs through the `__call__` method.
        Assumes as input a pdf as dictionary with:

        .. code-block:: python

            pdf = {
                'metadata' : some info,
                'members' : {
                    'V' : list,
                    'g' : list,
                    }
            }


        Parameters
        ----------
            op_members : dict
                list of all members
            xgrid : np.array
                list of basis x points
    """
    def __init__(self, op_members, xgrid):
        self.op_members = op_members
        self.xgrid = xgrid

    def _get_corresponding_operators(self, input_pdf_name):
        """
            Searches all operators, that are generated by `input_pdf_name`.

            Parameters
            ----------
                input_pdf_name : str
                    input name

            Returns
            -------
                active_ops : list
                    list with relevant operators
        """
        # TODO change how this work internally
        active_ops = []
        for _, op in self.op_members.items():
            if op.input == input_pdf_name:
                active_ops.append(op)
        return active_ops

    def _apply_evolution_basis(self, pdf):
        """
            Apply the operator on the evolution basis.

            Parameters
            ----------
                pdf : dict
                    input pdf

            Returns
            -------
                return_pdf : dict
                    target pdf
        """
        # Generate the return pdf as a copy of the original one
        # with all members set to 0
        return_members = {}
        for member, item in pdf['members'].items():
            return_members[member] = np.zeros_like(item)
        for member_name, value in pdf['members'].items():
            act_ops = self._get_corresponding_operators(member_name)
            for op in act_ops:
                res, err = op(value)
                return_members[op.target] += res
        return_pdf = {
                'metadata' : pdf['metadata'],
                'members' : return_members
                }
        return return_pdf

    def __call__(self, pdf):
        """
            Apply operator to input pdf

            Parameters
            ----------
                pdf : dict
                    input pdf

            Returns
            -------
                return_pdf : dict
                    target pdf
        """
        # TODO fill this up, check whether flavour or evol basis
        # and act depending on that
        return self._apply_evolution_basis(pdf)

    # TODO this is a legacy wrapper as the benchmark files use this dictionary
    @property
    def ret(self):
        """
            LEGACY WRAPPER

            .. todo:: remove
        """
        ret = { "operators" : {}, "operator_errors" : {}, "xgrid" : self.xgrid }
        for key, new_op in self.op_members.items():
            ret["operators"][key] = new_op.value
            ret["operator_errors"][key] = new_op.error
        return ret

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

    def compose(self, op_list, instruction_set):
        """
            Compose all :class:`Operator` together.

            Calls :meth:`compute`, if necessary.

            Parameters
            ----------
                op_list : list(Operator)
                    list of operators to merge
                instruction_set : dict
                    list of instructions (generated by :class:`eko.thresholds.FlavourTarget`)

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
        return PhysicalOperator(new_ops, self.xgrid)

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
