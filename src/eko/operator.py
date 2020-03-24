"""
    This module contains the operator class
"""

import logging
import numpy as np
import numba as nb
import eko.mellin as mellin
from eko.utils import operator_product
logger = logging.getLogger(__name__)

def _get_kernel_integrands(singlet_integrands, nonsinglet_integrands, delta_t, xgrid):
    # Generic parameters
    cut = 1e-2
    grid_size = len(xgrid)
    grid_logx = np.log(xgrid)

    def run_singlet():
        # perform
        print("Starting singlet")
        all_output = []
        log_prefix = "computing Singlet operator - %s"
        for k, logx in enumerate(grid_logx):
            extra_args = nb.typed.List()
            extra_args.append(logx)
            extra_args.append(delta_t)
            # Path parameters
            extra_args.append(-0.4*16/(-1.0+logx))
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

        singlet_names = ["S_qq", "S_qg", "S_gq", "S_gg"]
        op_dict = {}
        for i, name in enumerate(singlet_names):
            op = output_array[:, :, i,0]
            er = output_array[:, :, i,1]
            new_op = OperatorMember(op, er, name)
            op_dict[name] = new_op

        return op_dict

    def run_nonsinglet():
        print("Starting non-singlet")
        operators = []
        operator_errors = []
        log_prefix = "computing NS operator - %s"
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
    """ Operator members """

    def __init__(self, value, error, name):
        self.value = value
        self.error = error
        self._name = name

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def target(self):
        name_spl = self._name.split(".")
        if len(name_spl) != 2:
            raise TypeError("This operator is not defining any targets")
        return name_spl[1]

    @property
    def input(self):
        name_spl = self._name.split(".")
        if len(name_spl) != 2:
            raise TypeError("This operator is not defining any input")
        return name_spl[0]

    def __call__(self, pdf_member):
        """ The operator member can act on a pdf member """
        result = np.dot(self.value, pdf_member)
        error = np.dot(self.error, pdf_member)
        return result, error


    def __str__(self):
        return self.name

    def __mul__(self, operator_member):
        if isinstance(operator_member, (np.int, np.float, np.integer)):
            rval = operator_member
            rerror = 0.0
            new_name = self.name
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
    """ Physical Operator exposed to the outside world
    This operator is computed via the composition method of the
    Operator class.

    This operator can act on PDFs through the __call__ method
    """

    def __init__(self, op_members, xgrid):
        self.op_members = op_members
        self.xgrid = xgrid

    def _get_corresponding_op(self, input_pdf_name):
        # TODO change how this work internally
        active_op = []
        for _, op in self.op_members.items():
            if op.input == input_pdf_name:
                active_op.append(op)
        return active_op

    def _apply_evolution_basis(self, pdf):
        """ Apply the operator on the evolution basis
        Assumes pdf is a dictionary with:
            pdf = {
            'metadata' : some info,
            'members' : {
                'V' : tensor,
                'g' : tensor,
                }
            }
        """
        # Generate the return pdf as a copy of the original one
        # with all members set to 0
        return_members = {}
        for member, item in pdf['members'].items():
            return_members[member] = np.zeros_like(item)
        for member_name, value in pdf['members'].items():
            act_ops = self._get_corresponding_op(member_name)
            for op in act_ops:
                res, err = op(value)
                return_members[op.target] += res
        return_pdf = {
                'metadata' : pdf['metadata'],
                'members' : return_members
                }
        return return_pdf

    def __call__(self, pdf):
        """ pdf """
        return self._apply_evolution_basis(pdf)
        # TODO fill this up, check whether flavour or evol basis
        # and act depending on that

    # TODO this is a legacy wrapper as the benchmark files use this dictionary
    @property
    def ret(self):
        ret = { "operators" : {}, "operator_errors" : {}, "xgrid" : self.xgrid }
        for key, new_op in self.op_members.items():
            ret["operators"][key] = new_op.value
            ret["operator_errors"][key] = new_op.error
        return ret

class Operator:
    """ Computed only upon calling compute """

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
        return self._metadata['nf']

    @property
    def q2ref(self):
        return self._metadata['q2ref']

    @property
    def q2(self):
        return self._metadata['q2']

    @property
    def xgrid(self):
        return self._xgrid

    def compose(self, op_list, instruction_set):
        if not self._computed:
            self.compute()
        op_to_compose = [self.op_members] + [i.op_members for i in reversed(op_list)]
        new_op = {}
        for name, instructions in instruction_set:
            for origin, paths in instructions.items():
                key = f'{name}.{origin}'
                new_op[key] = operator_product(op_to_compose, paths, key)
        return PhysicalOperator(new_op, self.xgrid)

    def compute(self):
        op_members_ns = self._compute_nonsinglet()
        op_members_s = self._compute_singlet()
        self._computed = True
        self.op_members.update(op_members_s)
        self.op_members.update(op_members_ns)
