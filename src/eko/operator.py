"""
    This module contains the operator class
"""

import numpy as np
import numba as nb
import eko.mellin as mellin
from eko.utils import merge_dicts
import logging
logger = logging.getLogger(__name__)

# evolution basis names
Vs = ["V3", "V8", "V15", "V24", "V35"]
Ts = ["T3", "T8", "T15", "T24", "T35"]

def _run_kernel_integrands(singlet_integrands, nonsinglet_integrands, delta_t, xgrid):
    # TODO: move to the OpMaster
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

        # insert operators
        ret = {"operators": {}, "operator_errors": {}}
        ret["operators"]["S_qq"] = output_array[:, :, 0, 0]
        ret["operators"]["S_qg"] = output_array[:, :, 1, 0]
        ret["operators"]["S_gq"] = output_array[:, :, 2, 0]
        ret["operators"]["S_gg"] = output_array[:, :, 3, 0]
        ret["operator_errors"]["S_qq"] = output_array[:, :, 0, 1]
        ret["operator_errors"]["S_qg"] = output_array[:, :, 1, 1]
        ret["operator_errors"]["S_gq"] = output_array[:, :, 2, 1]
        ret["operator_errors"]["S_gg"] = output_array[:, :, 3, 1]

        return ret, op_dict

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

        ns_names = ["NS_p", "NS_m", "NS_v"]
        op_dict = {}
        for _, name in enumerate(ns_names):
            op = np.array(operators)
            op_err = np.array(operator_errors)
            new_op = OperatorMember(op, op_err, name)
            op_dict[name] = new_op

        # in LO v=+=-
        ret = {
            "operators": {"NS_p": op.copy(), "NS_m": op.copy(), "NS_v": op.copy()},
            "operator_errors": {
                "NS_p": op_err.copy(),
                "NS_m": op_err.copy(),
                "NS_v": op_err.copy(),
            },
        }
        return ret, op_dict

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
        new_err = np.sqrt(np.matmul(lval, rerror) + np.matmul(rval, ler))
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


class Operator:
    """ Computed only upon calling compute """

    def __init__(self, delta_t, xgrid, integrands_ns, integrands_s, metadata):
        # Save the metadata
        self._metadata = metadata
        # Get ready for the computation
        singlet, nons = _run_kernel_integrands(integrands_s, integrands_ns, delta_t, xgrid)
        self._compute_singlet = singlet
        self._compute_nonsinglet = nons
        self._internal_ret = None
        self._internal_ops = {}

    @property
    def nf(self):
        return self._metadata['nf']

    @property
    def qref(self):
        return self._metadata['qref']

    @property
    def q(self):
        return self._metadata['q']

    def pdf_space(self, scheme = None):
        # TODO do this with more elegance
        step = self._internal_ret
        # join elements
        ret = {"operators": {}, "operator_errors": {}}

        def set_helper(to, from1):
            # Save everything twice for now for some reason
            ret["operators"][to] = step["operators"][from1]
            ret["operator_errors"][to] = step["operator_errors"][from1]

        # join quarks flavors
        # v.v = V
        set_helper("V.V", "NS_v")
        for v, t in list(zip(Vs, Ts))[: self.nf - 1]:  # already there
            set_helper(f"{v}.{v}", "NS_m")
            set_helper(f"{t}.{t}", "NS_p")
        if scheme != 'FFNS':
            for v, t in list(zip(Vs, Ts))[self.nf - 1 :]:  # generate dynamically
                set_helper(f"{v}.V", "NS_v")
                set_helper(f"{t}.S", "S_qq")
                set_helper(f"{t}.g", "S_qg")
        # Singlet + gluon
        set_helper("S.S", "S_qq")
        set_helper("S.g", "S_qg")
        set_helper("g.S", "S_gq")
        set_helper("g.g", "S_gg")
        return ret

    def compute(self):
        ret_ns, op_members_ns = self._compute_nonsinglet()
        ret_s, op_members_s = self._compute_singlet()
        self._computed = True
        step = merge_dicts(ret_ns, ret_s)
        self._internal_ops.update(op_members_s)
        self._internal_ops.update(op_members_ns)
        self._internal_ret = step

    def __mul__(self, pdf_object):
        """ The multiplication operator needs to act on a pdf object
        This pdf object can be a LHAPDF array or a NNPDF array or
        whatever """
        if pdf_object == "nnpdf":
            return self._multiply_nnpdf(pdf_object)

    def multiply_nnpdf(self, nnpdf_object):
        """ Act on a NNPDF pdf object """
        return nnpdf_object 
