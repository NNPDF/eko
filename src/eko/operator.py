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

        return ret

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

        op = np.array(operators)
        op_err = np.array(operator_errors)

        # in LO v=+=-
        ret = {
            "operators": {"NS_p": op.copy(), "NS_m": op.copy(), "NS_v": op.copy()},
            "operator_errors": {
                "NS_p": op_err.copy(),
                "NS_m": op_err.copy(),
                "NS_v": op_err.copy(),
            },
        }
        return ret

    return run_singlet, run_nonsinglet


class Operator:
    """ Computed only upon calling compute """

    def __init__(self, delta_t, xgrid, integrands_ns, integrands_s, metadata):
        # Save the metadata
        self._metadata = metadata
        # Get ready for the computation
        singlet, nons = _run_kernel_integrands(integrands_s, integrands_ns, delta_t, xgrid)
        self._compute_singlet = singlet
        self._compute_nonsinglet = nons

    @property
    def nf(self):
        return self._metadata['nf']

    @property
    def qref(self):
        return self._metadata['qref']

    @property
    def q(self):
        return self._metadata['q']


    def compute(self):
        ret_ns = self._compute_nonsinglet()
        ret_s = self._compute_singlet()
        self._computed = True

        ####
        ret_step = merge_dicts(ret_ns, ret_s)

        ret = {"operators": {}, "operator_errors": {}}

        def set_helper(a, b):
            ret["operators"][a] = ret_step["operators"][b]
            ret["operator_errors"][a] = ret_step["operator_errors"][b]

        # join quarks flavors
        set_helper("V.V", "NS_v")
        for v, t in list(zip(Vs, Ts))[: self.nf - 1]:  # provide only computations up to nf
            set_helper(f"{v}.{v}", "NS_m")
            set_helper(f"{t}.{t}", "NS_p")
        # Singlet + gluon
        set_helper("S.S", "S_qq")
        set_helper("S.g", "S_qg")
        set_helper("g.S", "S_gq")
        set_helper("g.g", "S_gg")
        self.ret = ret
        ####


    def __mul__(self, operator):
        """ Does the internal product of two operators """
        # Check that the operators are compatible

    def __add__(self, operator):
        """ Does the summation of two operators """
