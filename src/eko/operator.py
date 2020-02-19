"""
    This module contains the operator class
"""

import numpy as np
import numba as nb
import eko.mellin as mellin


def _run_nonsinglet(kernel_integrands, delta_t, xgrid):
    operators = []
    operator_errors = []
    grid_size = len(xgrid)
    log_prefix = "computing NS operator - %s"
    for _, xk in enumerate(xgrid):
        extra_args = nb.typed.List()
        extra_args.append(np.log(xk))
        extra_args.append(delta_t)
        # Path parameters
        extra_args.append(-0.4*16/(-1.0+np.log(xk)))
        extra_args.append(1.0)

        results = []
        cut = 1e-2
        for integrand in kernel_integrands:
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
        
def _run_singlet(kernel_integrands, delta_t, xgrid):
    cut = 1e-2
    # perform
    all_output = []
    log_prefix = "computing Singlet operator - %s"
    grid_size = len(xgrid)
    for k, xk in enumerate(xgrid):
        extra_args = nb.typed.List()
        extra_args.append(np.log(xk))
        extra_args.append(-0.4*16/(-1.0+np.log(xk)))
        extra_args.append(1.0)
        results = []
        for integrand in kernel_integrands:
            result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
            results.append(result)
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




class Operator:
    """ Computed only upon calling compute """

    def __init__(self, q_from, q_to, delta_t, nf, ker):
        self.qref = q_from
        self.q = q_to
        self.delta_t = delta_t
        self.nf = nf
        self.kernel = ker
        self._computed = False

    def compute(self, xgrid):
        self._computed = True
        integrands_ns = self.kernel.get_non_singlet_for_nf(self.nf)
        integrands_s = self.kernel.get_singlet_for_nf(self.nf)
        # run non-singlet
        ret_ns = _run_nonsinglet(integrands_ns, self.delta_t, xgrid)
        # run singlet
        ret_s = _run_singlet(integrands_s, self.delta_t, xgrid)
        # join elements
        ret = utils.merge_dicts(ret_ns, ret_s)
        return ret



    def __mul__(self, operator):
        """ Does the internal product of two operators """
        # Check that the operators are compatible

    def __add__(self, operator):
        """ Does the summation of two operators """
