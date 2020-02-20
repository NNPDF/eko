# -*- coding: utf-8 -*-
"""
    This file contains the main loop for the DGLAP calculations.
"""
import logging
import joblib
import numpy as np
import numba as nb
from yaml import dump

import eko.alpha_s as alpha_s
import eko.interpolation as interpolation
import eko.mellin as mellin
import eko.utils as utils
from eko.kernel_generation import KernelDispatcher
from eko.thresholds import Threshold
from eko.operator_grid import OperatorGrid
from eko.constants import Constants
from eko.alpha_s import StrongCoupling

logger = logging.getLogger(__name__)

def run_dglap(setup):
    r"""
        This function takes a DGLAP theory configuration dictionary
        and performs the solution of the DGLAP equations.

        The EKO :math:`\hat O_{k,j}^{(0)}(t_1,t_0)` is determined in order
        to fullfill the following evolution

        .. math::
            f^{(0)}(x_k,t_1) = \hat O_{k,j}^{(0)}(t_1,t_0) f^{(0)}(x_j,t_0)

        Parameters
        ----------
        setup: dict
            a dictionary with the theory parameters for the evolution

            =============== ========================================================================
            key             description
            =============== ========================================================================
            'PTO'           order of perturbation theory: ``0`` = LO, ...
            'alphas'        reference value of the strong coupling :math:`\alpha_s(\mu_0^2)`
            'xgrid_size'    size of the interpolation grid
            'xgrid_min'     lower boundry of the interpolation grid - defaults to ``1e-7``
            'xgrid_type'    generating function for the interpolation grid - see below
            'log_interpol'  boolean, whether it is log interpolation or not, defaults to `True`
            =============== ========================================================================

        Returns
        -------
        ret: dict
            a dictionary with a defined set of keys

            =================  ====================================================================
            key                description
            =================  ====================================================================
            'xgrid'            list of x-values which build the support of the interpolation
            'targetgrid'       list of x-values which are computed
            'operators'        list of computed operators
            'operator_errors'  list of integration errors associated to the operators
            =================  ====================================================================

        Notes
        -----

        * xgrid_type
            - ``linear``: nodes distributed linear in linear-space
            - ``log``: nodes distributed linear in log-space
            - ``custom``: custom xgrid, supplied by the key ``xgrid``
    """

    # Print theory id setup
    logger.info("Setup: %s", setup)

    # Load constants and compute parameters
    constants = Constants()

    # Setup interpolation
    xgrid = interpolation.generate_xgrid(**setup)
    is_log_interpolation = bool(setup.get("log_interpol", True))
    polynom_rank = setup.get("xgrid_polynom_rank", 4)
    logger.info("Interpolation mode: %s", setup["xgrid_type"])
    logger.info("Log interpolation: %s", is_log_interpolation)

    # Generate the dispatcher for the basis functions
    basis_function_dispatcher = interpolation.InterpolatorDispatcher(
        xgrid, polynom_rank, log=is_log_interpolation
    )
    # Generate the dispatcher for the Kernel functions
    kernel_dispatcher = KernelDispatcher(basis_function_dispatcher, constants)

    # Get the scheme and set up the thresholds if any
    # TODO the setup dictionary is a mess tbh
    FNS = setup["FNS"]
    qref = pow(setup["Q0"],2)
    if FNS != "FFNS":
        qmc = setup['Qmc']
        qmb = setup['Qmb']
        qmt = setup['Qmt']
        threshold_list = pow(np.array([qmc, qmb, qmt]), 2)
        nf = None
    else:
        nf = setup["NfFF"]
        threshold_list = None
    # TODO
    threshold_holder = Threshold(qref = qref, scheme = FNS, threshold_list=threshold_list, nf=nf)

    # Now generate the operator alpha_s class
    alpha_ref = setup['alphas']
    q_ref = pow(setup["Qref"],2)
    alpha_s = StrongCoupling(constants, alpha_ref, q_ref, threshold_holder)

    # And now compute the grid
    op_grid = OperatorGrid(threshold_holder, alpha_s, kernel_dispatcher, xgrid)
    qgrid = [setup["Q2grid"][0]]
    op_grid.compute_qgrid(qgrid)
    new_ret = op_grid.get_op_at_Q(setup["Q2grid"][0])
    ret = {
        "xgrid": xgrid,
        "polynomial_degree": polynom_rank,
        "log": is_log_interpolation,
        "basis": basis_function_dispatcher,
        "operators": {},
        "operator_errors": {},
    }
    ret = utils.merge_dicts(ret, new_ret)
    return ret

def apply_operator(ret, inputs, targetgrid=None):
    # TODO: move to the operator class
    """
        Apply all available operators to the input PDFs.

        Parameters
        ----------
            ret : dict
                operator definitions - return value of `run_dglap`
            inputs : dict
                input PDFs as dictionary name -> function
            targetgrid : array
                if given, interpolates to the pdfs given at targetgrid (instead of xgrid)

        Returns
        ---------
            outs : dict
                output PDFs
            out_errors : dict
                associated error to the output PDFs
    """
    # TODO rotation from the evolution basis to flavor basis? if yes, here!
    # turn inputs into lists
    input_lists = {}
    for k in inputs:
        l = []
        for x in ret["xgrid"]:
            l.append(inputs[k](x))
        input_lists[k] = np.array(l)
    # build output
    outs = {}
    out_errors = {}
    for k in ret["operators"]:
        out_key, in_key = k.split(".")
        # basis vector available?
        if in_key not in inputs:
            # thus can I not complete the calculation for this out_key?
            if out_key in outs:
                outs[out_key] = None
            continue
        op = ret["operators"][k]
        op_err = ret["operator_errors"][k]
        # is out_key new?
        if out_key not in outs:
            # set output
            outs[out_key] = np.matmul(op, input_lists[in_key])
            out_errors[out_key] = np.matmul(op_err, input_lists[in_key])
        else:
            # is out_key already blocked?
            if outs[out_key] is None:
                continue
            # else add to it
            outs[out_key] += np.matmul(op, input_lists[in_key])
            out_errors[out_key] += np.matmul(op_err, input_lists[in_key])
    # interpolate to target grid
    if targetgrid is not None:
        rot = ret["basis"].get_interpolation(targetgrid)
        for k in outs:
            outs[k] = np.matmul(rot, outs[k])
            out_errors[k] = np.matmul(rot, out_errors[k])
    return outs, out_errors


def multiply_operators(step2, step1):
    # TODO move to the operator class
    """
        Multiplies two operators in given order, i.e. step2 > step1.

        Note that the arguments have to be given in decreasing order, i.e.
        for :math:`Q_3^2 > Q_2^2 > Q_1^2` the steps have to be: `step2` (first
        argument) evolves :math:`Q_2^2 \\to Q_3^2` and `step1` (second
        argument)  evolves :math:`Q_1^2 \\to Q_2^2`.

        Parameters
        ----------
            step2 : dict
                last evolution step
            step1 : dict
                first evolution step

        Returns
        -------
            joined : dict
                combined evolution `step2 * step1`
    """
    # check compatibility
    # TODO we should really also test on (full) setup
    if step1["basis"] != step2["basis"]:
        raise ValueError("basis functions do not match.")
    # rebuild
    joined = {
        "xgrid": step1["xgrid"],
        "basis": step1["basis"],
        "operators": {},
        "operator_errors": {},
    }
    # search elements
    keys1 = step1["operators"].keys()
    for k2 in step2["operators"].keys():
        to2, fromm2 = k2.split(".")
        for k1 in keys1:
            to1, fromm1 = k1.split(".")
            # match?
            if fromm2 == to1:
                # join
                newk = f"{to2}.{fromm1}"
                op, op_err = utils.operator_product_helper([step2, step1], [[k2, k1]])
                if newk not in joined["operators"]:
                    joined["operators"][newk] = op
                    joined["operator_errors"][newk] = op_err
                else:
                    joined["operators"][newk] += op
                    joined["operator_errors"][newk] += op_err
    return joined


def get_YAML(ret, stream=None):
    """
        Serialize result as YAML.

        Parameters
        ----------
            ret : dict
                DGLAP result
            stream : (Default: None)
                if given, is written on

        Returns
        -------
            dump :
                result of dump(output, stream), i.e. a string, if no stream is given or
                the Null, if output is written sucessfully to stream
    """
    out = {
        "polynomial_degree": ret["polynomial_degree"],
        "log": ret["log"],
        "operators": {},
        "operator_errors": {}
    }
    # make raw lists - we might want to do somthing more numerical here
    for k in ["xgrid"]:
        out[k] = ret[k].tolist()
    for k in ret["operators"]:
        out["operators"][k] = ret["operators"][k].tolist()
        out["operator_errors"][k] = ret["operator_errors"][k].tolist()
    return dump(out, stream)


def write_YAML_to_file(ret, filename):
    """
        Writes YAML representation to a file.

        Parameters
        ----------
            ret : dict
                DGLAP result
            filename : string
                target file name

        Returns
        -------
            ret :
                result of dump(output, stream), i.e. Null if written sucessfully
    """
    with open(filename, "w") as f:
        ret = get_YAML(ret, f)
    return ret
