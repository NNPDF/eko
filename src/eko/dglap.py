# -*- coding: utf-8 -*-
"""
This file contains the main loop for the DGLAP calculations.

"""
import logging
import joblib
import numpy as np

import eko.alpha_s as alpha_s
import eko.interpolation as interpolation
import eko.mellin as mellin
from eko.kernel_generation import KernelDispatcher
from eko.constants import Constants

# List of all evolution basis elements
evolution_basis_label_list = ["V","T3","V3","T8","V8","T15","V15","T24","V24","T35","V35","S_qq","S_qg","S_gq","S_gg","g"]

logger = logging.getLogger(__name__)

def _parallelize_on_basis(basis_functions, pfunction, xk, n_jobs=1):
    """Provide parallization over all basis functions
    
    Parameters
    ----------
        basis_functions : list
            basis list
        pfunction : function
            executed function
        xk : t_float
            Mellin inversion point
        n_jobs : int
            number of parallel jobs
        
    Returns
    -------
        out : list
            output list for all jobs
    """
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(pfunction)(fun, xk) for fun in basis_functions
    )
    return out


def _get_evoultion_params(setup, mu2init, mu2final):
    """Compute evolution parameters

    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution
    mu2init : float
        initial scale
    mu2final : flaot
        final scale

    Returns
    -------
        delta_t : t_float
            scale difference
    """
    # setup constants
    nf = setup["NfFF"]
    # setup inital+final scale
    qref2 = setup["Qref"] ** 2
    pto = setup["PTO"]
    alphas = setup["alphas"]
    # Generate the alpha_s functions
    a_s = alpha_s.alpha_s_generator(alphas, qref2, nf, "analytic")
    a0 = a_s(pto, mu2init)
    a1 = a_s(pto, mu2final)
    # evolution parameters
    t0 = np.log(1.0 / a0)
    t1 = np.log(1.0 / a1)
    return t1 - t0


def _run_nonsinglet(kernel_dispatcher, targetgrid):
    """Solves the non-singlet case.

    Parameters
    ----------
        kernel_dispatcher: KernelDispatcher
            instance of kerneldispatcher from which compiled kernels can be obtained
        targetgrid: array
            list of x-values which are computed

    Returns
    -------
        ret : dict
            dictionary containing the keys `operators` and `operator_errors`
    """
    # Receive all precompiled kernels
    kernels = kernel_dispatcher.compile_nonsinglet()

    # Setup path
    gamma = 1.0
    cut = 1e-2
    path, jac = mellin.get_path_Cauchy_tan(gamma, 1.0)

    # Generate integrands
    integrands = []
    for kernel in kernels:
        kernel_int = mellin.compile_integrand(kernel, path, jac)
        integrands.append(kernel_int)

    def run_thread(integrand, logx):
        result = mellin.inverse_mellin_transform(integrand, cut, logx)
        return result

    log_prefix = "computing NS operator - %s"
    logger.info(log_prefix, "kernel compiled")
    operators = []
    operator_errors = []

    targetgrid_size = len(targetgrid)
    for k, xk in enumerate(targetgrid):
        out = _parallelize_on_basis(integrands, run_thread, np.log(xk))
        operators.append(np.array(out)[:, 0])
        operator_errors.append(np.array(out)[:, 1])
        log_text = f"{k+1}/{targetgrid_size}"
        logger.info(log_prefix, log_text)
    logger.info(log_prefix, "done.")

    ret = {
        "operators" : {
            "NS" : np.array(operators)
        },
        "operator_errors" : {
            "NS" : np.array(operator_errors)
        }
    }
    return ret


def _run_singlet(kernel_dispatcher, targetgrid):
    """Solves the singlet case.

    Parameters
    ----------
        kernel_dispatcher: KernelDispatcher
            instance of kerneldispatcher from which compiled kernels can be obtained
        targetgrid: array
            list of x-values which are computed

    Returns
    -------
        ret : dict
            dictionary containing the keys `operators` and `operator_errors`
    """
    # Receive all precompiled kernels
    kernels = kernel_dispatcher.compile_singlet()

    # Setup path
    cut = 1e-2
    gamma = 1.0
    path, jac = mellin.get_path_Cauchy_tan(gamma, 1.0)

    # Generate integrands
    integrands = []
    for kernel_set in kernels:
        kernel_int = []
        for ker in kernel_set:
            kernel_int.append(mellin.compile_integrand(ker, path, jac))
        integrands.append(kernel_int)

    # perform
    log_prefix = "computing singlet operator - %s"
    logger.info(log_prefix, "kernel compiled")

    def run_thread(integrands, logx):
        """ The output of this function is a list of tuple (result, error)
        for qq, qg, gq, gg in that order """
        all_res = []
        for integrand in integrands:
            result = mellin.inverse_mellin_transform(integrand, cut, logx)
            all_res.append(result)
        return all_res

    all_output = []
    targetgrid_size = len(targetgrid)
    for k, xk in enumerate(targetgrid):
        out = _parallelize_on_basis(integrands, run_thread, np.log(xk))
        all_output.append(out)
        log_text = f"{k+1}/{targetgrid_size}"
        logger.info(log_prefix, log_text)
    logger.info(log_prefix, "done.")

    output_array = np.array(all_output)

    # insert operators
    ret = {
        "operators" : {
        },
        "operator_errors" : {
        }
    }
    ret["operators"]["S_qq"] = output_array[:, :, 0, 0]
    ret["operators"]["S_qg"] = output_array[:, :, 1, 0]
    ret["operators"]["S_gq"] = output_array[:, :, 2, 0]
    ret["operators"]["S_gg"] = output_array[:, :, 3, 0]
    ret["operator_errors"]["S_qq"] = output_array[:, :, 0, 1]
    ret["operator_errors"]["S_qg"] = output_array[:, :, 1, 1]
    ret["operator_errors"]["S_gq"] = output_array[:, :, 2, 1]
    ret["operator_errors"]["S_gg"] = output_array[:, :, 3, 1]

    return ret

# https://stackoverflow.com/a/7205107
# from functools import reduce
# reduce(merge, [dict1, dict2, dict3...])
def _merge_dicts(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def _run_step(setup,constants,basis_function_dispatcher,targetgrid,nf,mu2init,mu2final):
    """Do a single convolution step in a fixed parameter configuration
    
    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    targetgrid : array
        output grid
    nf : int
        number of active flavours
    mu2init : float
        initial scale
    mu2final : flaot
        final scale

    Returns
    -------
        ret : dict
            output dictionary
    """
    # Setup the kernel dispatcher
    delta_t = _get_evoultion_params(setup, mu2init, mu2final)
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t
    )

    # run non-singlet
    raw_ret_ns = _run_nonsinglet(kernel_dispatcher, targetgrid)
    op_ns = raw_ret_ns["operators"]["NS"]
    op_err_ns = raw_ret_ns["operator_errors"]["NS"]
    ret_ns = {"operators": {}, "operator_errors": {}}
    for label in ["V","T3","V3","T8","V8","T15","V15","T24","V24","T35","V35"]:
        ret_ns["operators"][label] = op_ns.copy()
        ret_ns["operator_errors"][label] = op_err_ns.copy()

    # run singlet
    ret_s = _run_singlet(kernel_dispatcher, targetgrid)
    # join elements
    ret = _merge_dicts(ret_ns,ret_s)
    return ret

def _run_FFNS(setup,constants,basis_function_dispatcher,targetgrid):
    """Run the FFNS configuration.
    
    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    targetgrid : array
        output grid

    Returns
    -------
        ret : dict
            output dictionary
    """
    # do everything in one simple step
    ret = _run_step(setup,constants,basis_function_dispatcher,targetgrid,setup["NfFF"],setup["Q0"] ** 2, setup["Q2grid"][0])
    # TODO drop again all keys < n_f?
    return ret


def _run_ZM_VFNS(setup,constants,basis_function_dispatcher,xgrid,targetgrid):
    """Run the ZM-VFNS configuration.
    
    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    targetgrid : array
        output grid

    Returns
    -------
        ret : dict
            output dictionary
    """
    mu2init = setup["Q0"] ** 2
    mu2final = setup["Q2grid"][0]
    # collect HQ masses
    mH2s = []
    Qmc = setup.get("Qmc",None)
    mH2s.append(Qmc*Qmc)
    Qmb = setup.get("Qmb",None)
    mH2s.append(Qmb*Qmb)
    Qmt = setup.get("Qmt",None)
    mH2s.append(Qmt*Qmt)
    # prepare output
    targetgrid_size = len(targetgrid)
    identity = np.identity((targetgrid_size,targetgrid_size))
    void = np.zeros((targetgrid_size,targetgrid_size))
    ret = {"operators":{},"operator_errors": {}}
    for label in evolution_basis_label_list:
        ret["operators"][label] = identity.copy()
        ret["operator_erros"][label] = void.copy()
    mu2low = mu2init
    # mu -> mc
    if (mu2final > mH2s[0]):
        ret_step = _run_step(setup,constants,basis_function_dispatcher,targetgrid,3,mu2low, mH2s[0])
        # update all elements
        for label in evolution_basis_label_list:
            old_op = ret["operators"][label]
            new_op = ret_step["operators"][label]
            ret["operators"][label] = np.dot(old_op,new_op)
        mu2low = mH2s[0]
    # TODO drop again all keys < n_f?
    return ret


def run_dglap(setup):
    r"""This function takes a DGLAP theory configuration dictionary
    and performs the solution of the DGLAP equations.

    The EKO :math:`\hat O_{k,j}^{(0)}(t_1,t_0)` is determined in order
    to fullfill the following evolution

    .. math::
        f^{(0)}(x_k,t_1) = \hat O_{k,j}^{(0)}(t_1,t_0) f^{(0)}(x_j,t_0)

    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution

        =============== ==========================================================================
        key             description
        =============== ==========================================================================
        'PTO'           order of perturbation theory: ``0`` = LO, ...
        'alphas'        reference value of the strong coupling :math:`\alpha_s(\mu_0^2)`
        'xgrid_size'    size of the interpolation grid
        'xgrid_min'     lower boundry of the interpolation grid - defaults to ``1e-7``
        'xgrid_type'    generating function for the interpolation grid - see below
        'log_interpol'  boolean, whether it is log interpolation or not, defaults to `True`
        'targetgrid'    list of x-values which are computed - defaults to ``xgrid``, if not given
        =============== ==========================================================================

    Returns
    -------
    ret: dict
        a dictionary with a defined set of keys

        =================  =========================================================================
        key                description
        =================  =========================================================================
        'xgrid'            list of x-values which build the support of the interpolation
        'targetgrid'       list of x-values which are computed
        'operators'        list of computed operators
        'operator_errors'  list of integration errors associated to the operators
        =================  =========================================================================

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
    is_log_interpolation = setup.get("log_interpol", True)
    polynom_rank = setup.get("xgrid_polynom_rank", 4)
    logger.info(
        "Interpolation mode: %s", setup["xgrid_type"],
    )
    logger.info("Log interpolation: %s", is_log_interpolation)
    targetgrid = setup.get("targetgrid", xgrid)
    basis_function_dispatcher = interpolation.InterpolatorDispatcher(
        xgrid, polynom_rank, log=is_log_interpolation
    )

    # Start filling the output dictionary
    ret = {
        "xgrid": xgrid,
        "targetgrid": targetgrid,
        "operators": {},
        "operator_errors": {},
    }

    # check FNS and split
    FNS = setup["FNS"]
    if FNS == "FFNS":
        ret_ops = _run_FFNS(setup,constants,basis_function_dispatcher,targetgrid)
    elif FNS == "ZM-VFNS":
        ret_ops = _run_ZM_VFNS(setup,constants,basis_function_dispatcher,xgrid,targetgrid)
    else:
        raise ValueError(f"Unknown FNS: {FNS}")
    # join operators
    ret = _merge_dicts(ret,ret_ops)
    return ret


if __name__ == "__main__":
    n = 3
    xgrid_low = interpolation.get_xgrid_linear_at_log(n, 1e-7, 0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n, 0.1, 1.0)
    xgrid_high = np.array(
        []
    )  # 1.0-interpolation.get_xgrid_linear_at_log(10,1e-3,1.0 - 0.9)
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high)))
    polynom_rank = 4
    toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])[
        -3:
    ]

    ret1 = run_dglap(
        {
            "PTO": 0,
            "alphas": 0.35,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "NfFF": 4,
            "xgrid_type": "custom",
            "xgrid": xgrid,
            "xgrid_polynom_rank": polynom_rank,
            "xgrid_interpolation": "log",
            "targetgrid": toy_xgrid,
            "Q2grid": [1e4],
        }
    )
