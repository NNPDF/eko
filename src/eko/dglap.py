# -*- coding: utf-8 -*-
"""
This file contains the main loop for the DGLAP calculations.

"""
import logging
import joblib
import numpy as np

import eko.interpolation as interpolation
import eko.mellin as mellin
from eko.kernel_generation import KernelDispatcher
from eko.constants import Constants
from eko.alpha_s import Alphas_Dispatcher

logger = logging.getLogger(__name__)


def _parallelize_on_basis(basis_functions, pfunction, xk, n_jobs=1):
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(pfunction)(fun, xk) for fun in basis_functions
    )
    return out


def compute_deltas(alpha_s, q0, q2grid):
    """
    Compute evolution parameters

    Parameters
    ----------
        `alpha_s`: Alphas_Dispatcher
            dispatcher of alpha_s
        `Qref`: t_float
            reference scale for alpha_s
        `Q2grid`: array
            scales to compute

    Return
    ------
        deltas: array
            scale differences from `Q0` to all scales in `Q2grid`
    """
    # Ensure q2grid is an array
    q2grid = np.array(q2grid)

    # Generate the alpha_s values
    alphas_0 = alpha_s(pow(q0, 2))
    alphas_grid = alpha_s(q2grid)

    # Generate the array of evolution parameters
    ti = np.log(1.0 / alphas_0)
    tf = np.log(1.0 / alphas_grid)

    # Return the delta array
    return tf - ti


def compute_operators(kernel_dispatcher, targetgrid, ret, gamma=1.0, cut=1e-2):
    """ Solves the non-singet and the singlet cases """
    # Setup the path
    path, jac = mellin.get_path_Cauchy_tan(gamma, 1.0)

    # Get all precompiled kernels
    kernel_nonsinglet = kernel_dispatcher.compile_nonsinglet()
    kernel_singlet = kernel_dispatcher.compile_singlet()

    # Generate all integrands
    integrands = []
    for ker_ns, kers_s in zip(kernel_nonsinglet, kernel_singlet):
        compiled_kernels = []
        for ker in kers_s:
            comp_ker = mellin.compile_integrand(ker, path, jac)
            compiled_kernels.append(comp_ker)
        compiled_kernels.append(mellin.compile_integrand(ker_ns, path, jac))
        integrands.append(compiled_kernels)

    # Log
    log_prefix = "Computing operators - %s"
    logger.info(log_prefix, "kernels compiled")

    def run_thread(integrands, logx):
        """ The output of this function is a list of tuple (result, error)
        for qq, qg, gq, gg, NS in that order """
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

    ret["operators"]["S_qq"] = output_array[:, :, 0, 0]
    ret["operators"]["S_qg"] = output_array[:, :, 1, 0]
    ret["operators"]["S_gq"] = output_array[:, :, 2, 0]
    ret["operators"]["S_gg"] = output_array[:, :, 3, 0]
    ret["operator_errors"]["S_qq"] = output_array[:, :, 0, 1]
    ret["operator_errors"]["S_qg"] = output_array[:, :, 1, 1]
    ret["operator_errors"]["S_gq"] = output_array[:, :, 2, 1]
    ret["operator_errors"]["S_gg"] = output_array[:, :, 3, 1]
    ret["operators"]["NS"] = output_array[:, :, 4, 0]
    ret["operator_errors"]["NS"] = output_array[:, :, 4, 1]


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

        =================  ============================================================================
        key                description
        =================  ============================================================================
        'xgrid'            list of x-values which build the support of the interpolation
        'targetgrid'       list of x-values which are computed
        'operators'        list of computed operators
        'operator_errors'  list of integration errors associated to the operators
        =================  ============================================================================

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
    nf = setup["NfFF"]
    pto = setup["PTO"]
    alphas = setup["alphas"]
    qref = setup["Qref"]
    alphas_dispatcher = Alphas_Dispatcher(
        constants, alphas, pow(qref, 2), nf, order=pto
    )

    delta_t = compute_deltas(alphas_dispatcher, setup["Q0"], setup["Q2grid"])[0]

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

    # Setup the kernel dispatcher
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t
    )

    compute_operators(kernel_dispatcher, targetgrid, ret)

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
