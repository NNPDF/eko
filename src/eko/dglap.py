# -*- coding: utf-8 -*-
"""
This file contains the main loop for the DGLAP calculations.

"""
import logging
import joblib
import numpy as np
import numba as nb

from eko import t_float
import eko.alpha_s as alpha_s
import eko.splitting_functions_LO as sf_LO
import eko.interpolation as interpolation
import eko.mellin as mellin
from eko.constants import Constants

logObj = logging.getLogger(__name__)

def _parallelize_on_basis(basis_functions, pfunction, xk, n_jobs = 1):
    out = joblib.Parallel(n_jobs = n_jobs)(
            joblib.delayed(pfunction)(fun, xk)
            for fun in basis_functions
            )
    return out


def _get_xgrid(setup):
    """Compute input grid

    Parameters
    ----------
    setup : dict
        a dictionary with the theory parameters for the evolution

    Returns
    -------
        xgrid : array
            input grid
    """
    xgrid = np.array([])
    # grid type
    xgrid_type = setup.get("xgrid_type", "Chebyshev@log")
    if xgrid_type == "custom": # custom grid
        if "xgrid_custom" not in setup:
            raise ValueError("'xgrid_type' is 'custom', but 'xgrid_custom' is not given")
        xgrid = np.array(setup["xgrid_custom"])
    else: # auto-generated grid
        # read params
        xgrid_size = setup["xgrid_size"]
        xgrid_min = setup.get("xgrid_min", 1e-7)
        # generate
        if xgrid_type == "Chebyshev@log":
            xgrid = interpolation.get_xgrid_Chebyshev_at_log(xgrid_size, xgrid_min)
        elif xgrid_type == "linear@log":
            xgrid = interpolation.get_xgrid_linear_at_log(xgrid_size, xgrid_min)
        else:
            raise ValueError("Unkonwn 'xgrid_type'")
    unique_xgrid = np.unique(xgrid)
    if not len(unique_xgrid) == len(xgrid):
        raise ValueError("given 'xgrid' is not unique!") # relax to warning?
    return unique_xgrid


def _get_evoultion_params(setup):
    """Compute evolution parameters

    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution

    Returns
    -------
        delta_t : t_float
            scale difference
    """
    # setup constants
    nf = setup["NfFF"]
    # setup inital+final scale
    # TODO iterate Q2grid
    qref2 = setup["Qref"] ** 2
    pto = setup["PTO"]
    alphas = setup["alphas"]
    # Generate the alpha_s functions
    a_s = alpha_s.alpha_s_generator(alphas, qref2, nf, "analytic")
    a0 = a_s(pto, setup["Q0"]**2)
    a1 = a_s(pto, setup["Q2grid"][0])
    # evolution parameters
    t0 = np.log(1.0 / a0)
    t1 = np.log(1.0 / a1)
    return t1 - t0

def _run_nonsinglet(setup,constants,delta_t,is_log_interpolation,basis_functions,ret):
    """Solves the non-singlet case.

    This method updates the `ret` parameter instead of returning something.

    Parameters
    ----------
        setup : dict
            a dictionary with the theory parameters for the evolution
        constants : Constants
            used set of constants
        delta_t : t_float
            evolution step
        is_log_interpolation : bool
            use a logarithmic interpolation
        basis_function_coeffs : array
            coefficient list for the basis functions
        ret : dict
            a dictionary for the output
    """
    # setup constants
    xgrid = ret["xgrid"]
    targetgrid = ret["targetgrid"]
    targetgrid_size = len(targetgrid)
    nf = setup["NfFF"]
    beta0 = alpha_s.beta_0(nf, constants.CA, constants.CF, constants.TF)
    CA = constants.CA
    CF = constants.CF

    # prepare
    def get_kernel_ns(basis_function):
        """return non-siglet integration kernel"""

        @nb.njit
        def ker(N, lnx):
            """non-siglet integration kernel"""
            ln = -delta_t * sf_LO.gamma_ns_0(N, nf, CA, CF) / beta0
            interpoln = basis_function(N, lnx)
            return np.exp(ln) * interpoln

        return ker



    # perform
    xgrid_size = len(xgrid)
    op_ns = np.empty( (targetgrid_size, xgrid_size) )
    op_ns_err = np.empty( (targetgrid_size, xgrid_size) )
    #path, jac = mellin.get_path_Talbot()
    logPre = "computing NS operator - %s " # TODO: add a logger prefix

    gamma = 1.0
    cut = 1e-2
    path,jac = mellin.get_path_Cauchy_tan(gamma,1.0)

    # Generate integrands
    integrands = []
    for basis_function in basis_functions:
        ker = get_kernel_ns(basis_function.callable)
        integrands.append(mellin.compile_integrand(ker, path, jac))


    def run_thread(integrand, logx):
        res = mellin.inverse_mellin_transform_simple(integrand, cut, extra_args = logx)
        return res

    logObj.info(logPre, "...")
    operators = []
    operator_errors = []
    for k, xk in enumerate(targetgrid):
        out = _parallelize_on_basis(integrands, run_thread, np.log(xk))
        operators.append(np.array(out)[:,0])
        operator_errors.append(np.array(out)[:,1])
        log_text = f"{k+1}/{targetgrid_size}"
        logObj.info(logPre, log_text)
    logObj.info(logPre, "done.")

    # insert operators
    ret["operators"]["NS"] = np.array(operators)
    ret["operator_errors"]["NS"] = np.array(operator_errors)


def _run_singlet(setup,constants,delta_t,is_log_interpolation,basis_functions,ret):
    """Solves the singlet case.

    This method updates the `ret` parameter instead of returning something.

    Parameters
    ----------
        setup : dict
            a dictionary with the theory parameters for the evolution
        constants : Constants
            used set of constants
        delta_t : t_float
            evolution step
        is_log_interpolation : bool
            use a logarithmic interpolation
        basis_function_coeffs : array
            coefficient list for the basis functions
        ret : dict
            a dictionary for the output
    """
    # setup constants
    xgrid = ret["xgrid"]
    targetgrid = ret["targetgrid"]
    targetgrid_size = len(targetgrid)
    nf = setup["NfFF"]
    beta0 = alpha_s.beta_0(nf, constants.CA, constants.CF, constants.TF)
    CA = constants.CA
    CF = constants.CF

    # prepare
    def get_kernels_s(basis_function):
        """return siglet integration kernels"""

        def get_ker(k,l):

            @nb.njit # TODO here we are repeating too many things!
            def ker(N, lnx):
                """singlet integration kernel"""
                l_p,l_m,e_p,e_m = sf_LO.get_Eigensystem_gamma_singlet_0(N,nf,CA,CF)
                ln_p = - delta_t * l_p  / beta0
                ln_m = - delta_t * l_m  / beta0
                interpoln = basis_function(N, lnx)
                return (e_p[k][l] * np.exp(ln_p) + e_m[k][l] * np.exp(ln_m)) * interpoln
            return ker

        return get_ker(0,0), get_ker(0,1), get_ker(1,0), get_ker(1,1)

    # perform
    xgrid_size = len(xgrid)
    logPre = "computing singlet operator - "
    logObj.info(logPre+"...")

    cut = 1e-2
    gamma = 1.0
    path,jac = mellin.get_path_Cauchy_tan(gamma,1.0)

    integrands = []
    for basis_function in basis_functions:
        kernels = get_kernels_s(basis_function.callable)
        kernel_int = []
        for ker in kernels:
            kernel_int.append(mellin.compile_integrand(
                ker, path, jac))
        integrands.append(kernel_int)

    def run_thread(integrands, logx):
        """ The output of this function is a list of tuple (result, error)
        for qq, qg, gq, gg in that order """
        all_res = []
        for integrand in integrands:
            result = mellin.inverse_mellin_transform_simple(integrand, cut, logx)
            all_res.append(result)
        return all_res


    all_output = []
    for k, xk in enumerate(targetgrid):
        out = _parallelize_on_basis(integrands, run_thread, np.log(xk))
        all_output.append(out)
        logObj.info(logPre+" %d/%d",k+1,targetgrid_size)
    logObj.info(logPre,"done.")

    output_array = np.array(all_output)

    # insert operators
    ret["operators"]["S_qq"] = output_array[:, :, 0, 0]
    ret["operators"]["S_qg"] = output_array[:, :, 1, 0]
    ret["operators"]["S_gq"] = output_array[:, :, 2, 0]
    ret["operators"]["S_gg"] = output_array[:, :, 3, 0]
    ret["operator_errors"]["S_qq"] = output_array[:, :, 0, 1]
    ret["operator_errors"]["S_qg"] = output_array[:, :, 1, 1]
    ret["operator_errors"]["S_gq"] = output_array[:, :, 2, 1]
    ret["operator_errors"]["S_gg"] = output_array[:, :, 3, 1]

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
        'targetgrid'    list of x-values which are computed - defaults to ``xgrid``, if not given
        =============== ==========================================================================

    Returns
    -------
    ret: dict
        a dictionary with a defined set of keys

        ============  ============================================================================
        key           description
        ============  ============================================================================
        'xgrid'       list of x-values which build the support of the interpolation
        'targetgrid'  list of x-values which are computed
        'operators'   list of computed operators
        ============  ============================================================================

    Notes
    -----

    * xgrid_type
        - ``linear@log``: nodes distributed linear in log-space
        - ``custom``: custom xgrid, supplied by the key ``xgrid_custom``

    """

    # print theory id setup
    logObj.info("setup: %s",setup)

    # return dictionay
    # TODO decide on which level to iterate Q2
    ret = {}

    # evolution parameters
    delta_t = _get_evoultion_params(setup)

    # setup input grid: xgrid
    xgrid = _get_xgrid(setup)
    ret["xgrid"] = xgrid
    polynom_rank = setup.get("xgrid_polynom_rank",4)
    is_log_interpolation = not setup.get("xgrid_interpolation","log") == "id"
    basis_function_dispatcher = interpolation.InterpolatorDispatcher(xgrid, polynom_rank, is_log_interpolation)
    logObj.info("is_log_interpolation = %s",is_log_interpolation)

    # setup output grid: targetgrid
    targetgrid = setup.get("targetgrid", xgrid)
    ret["targetgrid"] = targetgrid

    # prepare return of operators
    ret["operators"] = {"NS": None,"S_qq": None,"S_qg": None,"S_gq": None,"S_gg": None}
    ret["operator_errors"] = {"NS": None,"S_qq": None,"S_qg": None,"S_gq": None,"S_gg": None}

    # load constants
    constants = Constants()

    # run non-singlet
    _run_nonsinglet(setup,constants,delta_t,is_log_interpolation,basis_function_dispatcher,ret)

    # run singlet
    _run_singlet(setup,constants,delta_t,is_log_interpolation,basis_function_dispatcher,ret)

    #   Points to be implemented:
    #   TODO implement NLO
    return ret

if __name__ == "__main__":
    n = 3
    xgrid_low = interpolation.get_xgrid_linear_at_log(n,1e-7,0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n,0.1,1.0)
    xgrid_high = np.array([])#1.0-interpolation.get_xgrid_linear_at_log(10,1e-3,1.0 - 0.9)
    xgrid = np.unique(np.concatenate((xgrid_low,xgrid_mid,xgrid_high)))
    polynom_rank = 4
    toy_xgrid = np.array([1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,.1,.3,.5,.7,.9])[-3:]

    ret1 = run_dglap({
        "PTO": 0,
        'alphas': 0.35,
        'Qref': np.sqrt(2),
        'Q0': np.sqrt(2),
        'NfFF': 4,

        "xgrid_type": "custom",
        "xgrid_custom": xgrid,
        "xgrid_polynom_rank": polynom_rank,
        "xgrid_interpolation": "log",
        "targetgrid": toy_xgrid,
        "Q2grid": [1e4]
        })
