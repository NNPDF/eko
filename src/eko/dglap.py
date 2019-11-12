# -*- coding: utf-8 -*-
"""
This file contains the main loop for the DGLAP calculations.

"""
import logging
import numpy as np

from eko import t_float
import eko.alpha_s as alpha_s
import eko.splitting_functions_LO as sf_LO
import eko.interpolation as interpolation
import eko.mellin as mellin
from eko.constants import Constants

logObj = logging.getLogger(__name__)

def _get_xgrid(setup):
    """Compute input grid

    Parameters
    ----------
    setup: dict
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
        t0 : t_float
            initial scale

        t1 : t_float
            final scale
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
    return t0, t1


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
        - ``Chebyshev@log`` (default): nodes distributed along Chebyshev-roots in log-space
        - ``linear@log``: nodes distributed linear in log-space
        - ``custom``: custom xgrid, supplied by the key ``xgrid_custom``

    """

    # print theory id setup
    logObj.info("setup: %s",setup)

    # return dictionay
    # TODO decide on which level to iterate Q2
    ret = {}

    # evolution parameters
    t0, t1 = _get_evoultion_params(setup)

    # setup input grid: xgrid
    xgrid = _get_xgrid(setup)
    ret["xgrid"] = xgrid
    polynom_rank = setup.get("xgrid_polynom_rank",4)
    is_log_interpolation = not setup.get("xgrid_interpolation","log") == "id"
    if is_log_interpolation:
        basis_function_coeffs = interpolation.get_Lagrange_basis_functions_log(xgrid,polynom_rank)
    else:
        basis_function_coeffs = interpolation.get_Lagrange_basis_functions(xgrid,polynom_rank)
    logObj.info("is_log_interpolation = %d",1 if is_log_interpolation else 0)

    # setup output grid: targetgrid
    targetgrid = setup.get("targetgrid", xgrid)
    targetgrid_size = len(targetgrid)
    ret["targetgrid"] = targetgrid

    # prepare return of operators
    ret["operators"] = {"NS": 0}
    ret["operator_errors"] = {"NS": 0}

    # setup constants
    nf = setup["NfFF"]
    constants = Constants()
    beta0 = alpha_s.beta_0(nf, constants.CA, constants.CF, constants.TF)
    delta_t = t1 - t0

    # prepare non-siglet evolution
    def get_kernel_ns(j,lnx):
        """return non-siglet integration kernel"""
        current_coeff = basis_function_coeffs[j]
        if is_log_interpolation:
            fN = interpolation.evaluate_Lagrange_basis_function_log_N
        else:
            fN = interpolation.evaluate_Lagrange_basis_function_N
        def ker(N):
            """non-siglet integration kernel"""
            ln = -delta_t * sf_LO.gamma_ns_0(N, nf, constants.CA, constants.CF) / beta0
            #interpoln = interpolation.get_Lagrange_interpolators_log_N(N, xgrid, j)
            interpoln = fN(N,current_coeff,lnx)
            return np.exp(ln) * interpoln

        return ker

    # perform non-singlet evolution
    xgrid_size = len(xgrid)
    op_ns = np.zeros((targetgrid_size, xgrid_size), dtype=t_float)
    op_ns_err = np.zeros((targetgrid_size, xgrid_size), dtype=t_float)
    #path, jac = mellin.get_path_Talbot()
    logPre = "computing NS operator - "
    logObj.info(logPre+"...")
    for k in range(targetgrid_size):
        xk = targetgrid[k]
        #path,jac = mellin.get_path_line(path_length)
        if False and xk < 1e-3:
            cut = 0.1
            gamma = 2.0
        else:
            cut = 1e-2
            gamma = 1.0
        path,jac = mellin.get_path_Cauchy_tan(gamma,1.0)
        for j in range(xgrid_size):
            res = mellin.inverse_mellin_transform(
                get_kernel_ns(j,np.log(xk)), path, jac, cut
            )
            op_ns[k, j] = res[0]
            op_ns_err[k, j] = res[1]
        logObj.info(logPre+" %d/%d",k+1,targetgrid_size)
    logObj.info(logPre+"done.")

    # insert operators
    ret["operators"]["NS"] = op_ns
    ret["operator_errors"]["NS"] = op_ns_err

    #   Points to be implemented:
    #   TODO implement singlet case
    #   TODO implement NLO
    return ret
