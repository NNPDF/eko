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
    # read params
    xgrid_size = setup["xgrid_size"]
    xgrid_min = 1e-7 if not "xgrid_min" in setup else setup["xgrid_min"]
    xgrid_type = "Chebyshev@log" if not "xgrid_type" in setup else setup["xgrid_type"]
    # generate
    xgrid = np.array([])
    if xgrid_type == "Chebyshev@log":
        xgrid = interpolation.get_xgrid_Chebyshev_at_log(xgrid_size,xgrid_min)
    elif xgrid_type == "linear@log":
        xgrid = interpolation.get_xgrid_linear_at_log(xgrid_size,xgrid_min)
    else:
        raise ValueError("Unkonwn 'xgrid_type'")
    return xgrid

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
    a0 = alpha_s.a_s(setup["PTO"],setup["alphas"],
                     setup["Qref"]**2,setup["Q0"]**2,nf,"analytic")
    a1 = alpha_s.a_s(setup["PTO"],setup["alphas"],
                     setup["Qref"]**2,setup["Q2grid"][0],nf,"analytic")
    # evolution parameters
    t0 = np.log(1./a0)
    t1 = np.log(1./a1)
    return t0,t1

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
        - ``Chebyshev@log`` (default)
        - ``linear@log``

    """

    # print theory id setup
    logObj.info(setup)

    # return dictionay
    # TODO decide on which level to iterate Q2
    ret = {}

    # evolution parameters
    t0,t1 = _get_evoultion_params(setup)

    # setup input grid: xgrid
    xgrid =_get_xgrid(setup)
    ret["xgrid"] = xgrid

    # setup output grid: targetgrid
    targetgrid = xgrid if not "targetgrid" in setup else setup["targetgrid"]
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
    def get_kernel_ns(j,delta_t=delta_t,
                       g_ns_0=sf_LO.gamma_ns_0,nf=nf,constants=constants,beta0=beta0,
                       pN=interpolation.get_Lagrange_interpolators_log_N,xgrid=xgrid):
        """return non-siglet integration kernel"""
        def ker(N):
            """non-siglet integration kernel"""
            ln = - delta_t * g_ns_0(N,nf,constants.CA,constants.CF) / beta0
            return np.exp(ln) * pN(N,xgrid,j)
        return ker
    # perform non-singlet evolution
    xgrid_size = len(xgrid)
    op_ns = np.zeros((targetgrid_size,xgrid_size),dtype=t_float)
    op_ns_err = np.zeros((targetgrid_size,xgrid_size),dtype=t_float)
    path,jac = mellin.get_path_Talbot()
    for j in range(xgrid_size):
        for k in range(targetgrid_size):
            res = mellin.inverse_mellin_transform(get_kernel_ns(j),path,jac,targetgrid[k],1e-2)
            op_ns[k,j] = res[0]
            op_ns_err[k,j] = res[1]
    # insert operators
    ret["operators"]["NS"] = op_ns
    ret["operator_errors"]["NS"] = op_ns_err

#   Points to be implemented:
#   TODO implement singlet case
#   TODO implement NLO
    return ret
