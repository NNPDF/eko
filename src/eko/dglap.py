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
import eko.Mellin as Mellin
from eko.constants import Constants

logObj = logging.getLogger(__name__)

def run_dglap(setup):
    """This function takes a DGLAP theory configuration dictionary
    and performs the solution of the DGLAP equations.

    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the DGLAP

    Returns
    -------
    ret: dict
        a dictionary with a defined set of keys
    """

    # print theory id setup
    logObj.info(setup)

    # return dictionay
    ret = {}

    # setup constants
    nf = setup["NfFF"]
    constants = Constants()
    beta0 = alpha_s.beta_0(nf, constants.CA, constants.CF, constants.TF)
    # setup inital+final scale
    a0 = alpha_s.a_s(setup["PTO"],setup["alphas"],
                     setup["Qref"]**2,setup["Q0"]**2,nf,"analytic")
    a1 = alpha_s.a_s(setup["PTO"],setup["alphas"],
                     setup["Qref"]**2,setup["Q2grid"][0],nf,"analytic")
    # evolution parameters
    t0 = np.log(1./a0)
    t1 = np.log(1./a1)
    # setup grid
    xgrid_size = setup["xgrid_size"]
    xgrid = interpolation.get_xgrid_Chebyshev_at_log10(xgrid_size,1e-7)
    ret["xgrid"] = xgrid
    targetgrid = xgrid if not "targetgrid" in setup else setup["targetgrid"]
    targetgrid_size = len(targetgrid)

    # prepare return of operators
    ret["operators"] = {"NS": 0}
    ret["operator_errors"] = {"NS": 0}

    # prepare non-siglet evolution
    def get_kernel_ns(j,t1=t1,t0=t0,
                       g_ns_0=sf_LO.gamma_ns_0,nf=nf,constants=constants,beta0=beta0,
                       pN=interpolation.get_Lagrange_iterpolators_log_N,xgrid=xgrid):
        """return non-siglet integration kernel"""
        delta_t = t1 - t0
        def ker(N):
            """non-siglet integration kernel"""
            ln = - delta_t * g_ns_0(N,nf,constants.CF,constants.CF) / beta0
            return np.exp(ln) * pN(N,xgrid,j)
        return ker
    # perform non-singlet evolution
    op_ns = np.zeros((targetgrid_size,xgrid_size),dtype=t_float)
    op_ns_err = np.zeros((targetgrid_size,xgrid_size),dtype=t_float)
    path,jac = Mellin.get_path_Talbot()
    for j in range(xgrid_size):
        for k in range(targetgrid_size):
            res = Mellin.inverse_Mellin_transform(get_kernel_ns(j),path,jac,targetgrid[k],1e-2)
            op_ns[k,j] = res[0]
            op_ns_err[k,j] = res[1]
    # insert operators
    ret["operators"]["NS"] = op_ns
    ret["operator_errors"]["NS"] = op_ns_err

#   Points to be implemented:
#   TODO implement siglet case
#   TODO implement NLO
    return ret
