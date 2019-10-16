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
    xgrid: array
        a list of nodes on which to evaluate the PDF
    kernel: array
        a tensor operator in x-space.
    """
    constants = Constants()

    # print theory id setup
    logObj.info(setup)
    # print constants
    logObj.info(constants)

    # setup constants
    nf = setup["NfFF"]
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
    xgrid = interpolation.get_xgrid_Chebyshev_at_id(xgrid_size)
    # perform non-singlet evolution
    op_ns = np.zeros((xgrid_size,xgrid_size),dtype=t_float)
    op_ns_err = np.zeros((xgrid_size,xgrid_size),dtype=t_float)
    path,jac = Mellin.get_path_Talbot()
    for j in range(xgrid_size):
        for k in range(xgrid_size):
            res = Mellin.inverse_Mellin_transform(
                lambda N,t1=t1,t0=t0,
                       g_ns_0=sf_LO.gamma_ns_0,nf=nf,constants=constants,beta0=beta0,
                       pN=interpolation.get_Lagrange_iterpolators_N,j=j,xgrid=xgrid :
                           np.exp(
                            (t1-t0) * g_ns_0(N,nf,constants.CF,constants.CF) / beta0
                           ) * pN(N,xgrid,j),
                path,jac,xgrid[k],1e-2
            )
            op_ns[j,k] = res[0]
            op_ns_err[j,k] = res[1]

#   Points to be implemented:
#   TODO allocate splittings, running
#   TODO solve DGLAP in N-space
#   TODO perform Mellin inverse
#   TODO return the kernel operator in x-space
    return (xgrid,op_ns,op_ns_err)

if __name__ == "__main__":
    print("### TEST ###")
    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.flush = sys.stdout.flush
    ch.setLevel(0)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    xgrid,op_ns,op_ns_err = run_dglap({
        "PTO": 0,
        'alphas': 0.118,
        'Qref': 91.2,
        'Q0': 1.65,
        'NfFF': 5,

        "xgrid_size": 3,
        "Q2grid": [10.]
    })
    print("xgrid = ",xgrid)
    print("op_ns = ",op_ns)
    print("op_ns_err = ",op_ns_err)
    ch.close()
