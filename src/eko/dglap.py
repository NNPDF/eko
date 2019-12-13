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

logger = logging.getLogger(__name__)

# evolution basis names
Vs = ["V3", "V8", "V15", "V24", "V35"]
Ts = ["T3", "T8", "T15", "T24", "T35"]


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


def _get_evoultion_params(setup, nf, mu2init, mu2final):
    """Compute evolution parameters

    Parameters
    ----------
    setup: dict
        a dictionary with the theory parameters for the evolution
    nf : int
        number of active flavours
    mu2init : float
        initial scale
    mu2final : flaot
        final scale

    Returns
    -------
        delta_t : t_float
            scale difference
    """
    # setup params
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

    op = np.array(operators)
    op_err = np.array(operator_errors)

    # in LO v=+=-
    ret = {
        "operators": {"NS_+": op.copy(), "NS_-": op.copy(), "NS_v": op.copy()},
        "operator_errors": {
            "NS_+": op_err.copy(),
            "NS_-": op_err.copy(),
            "NS_v": op_err.copy(),
        },
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


# https://stackoverflow.com/a/7205107
# from functools import reduce
# reduce(merge, [dict1, dict2, dict3...])
def _merge_dicts(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def _run_step(
    setup, constants, basis_function_dispatcher, targetgrid, nf, mu2init, mu2final
):
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
    delta_t = _get_evoultion_params(setup, nf, mu2init, mu2final)
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t
    )

    # run non-singlet
    ret_ns = _run_nonsinglet(kernel_dispatcher, targetgrid)
    # run singlet
    ret_s = _run_singlet(kernel_dispatcher, targetgrid)
    # join elements
    ret = _merge_dicts(ret_ns, ret_s)
    return ret


def _run_FFNS(setup, constants, basis_function_dispatcher, targetgrid):
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
    nf = setup["NfFF"]
    # do everything in one simple step
    logger.info(
        "FFNS: nf=%d, evolve [GeV^2] %e -> %e", nf, setup["Q0"] ** 2, setup["Q2grid"][0]
    )
    ret_step = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        targetgrid,
        nf,
        setup["Q0"] ** 2,
        setup["Q2grid"][0],
    )
    # join elements
    ret = {"operators": {}, "operator_errors": {}}

    def set_helper(a, b):
        ret["operators"][a] = ret_step["operators"][b]
        ret["operator_errors"][a] = ret_step["operator_errors"][b]

    # join quarks flavors
    set_helper("V.V", "NS_v")
    for v, t in list(zip(Vs, Ts))[: nf - 1]:  # provide only computations up to nf
        set_helper(f"{v}.{v}", "NS_-")
        set_helper(f"{t}.{t}", "NS_+")
    # Singlet + gluon
    set_helper("S.S", "S_qq")
    set_helper("S.g", "S_qg")
    set_helper("g.S", "S_gq")
    set_helper("g.g", "S_gg")
    return ret

def _run_ZMVFNS_0threshold(
    setup, constants, basis_function_dispatcher, targetgrid, nf
):
    """Run the ZM-VFNS with 0 crossed threshold.

    Parameters
    ----------
    setup : dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    targetgrid : array
        output grid
    nf : int
        number of light flavors, i.e., before the threshold
    Returns
    -------
        ret : dict
            output dictionary
    """
    # setup
    mu2init = setup["Q0"] ** 2
    mu2final = setup["Q2grid"][0]
    # step one
    logger.info(
        "ZM-VFNS: nf=%d, evolve [GeV^2] %e -> %e", nf, mu2init, mu2final
    )
    step = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        targetgrid,
        nf,
        mu2init,
        mu2final,
    )
    # join elements
    ret = {"operators": {}, "operator_errors": {}}

    def set_helper(to, from1):
        ret["operators"][to] = step["operators"][from1]
        ret["operator_errors"][to] = step["operator_errors"][from1]

    # join quarks flavors
    # v.v = V
    set_helper("V.V","NS_v")
    for v, t in list(zip(Vs, Ts))[: nf - 1]: # already there
        set_helper(f"{v}.{v}", "NS_-")
        set_helper(f"{t}.{t}", "NS_+")
    for v, t in list(zip(Vs, Ts))[nf - 1:]: # generate dynamically
        set_helper(f"{v}.V", "NS_v")
        set_helper(f"{t}.S", "S_qq")
        set_helper(f"{t}.S", "S_qg")
    # Singlet + gluon
    set_helper("S.S", "S_qq")
    set_helper("S.g", "S_qg")
    set_helper("g.S", "S_gq")
    set_helper("g.g", "S_gg")

    return ret

def get_singlet_paths(to, fromm, depth):
    """Compute all possible path in the singlet sector to reach `to` starting from  `fromm`.
    
    Parameters
    ----------
        to : 'q' or 'g'
            final point
        fromm : 'q' or 'g'
            starting point
        depth : int
            nesting level; 1 corresponds to the trivial first step

    Returns
    -------
        ls : list
            list of all possible paths
    """
    if depth < 1:
        raise ValueError(f"Invalid arguments: depth >= 1, but got {depth}")
    if to not in ["q","g"]:
        raise ValueError(f"Invalid arguments: to in [q,g], but got {to}")
    if fromm not in ["q","g"]:
        raise ValueError(f"Invalid arguments: fromm in [q,g], but got {fromm}")
    # trivial?
    if depth == 1:
        return [[f"S_{to}{fromm}"]]
    # do recursion
    qs = get_singlet_paths(to,"q",depth - 1)
    for q in qs:
        q.append(f"S_q{fromm}")
    gs = get_singlet_paths(to,"g",depth - 1)
    for g in gs:
        g.append(f"S_g{fromm}")
    return qs + gs


def _run_ZMVFNS_1threshold(
    setup, constants, basis_function_dispatcher, xgrid, targetgrid, m2Threshold, nf_init
):
    """Run the ZM-VFNS with 1 crossed threshold.

    Parameters
    ----------
    setup : dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    xgrid : array
        grid used for intermediate steps
    targetgrid : array
        output grid
    m2Threshold : t_float
        threshold mass that is crossed
    nf_init : int
        number of light flavors, i.e., before the threshold
    Returns
    -------
        ret : dict
            output dictionary
    """
    # setup
    mu2init = setup["Q0"] ** 2
    mu2final = setup["Q2grid"][0]
    # step one
    logger.info(
        "ZM-VFNS: nf=%d, evolve [GeV^2] %e -> %e", nf_init, mu2init, m2Threshold
    )
    step1 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        xgrid,
        nf_init,
        mu2init,
        m2Threshold,
    )
    # step two
    logger.info(
        "ZM-VFNS: nf=%d, evolve [GeV^2] %e -> %e", nf_init + 1, m2Threshold, mu2final
    )
    step2 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        targetgrid,
        nf_init + 1,
        m2Threshold,
        mu2final,
    )
    # join elements
    ret = {"operators": {}, "operator_errors": {}}

    def multiplication_helper(to, from2, from1):
        # force lists
        if not isinstance(from2, list):
            from2l = [from2]
        else:
            from2l = from2
        if not isinstance(from1, list):
            from1l = [from1]
        else:
            from1l = from1
        # iterate
        op = 0
        op_err = 0
        for a, b in zip(from2l, from1l):
            op += np.matmul(step2["operators"][a], step1["operators"][b])
            op_err += np.matmul(
                step2["operator_errors"][a], step1["operators"][b]
            ) + np.matmul(step2["operators"][a], step1["operator_errors"][b])
        ret["operators"][to] = op
        ret["operator_errors"][to] = op_err

    # join quarks flavors
    # v.v = V
    multiplication_helper("V.V", "NS_v", "NS_v")
    # -.-
    for b in Vs[: nf_init - 1]:
        multiplication_helper(f"{b}.{b}", "NS_-", "NS_-")
    # -.v
    b = Vs[nf_init - 1]
    multiplication_helper(f"{b}.V", "NS_-", "NS_v")
    # v.v for higher combinations
    for b in Vs[nf_init:]:
        multiplication_helper(f"{b}.V", "NS_v", "NS_v")
    # +.+
    for b in Ts[: nf_init - 1]:
        multiplication_helper(f"{b}.{b}", "NS_+", "NS_+")
    # +.S
    b = Ts[nf_init - 1]
    multiplication_helper(f"{b}.S", "NS_+", "S_qq")
    multiplication_helper(f"{b}.g", "NS_+", "S_qg")
    # S.S
    for b in Ts[nf_init:]:
        multiplication_helper(f"{b}.S", ["S_qq", "S_qg"], ["S_qq", "S_gq"])
        multiplication_helper(f"{b}.g", ["S_qq", "S_qg"], ["S_qg", "S_gg"])
    # Singlet + gluon
    multiplication_helper("S.S", ["S_qq", "S_qg"], ["S_qq", "S_gq"])
    multiplication_helper("S.g", ["S_qq", "S_qg"], ["S_qg", "S_gg"])
    multiplication_helper("g.S", ["S_gq", "S_gg"], ["S_qq", "S_gq"])
    multiplication_helper("g.g", ["S_gq", "S_gg"], ["S_qg", "S_gg"])
    return ret

def operator_product_helper(rev_steps, paths):
    """Joins all matrix elements given by paths.
    
    Parameters
    ----------
        rev_steps : array
            list of evolution steps in increasing order
        paths : array
            list of all necessary path

    Returns
    -------
        tot_op : array
            joined operator
        tot_op_err : array
            combined error for operator

    """
    # setup
    len_steps = len(rev_steps)
    # collect all paths
    tot_op = 0
    tot_op_err = 0
    for k,e in enumerate(rev_steps):
        print(k,"->",e["operators"].keys())
    for path in paths:
        # init multiplications with a 1
        cur_op = None
        cur_op_err = None
        # check length
        if len(path) != len_steps:
            raise ValueError("Number of steps and number of elements in a path do not match!")
        print("path = ",path)
        # iterate steps
        for k,el in enumerate(path[::-1]):
            print("k,el = ",k,el)
            print(rev_steps[k]["operators"].keys())
            op = rev_steps[k]["operators"][el]
            op_err = rev_steps[k]["operator_errors"][el]
            if cur_op is None:
                cur_op = op
                cur_op_err = op_err
            else:
                old_op = cur_op.copy() # make copy for error determination
                cur_op = np.matmul(op,cur_op)
                cur_op_err = np.matmul(op_err,old_op) + np.matmul(op,cur_op_err)
        # add up
        tot_op += cur_op
        tot_op_err += cur_op_err

    return tot_op, tot_op_err


def _run_ZMVFNS_2thresholds(
    setup, constants, basis_function_dispatcher, xgrid, targetgrid, m2Threshold1, m2Threshold2, nf_init
):
    """Run the ZM-VFNS with 2 crossed threshold.

    Parameters
    ----------
    setup : dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    xgrid : array
        grid used for intermediate steps
    targetgrid : array
        output grid
    m2Threshold1 : t_float
        first threshold mass that is crossed
    m2Threshold2 : t_float
        second threshold mass that is crossed
    nf_init : int
        number of light flavors, i.e., before any threshold
    Returns
    -------
        ret : dict
            output dictionary
    """
    # setup
    mu2init = setup["Q0"] ** 2
    mu2final = setup["Q2grid"][0]
    # step one
    logger.info(
        "ZM-VFNS: nf=%d, evolve [GeV^2] %e -> %e", nf_init, mu2init, m2Threshold1
    )
    step1 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        xgrid,
        nf_init,
        mu2init,
        m2Threshold1,
    )
    # step two
    logger.info(
        "ZM-VFNS: nf=%d, evolve [GeV^2] %e -> %e", nf_init + 1, m2Threshold1, m2Threshold2
    )
    step2 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        targetgrid,
        nf_init + 1,
        m2Threshold1,
        m2Threshold2,
    )
    # step three
    logger.info(
        "ZM-VFNS: nf=%d, evolve [GeV^2] %e -> %e", nf_init + 2, m2Threshold2, mu2final
    )
    step3 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        targetgrid,
        nf_init + 2,
        m2Threshold2,
        mu2final,
    )
    # join elements
    ret = {"operators": {}, "operator_errors": {}}

    def multiplication_helper(to, from3, from2, from1):
        # force lists
        if not isinstance(from3, list):
            from3l = [from3]
        else:
            from3l = from3
        if not isinstance(from2, list):
            from2l = [from2]
        else:
            from2l = from2
        if not isinstance(from1, list):
            from1l = [from1]
        else:
            from1l = from1
        # iterate
        op = 0
        op_err = 0
        for a, b, c in zip(from3l,from2l, from1l):
            # join bc
            op_bc = np.matmul(step2["operators"][b], step1["operators"][c])
            op_err_bc = np.matmul(
                step2["operator_errors"][b], step1["operators"][c]
            ) + np.matmul(step2["operators"][c], step1["operator_errors"][c])
            # rest
            op += np.matmul(step3["operators"][a], op_bc)
            op_err += np.matmul(
                step3["operator_errors"][a], op_bc
            ) + np.matmul(step3["operators"][a], op_err_bc)
        ret["operators"][to] = op
        ret["operator_errors"][to] = op_err

    # join quarks flavors
    # v.v.v = V
    multiplication_helper("V.V", "NS_v", "NS_v", "NS_v")
    # -.-.-
    for v in Vs[:nf_init-1]:
        multiplication_helper(f"{v}.{v}", "NS_-", "NS_-", "NS_-")
    # -.-.v
    b = Vs[nf_init-1]
    multiplication_helper(f"{b}.V", "NS_-", "NS_-", "NS_v")
    # -.v.v
    b = Vs[nf_init]
    multiplication_helper(f"{b}.V", "NS_-", "NS_v", "NS_v")
    # v.v.v for higher combinations
    for b in Vs[nf_init+1:]:
        multiplication_helper(f"{b}.V", "NS_v", "NS_v", "NS_v")
    # +.+.+
    for b in Ts[: nf_init - 1]:
        multiplication_helper(f"{b}.{b}", "NS_+", "NS_+", "NS_+")
    # +.+.S
    b = Ts[nf_init-1]
    multiplication_helper(f"{b}.S", "NS_+", "NS_+", "S_qq")
    # +.S.S
    b = Ts[nf_init]
    multiplication_helper(f"{b}.S", ["NS_+","NS_+"], ["S_qq","S_qg"], ["S_qq","S_gq"])
    # fmt: off
    # S.S.S
    for b in Ts[nf_init+1:]:
        multiplication_helper(f"{b}.S", ["S_qq","S_qg","S_qq","S_qg"],
                                        ["S_qq","S_gq","S_qg","S_gg"],
                                        ["S_qq","S_qq","S_gq","S_gq"])
        multiplication_helper(f"{b}.g", ["S_qq","S_qg","S_qq","S_qg"],
                                        ["S_qg","S_gg","S_qq","S_gq"],
                                        ["S_gg","S_gg","S_qg","S_qg"])

    # Singlet + gluon
    multiplication_helper("S.S", ["S_qq","S_qg","S_qq","S_qg"],
                                 ["S_qq","S_gq","S_qg","S_gg"],
                                 ["S_qq","S_qq","S_gq","S_gq"])
    multiplication_helper("S.g", ["S_qq","S_qg","S_qq","S_qg"],
                                 ["S_qg","S_gg","S_qq","S_gq"],
                                 ["S_gg","S_gg","S_qg","S_qg"])
    multiplication_helper("g.S", ["S_gq","S_gg","S_gq","S_gg"],
                                 ["S_qq","S_gq","S_qg","S_gg"],
                                 ["S_qq","S_qq","S_gq","S_gq"])
    multiplication_helper("g.g", ["S_gq","S_gg","S_gq","S_gg"],
                                 ["S_qg","S_gg","S_qq","S_gq"],
                                 ["S_gg","S_gg","S_qg","S_qg"])
    # fmt: on 
    return ret


def _run_ZM_VFNS(setup, constants, basis_function_dispatcher, xgrid, targetgrid):
    """Run the ZM-VFNS configuration.

    Parameters
    ----------
    setup : dict
        a dictionary with the theory parameters for the evolution
    constants : Constants
        physical constants
    xgrid : array
        grid used for intermediate steps
    targetgrid : array
        output grid

    Returns
    -------
        ret : dict
            output dictionary
    """
    mu2init = setup["Q0"] ** 2
    mu2final = setup["Q2grid"][0]
    # collect HQ masses - add 0 as init
    mH2s = [0]
    Qmc = setup.get("Qmc", 0)
    mH2s.append(Qmc * Qmc)
    Qmb = setup.get("Qmb", 0)
    mH2s.append(Qmb * Qmb)
    Qmt = setup.get("Qmt", 0)
    mH2s.append(Qmt * Qmt)
    # add infinity
    mH2s.append(np.inf)

    # 0 threshold
    for k in range(1, 5):
        if mH2s[k - 1] <= mu2init <= mu2final <= mH2s[k]:
            return _run_ZMVFNS_0threshold(
                setup,
                constants,
                basis_function_dispatcher,
                targetgrid,
                2 + k,
            )

    # 1 threshold
    for k in range(1, 4):
        if mH2s[k - 1] <= mu2init < mH2s[k] <= mu2final < mH2s[k + 1]:
            return _run_ZMVFNS_1threshold(
                setup,
                constants,
                basis_function_dispatcher,
                xgrid,
                targetgrid,
                mH2s[k],
                2 + k,
            )
    raise NotImplementedError("TODO")


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
    logger.info("Interpolation mode: %s", setup["xgrid_type"])
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
        ret_ops = _run_FFNS(setup, constants, basis_function_dispatcher, targetgrid)
    elif FNS == "ZM-VFNS":
        ret_ops = _run_ZM_VFNS(
            setup, constants, basis_function_dispatcher, xgrid, targetgrid
        )
    else:
        raise ValueError(f"Unknown FNS: {FNS}")
    # join operators
    ret = _merge_dicts(ret, ret_ops)
    return ret


def apply_operator(ret, inputs):
    """Apply all available operators to the input PDFs.

    Parameters
    ----------
        ret : dict
            operator definitions - return value of `run_dglap`
        inputs : dict
            input PDFs

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

    return outs, out_errors


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
