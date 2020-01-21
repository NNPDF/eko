# -*- coding: utf-8 -*-
"""
    This file contains the main loop for the DGLAP calculations.
"""
import logging
import copy
import joblib
import numpy as np
from yaml import dump

import eko.alpha_s as alpha_s
import eko.interpolation as interpolation
import eko.mellin as mellin
import eko.utils as utils
from eko.kernel_generation import KernelDispatcher
from eko.constants import Constants

logger = logging.getLogger(__name__)

# evolution basis names
Vs = ["V3", "V8", "V15", "V24", "V35"]
Ts = ["T3", "T8", "T15", "T24", "T35"]


def _parallelize_on_basis(basis_functions, pfunction, xk, n_jobs=1):
    """
        Provide parallization over all basis functions

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


def _run_nonsinglet(kernel_dispatcher, targetgrid):
    """
        Solves the non-singlet case.

        Parameters
        ----------
            kernel_dispatcher: KernelDispatcher
                instance of KernelDispatcher from which compiled kernels can be obtained
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
    cut = 1e-2
    path, jac = mellin.get_path_Talbot(1.0)

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
        "operators": {"NS_p": op.copy(), "NS_m": op.copy(), "NS_v": op.copy()},
        "operator_errors": {
            "NS_p": op_err.copy(),
            "NS_m": op_err.copy(),
            "NS_v": op_err.copy(),
        },
    }
    return ret


def _run_singlet(kernel_dispatcher, targetgrid):
    """
        Solves the singlet case.

        Parameters
        ----------
            kernel_dispatcher: KernelDispatcher
                instance of KernelDispatcher from which compiled kernels can be obtained
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
    path, jac = mellin.get_path_Talbot(1.0, 1.0)

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


def _run_step(
    setup, constants, basis_function_dispatcher, targetgrid, nf, mu2init, mu2final
):
    """
        Do a single convolution step in a fixed parameter configuration

        Parameters
        ----------
            setup: dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
            targetgrid : array
                output grid
            nf : int
                number of active flavours
            mu2init : float
                initial scale
            mu2final : float
                final scale

        Returns
        -------
            ret : dict
                output dictionary
    """
    logger.info("evolve [GeV^2] %e -> %e with nf=%d flavors", mu2init, mu2final, nf)
    # Setup the kernel dispatcher
    delta_t = alpha_s.get_evolution_params(setup, constants, nf, mu2init, mu2final)
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t
    )

    # run non-singlet
    ret_ns = _run_nonsinglet(kernel_dispatcher, targetgrid)
    # run singlet
    ret_s = _run_singlet(kernel_dispatcher, targetgrid)
    # join elements
    ret = utils.merge_dicts(ret_ns, ret_s)
    return ret


def _run_FFNS(setup, constants, basis_function_dispatcher, targetgrid):
    """
        Run the FFNS configuration.

        Parameters
        ----------
            setup: dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
            targetgrid : array
                output grid

        Returns
        -------
            ret : dict
                output dictionary
    """
    nf = setup["NfFF"]
    # do everything in one simple step
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
        set_helper(f"{v}.{v}", "NS_m")
        set_helper(f"{t}.{t}", "NS_p")
    # Singlet + gluon
    set_helper("S.S", "S_qq")
    set_helper("S.g", "S_qg")
    set_helper("g.S", "S_gq")
    set_helper("g.g", "S_gg")
    return ret


def _run_ZMVFNS_0threshold(setup, constants, basis_function_dispatcher, targetgrid, nf):
    """
        Run the ZM-VFNS with 0 crossed threshold.

        Parameters
        ----------
        setup : dict
            a dictionary with the theory parameters for the evolution
        constants : Constants
            physical constants
        basis_function_dispatcher : InterpolatorDispatcher
            basis functions
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
    # step one: mu^2_init -> mu^2_final
    step = _run_step(
        setup, constants, basis_function_dispatcher, targetgrid, nf, mu2init, mu2final
    )
    # join elements
    ret = {"operators": {}, "operator_errors": {}}

    def set_helper(to, from1):
        ret["operators"][to] = step["operators"][from1]
        ret["operator_errors"][to] = step["operator_errors"][from1]

    # join quarks flavors
    # v.v = V
    set_helper("V.V", "NS_v")
    for v, t in list(zip(Vs, Ts))[: nf - 1]:  # already there
        set_helper(f"{v}.{v}", "NS_m")
        set_helper(f"{t}.{t}", "NS_p")
    for v, t in list(zip(Vs, Ts))[nf - 1 :]:  # generate dynamically
        set_helper(f"{v}.V", "NS_v")
        set_helper(f"{t}.S", "S_qq")
        set_helper(f"{t}.S", "S_qg")
    # Singlet + gluon
    set_helper("S.S", "S_qq")
    set_helper("S.g", "S_qg")
    set_helper("g.S", "S_gq")
    set_helper("g.g", "S_gg")

    return ret


def _run_ZMVFNS_1threshold(
    setup, constants, basis_function_dispatcher, xgrid, targetgrid, m2Threshold, nf_init
):
    """
        Run the ZM-VFNS with 1 crossed threshold.

        Parameters
        ----------
            setup : dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
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
    # step one: mu^2_init -> m^2_q
    step1 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        xgrid,
        nf_init,
        mu2init,
        m2Threshold,
    )
    # step two: m^2_q -> mu^2_final
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

    # supply short wrapper
    def set_helper(to, paths):
        op, op_err = utils.operator_product_helper([step2, step1], paths)
        ret["operators"][to] = op
        ret["operator_errors"][to] = op_err

    # join quarks flavors
    # v.v = V
    set_helper("V.V", [["NS_v", "NS_v"]])
    # -.-
    for b in Vs[: nf_init - 1]:
        set_helper(f"{b}.{b}", [["NS_m", "NS_m"]])
    # -.v
    b = Vs[nf_init - 1]
    set_helper(f"{b}.V", [["NS_m", "NS_v"]])
    # v.v for higher combinations
    for b in Vs[nf_init:]:
        set_helper(f"{b}.V", [["NS_v", "NS_v"]])
    # +.+
    for b in Ts[: nf_init - 1]:
        set_helper(f"{b}.{b}", [["NS_p", "NS_p"]])
    # +.S
    b = Ts[nf_init - 1]
    set_helper(f"{b}.S", [["NS_p", "S_qq"]])
    set_helper(f"{b}.g", [["NS_p", "S_qg"]])
    # S.S
    paths_qq = utils.get_singlet_paths("q", "q", 2)
    paths_qg = utils.get_singlet_paths("q", "g", 2)
    for b in Ts[nf_init:]:
        set_helper(f"{b}.S", paths_qq)
        set_helper(f"{b}.g", paths_qg)

    # Singlet + gluon
    set_helper("S.S", paths_qq)
    set_helper("S.g", paths_qg)
    set_helper("g.S", utils.get_singlet_paths("g", "q", 2))
    set_helper("g.g", utils.get_singlet_paths("g", "g", 2))
    return ret


def _run_ZMVFNS_2thresholds(
    setup,
    constants,
    basis_function_dispatcher,
    xgrid,
    targetgrid,
    m2Threshold1,
    m2Threshold2,
    nf_init,
):
    """
        Run the ZM-VFNS with 2 crossed threshold.

        Parameters
        ----------
            setup : dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
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
    # step one: mu^2_init -> m^2_q1
    step1 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        xgrid,
        nf_init,
        mu2init,
        m2Threshold1,
    )
    # step two: m^2_q1 -> m^2_q2
    step2 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        targetgrid,
        nf_init + 1,
        m2Threshold1,
        m2Threshold2,
    )
    # step three: m^2_q2 -> mu^2_final
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

    # supply short wrapper
    def set_helper(to, paths):
        op, op_err = utils.operator_product_helper([step3, step2, step1], paths)
        ret["operators"][to] = op
        ret["operator_errors"][to] = op_err

    # join quarks flavors
    # v.v.v = V
    set_helper("V.V", [["NS_v", "NS_v", "NS_v"]])
    # -.-.-
    for v in Vs[: nf_init - 1]:
        set_helper(f"{v}.{v}", [["NS_m", "NS_m", "NS_m"]])
    # -.-.v
    b = Vs[nf_init - 1]
    set_helper(f"{b}.V", [["NS_m", "NS_m", "NS_v"]])
    # -.v.v
    b = Vs[nf_init]
    set_helper(f"{b}.V", [["NS_m", "NS_v", "NS_v"]])
    # v.v.v for higher combinations
    for b in Vs[nf_init + 1 :]:
        set_helper(f"{b}.V", [["NS_v", "NS_v", "NS_v"]])
    # +.+.+
    for b in Ts[: nf_init - 1]:
        set_helper(f"{b}.{b}", [["NS_p", "NS_p", "NS_p"]])
    # +.+.S
    b = Ts[nf_init - 1]
    set_helper(f"{b}.S", [["NS_p", "NS_p", "S_qq"]])
    # +.S.S
    b = Ts[nf_init]
    paths_qq_2 = utils.get_singlet_paths("q", "q", 2)
    for p in paths_qq_2:
        p.insert(0, "NS_p")
    set_helper(f"{b}.S", paths_qq_2)
    # S.S.S
    paths_qq_3 = utils.get_singlet_paths("q", "q", 3)
    paths_qg_3 = utils.get_singlet_paths("q", "g", 3)
    for b in Ts[nf_init + 1 :]:
        set_helper(f"{b}.S", paths_qq_3)
        set_helper(f"{b}.g", paths_qg_3)

    # Singlet + gluon
    set_helper("S.S", paths_qq_3)
    set_helper("S.g", paths_qg_3)
    set_helper("g.S", utils.get_singlet_paths("g", "q", 3))
    set_helper("g.g", utils.get_singlet_paths("g", "g", 3))
    return ret


def _run_ZMVFNS_3thresholds(
    setup, constants, basis_function_dispatcher, xgrid, targetgrid, m2c, m2b, m2t
):
    """
        Run the ZM-VFNS with 3 crossed threshold.

        Assumes nf_init = 3.

        Parameters
        ----------
            setup : dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
            xgrid : array
                grid used for intermediate steps
            targetgrid : array
                output grid
            m2c : t_float
                first threshold mass that is crossed = charm mass
            m2b : t_float
                second threshold mass that is crossed = bottom mass
            m2t : t_float
                third threshold mass that is crossed = top mass

        Returns
        -------
            ret : dict
                output dictionary
    """
    # setup
    mu2init = setup["Q0"] ** 2
    mu2final = setup["Q2grid"][0]
    # step one: mu^2_init -> m^2_c
    step1 = _run_step(
        setup, constants, basis_function_dispatcher, xgrid, 3, mu2init, m2c
    )
    # step two: m^2_c -> m^2_b
    step2 = _run_step(
        setup, constants, basis_function_dispatcher, targetgrid, 4, m2c, m2b
    )
    # step three: m^2_b -> m^2_t
    step3 = _run_step(
        setup, constants, basis_function_dispatcher, targetgrid, 5, m2b, m2t
    )
    # step four: m^2_t -> mu^2_final
    step4 = _run_step(
        setup, constants, basis_function_dispatcher, targetgrid, 6, m2t, mu2final
    )
    # join elements
    ret = {"operators": {}, "operator_errors": {}}

    # supply short wrapper
    def set_helper(to, paths):
        op, op_err = utils.operator_product_helper([step4, step3, step2, step1], paths)
        ret["operators"][to] = op
        ret["operator_errors"][to] = op_err

    # join quarks flavors
    # v.v.v.v = V
    set_helper("V.V", [["NS_v", "NS_v", "NS_v", "NS_v"]])
    # -.-.-.- = V3,V8
    for v in Vs[:2]:
        set_helper(f"{v}.{v}", [["NS_m", "NS_m", "NS_m", "NS_m"]])
    # -.-.-.v = V15
    b = Vs[3]
    set_helper(f"{b}.V", [["NS_m", "NS_m", "NS_m", "NS_v"]])
    # -.-.v.v = V24
    b = Vs[4]
    set_helper(f"{b}.V", [["NS_m", "NS_m", "NS_v", "NS_v"]])
    # -.v.v.v = V35
    b = Vs[5]
    set_helper(f"{b}.V", [["NS_m", "NS_v", "NS_v", "NS_v"]])
    # +.+.+.+ = T3,T8
    for b in Ts[:2]:
        set_helper(f"{b}.{b}", [["NS_p", "NS_p", "NS_p", "NS_p"]])
    # +.+.+.S = T15
    b = Ts[2]
    set_helper(f"{b}.S", [["NS_p", "NS_p", "NS_p", "S_qq"]])
    # +.+.S.S = T24
    b = Ts[3]
    paths_qq_2 = utils.get_singlet_paths("q", "q", 2)
    for p in paths_qq_2:
        p.insert(0, "NS_p")
        p.insert(0, "NS_p")
    set_helper(f"{b}.S", paths_qq_2)
    # +.S.S.S = T35
    b = Ts[4]
    paths_qq_3 = utils.get_singlet_paths("q", "q", 3)
    for p in paths_qq_3:
        p.insert(0, "NS_p")
    set_helper(f"{b}.S", paths_qq_3)

    # Singlet + gluon
    set_helper("S.S", utils.get_singlet_paths("q", "q", 4))
    set_helper("S.g", utils.get_singlet_paths("q", "g", 4))
    set_helper("g.S", utils.get_singlet_paths("g", "q", 4))
    set_helper("g.g", utils.get_singlet_paths("g", "g", 4))
    return ret


def _run_ZM_VFNS(setup, constants, basis_function_dispatcher, xgrid, targetgrid):
    """
        Run the ZM-VFNS configuration.

        Parameters
        ----------
            setup : dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            basis_function_dispatcher : InterpolatorDispatcher
                basis functions
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
    # collect HQ masses
    Qmc = setup.get("Qmc", 0)
    Qmb = setup.get("Qmb", 0)
    Qmt = setup.get("Qmt", 0)
    # check
    if Qmc > Qmb or Qmb > Qmt:
        raise ValueError("Quark masses are not in c < b < t order!")
    mH2s = [0]  # add 0 as init
    mH2s.append(Qmc * Qmc)
    mH2s.append(Qmb * Qmb)
    mH2s.append(Qmt * Qmt)
    # add infinity as final
    mH2s.append(np.inf)

    # 0 thresholds
    for k in range(1, 5):
        if mH2s[k - 1] <= mu2init <= mu2final <= mH2s[k]:
            return _run_ZMVFNS_0threshold(
                setup, constants, basis_function_dispatcher, targetgrid, 2 + k
            )

    # 1 threshold
    for k in range(1, 4):
        if mH2s[k - 1] <= mu2init < mH2s[k] <= mu2final <= mH2s[k + 1]:
            return _run_ZMVFNS_1threshold(
                setup,
                constants,
                basis_function_dispatcher,
                xgrid,
                targetgrid,
                mH2s[k],
                2 + k,
            )

    # 2 thresholds
    for k in range(1, 3):
        if mH2s[k - 1] <= mu2init < mH2s[k] < mH2s[k + 1] <= mu2final <= mH2s[k + 2]:
            return _run_ZMVFNS_2thresholds(
                setup,
                constants,
                basis_function_dispatcher,
                xgrid,
                targetgrid,
                mH2s[k],
                mH2s[k + 1],
                2 + k,
            )

    # 3 thresholds
    if mu2final < mH2s[1] < mH2s[2] < mH2s[3] < mu2final:
        return _run_ZMVFNS_3thresholds(
            setup,
            constants,
            basis_function_dispatcher,
            xgrid,
            targetgrid,
            mH2s[1],
            mH2s[2],
            mH2s[3],
        )

    # dead end
    raise NotImplementedError(
        "Unknown threshold configuration: m_c^2=%e, m_b^2=%e, m_t^2=%e; mu_init^2=%e, mu_final^2=%e"
        % (mH2s[1], mH2s[2], mH2s[3], mu2init, mu2final)
    )


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
            'targetgrid'    list of x-values which are computed - defaults to ``xgrid``, if not
                            given
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
    ret = utils.merge_dicts(ret, ret_ops)
    return ret


def apply_operator(ret, inputs):
    """
        Apply all available operators to the input PDFs.

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


def get_YAML(ret, stream = None):
    # copy as we will change things
    out = copy.deepcopy(ret)
    # make raw list - we might want to do somthing more numerical here
    for k in ["xgrid","targetgrid"]:
        out[k]  = out[k].tolist()
    for k in out["operators"]:
        out["operators"][k] = out["operators"][k].tolist()
        out["operator_errors"][k] = out["operator_errors"][k].tolist()
    return dump(out,stream)

def write_YAML_to_file(ret, filename):
    with open(filename,"w") as f:
        ret = get_YAML(ret,f)
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
