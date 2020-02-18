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
from eko.constants import Constants
from eko.alpha_s import StrongCoupling

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


def _run_nonsinglet(kernel_dispatcher, xgrid):
    """
        Solves the non-singlet case.

        Parameters
        ----------
            kernel_dispatcher: KernelDispatcher
                instance of KernelDispatcher from which compiled kernels can be obtained
            xgrid: array
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
    path, jac = mellin.get_path_Talbot()
    log_prefix = "computing NS operator - %s"

    # Generate integrands
    logger.info(log_prefix, "compiling kernels")
    integrands = []
    for kernel in kernels:
        kernel_int = mellin.compile_integrand(kernel, path, jac)
        integrands.append(kernel_int)
    logger.info(log_prefix, "compilation done")

    def run_thread(integrand, extra_args):
        result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
        return result

    operators = []
    operator_errors = []

    grid_size = len(xgrid)
    for k, xk in enumerate(xgrid):
        extra_args = nb.typed.List()
        extra_args.append(np.log(xk))
        extra_args.append(0.5)
        extra_args.append(0.0)
        out = _parallelize_on_basis(integrands, run_thread, extra_args)
        operators.append(np.array(out)[:, 0])
        operator_errors.append(np.array(out)[:, 1])
        log_text = f"{k+1}/{grid_size}"
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


def _run_singlet(kernel_dispatcher, xgrid):
    """
        Solves the singlet case.

        Parameters
        ----------
            kernel_dispatcher: KernelDispatcher
                instance of KernelDispatcher from which compiled kernels can be obtained
            xgrid: array
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
    path, jac = mellin.get_path_Talbot()

    log_prefix = "computing singlet operator - %s"
    logger.info(log_prefix, "compiling kernels")
    # Generate integrands
    integrands = []
    for kernel_set in kernels:
        kernel_int = []
        for ker in kernel_set:
            kernel_int.append(mellin.compile_integrand(ker, path, jac))
        integrands.append(kernel_int)
    logger.info(log_prefix, "compilation done")

    def run_thread(integrands, extra_args):
        """ The output of this function is a list of tuple (result, error)
        for qq, qg, gq, gg in that order """
        all_res = []
        for integrand in integrands:
            result = mellin.inverse_mellin_transform(integrand, cut, extra_args)
            all_res.append(result)
        return all_res

    # perform
    all_output = []
    grid_size = len(xgrid)
    for k, xk in enumerate(xgrid):
        extra_args = nb.typed.List()
        extra_args.append(np.log(xk))
        # p3: 1.0, 0.5 + np.power(xk,1.5)*25
        # p4: 0.5, 0.0
        # p5: -0.4*16/(-1+lnxk), 1.0
        # p6: -0.4*18/(-0.5+lnxk), 1.0
        # p7: -0.4*32/(-0.5+lnxk), 1.0
        extra_args.append(-0.4*16/(-1.0+np.log(xk)))
        extra_args.append(1.0)
        out = _parallelize_on_basis(integrands, run_thread, extra_args)
        #out = [[[0,0]]*4]*grid_size
        all_output.append(out)
        log_text = f"{k+1}/{grid_size}"
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
    setup, constants, basis_function_dispatcher, xgrid, nf, mu2init, mu2final, mu2step=None
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
            xgrid : array
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
    delta_t = alpha_s.get_evolution_params(setup, constants, nf, mu2init, mu2final,mu2step)
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t
    )

    # run non-singlet
    ret_ns = _run_nonsinglet(kernel_dispatcher, xgrid)
    # run singlet
    ret_s = _run_singlet(kernel_dispatcher, xgrid)
    # join elements
    ret = utils.merge_dicts(ret_ns, ret_s)
    return ret


def _run_FFNS(setup, constants, basis_function_dispatcher, xgrid):
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
            xgrid : array
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
        xgrid,
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


def _run_ZMVFNS_0threshold(setup, constants, basis_function_dispatcher, xgrid, nf):
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
        xgrid : array
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
        setup, constants, basis_function_dispatcher, xgrid, nf, mu2init, mu2final
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
    setup, constants, basis_function_dispatcher, xgrid, m2Threshold, nf_init
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
        xgrid,
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
        xgrid,
        nf_init + 1,
        m2Threshold1,
        m2Threshold2,
    )
    # step three: m^2_q2 -> mu^2_final
    step3 = _run_step(
        setup,
        constants,
        basis_function_dispatcher,
        xgrid,
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
    setup, constants, basis_function_dispatcher, xgrid, m2c, m2b, m2t
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
    step2 = _run_step(setup, constants, basis_function_dispatcher, xgrid, 4, m2c, m2b)
    # step three: m^2_b -> m^2_t
    step3 = _run_step(setup, constants, basis_function_dispatcher, xgrid, 5, m2b, m2t)
    # step four: m^2_t -> mu^2_final
    step4 = _run_step(
        setup, constants, basis_function_dispatcher, xgrid, 6, m2t, mu2final
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


def _run_ZM_VFNS(setup, constants, basis_function_dispatcher, xgrid):
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
                setup, constants, basis_function_dispatcher, xgrid, 2 + k
            )

    # 1 threshold
    for k in range(1, 4):
        if mH2s[k - 1] <= mu2init < mH2s[k] <= mu2final <= mH2s[k + 1]:
            return _run_ZMVFNS_1threshold(
                setup, constants, basis_function_dispatcher, xgrid, mH2s[k], 2 + k
            )

    # 2 thresholds
    for k in range(1, 3):
        if mH2s[k - 1] <= mu2init < mH2s[k] < mH2s[k + 1] <= mu2final <= mH2s[k + 2]:
            return _run_ZMVFNS_2thresholds(
                setup,
                constants,
                basis_function_dispatcher,
                xgrid,
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
    basis_function_dispatcher = interpolation.InterpolatorDispatcher(
        xgrid, polynom_rank, log=is_log_interpolation
    )

    # Get the scheme and set up the thresholds
    FNS = setup["FNS"]
    threshold_holder = Threshold(setup, scheme = FNS)

    # Now generate the operator alpha_s class
    alpha_ref = setup['alphas']
    q_ref = setup["Qref"]

    alpha_s = StrongCoupling(constants, alpha_ref, q_ref, threshold_holder) 

    # Start filling the output dictionary
    ret = {
        "xgrid": xgrid,
        "polynomial_degree": polynom_rank,
        "log": is_log_interpolation,
        "basis": basis_function_dispatcher,
        "operators": {},
        "operator_errors": {},
    }

    # check FNS and split
    FNS = setup["FNS"]
    if FNS == "FFNS":
        ret_ops = _run_FFNS(setup, constants, basis_function_dispatcher, xgrid)
    elif FNS == "ZM-VFNS":
        ret_ops = _run_ZM_VFNS(setup, constants, basis_function_dispatcher, xgrid)
    else:
        raise ValueError(f"Unknown FNS: {FNS}")
    # join operators
    ret = utils.merge_dicts(ret, ret_ops)
    return ret


def apply_operator(ret, inputs, targetgrid=None):
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
