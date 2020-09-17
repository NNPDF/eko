# -*- coding: utf-8 -*-
"""
    This module contains functions that generate other functions
    All functions will receive a set of parameters so that the output
    function can be always numba compilable

    The generator functions `get_kernel_{kernel_type}` have as signature
    the following parameters:

        - `basis_function`: a callable function with a (N, lnx) signature
        - `nf` : number of flavours
        - `constants`: an instance of the Constants class
"""

import logging

import numpy as np

import eko.strong_coupling as sc
import eko.anomalous_dimensions as ad
import eko.anomalous_dimensions.lo as ad_lo
import eko.anomalous_dimensions.nlo as ad_nlo
from eko import mellin
from eko import ekomath


logger = logging.getLogger(__name__)


class IntegrationKernelObject:
    def __init__(self, kernel_dispatcher):
        self.kernel_dispatcher = kernel_dispatcher
        self.pdf = self.kernel_dispatcher.interpol_dispatcher[0].callable
        self.mode = ""

    @property
    def is_singlet(self):
        return self.mode[0] == "S"

    def get_path_params(self):
        if self.is_singlet:
            return 0.4 * 16.0 / (1.0 - self.var("logx")), 1.0
        return 0.5, 0.0

    def fnc(self, name):
        return self.kernel_dispatcher.fnc[name]

    def var(self, name):
        return self.kernel_dispatcher.var[name]

    def compute_ns(self, n, s1):
        gamma_ns = ad.gamma_ns(
            self.kernel_dispatcher.config["order"],
            self.kernel_dispatcher.config["mode"],
            n,
            s1,
            self.var("nf"),
            self.kernel_dispatcher.constants.CA,
            self.kernel_dispatcher.constants.CF,
        )
        # TODO improve IntKerObj
        # - make ker_gen a package
        # - add a ns module
        # - 1 fnc per method per order
        # - provide map[method][order](bla)
        return np.exp(gamma_ns[0] * self.var("j00"))

    def compute_singlet(self, n, s1):
        gamma_S_0 = ad_lo.gamma_singlet_0(
            n,
            s1,
            self.var("nf"),
            self.kernel_dispatcher.constants.CA,
            self.kernel_dispatcher.constants.CF,
        )
        return ad.exp_singlet(gamma_S_0 * self.var("j00"))[0]

    def __call__(self, u):
        # get transformation to N integral
        path_params = self.get_path_params()
        n = self.fnc("path")(u, *path_params)
        jac = self.fnc("jac")(u, *path_params)
        # check PDF is active
        pj = self.pdf(n, self.var("logx"), self.var("areas"))
        # print(self.pdf.inspect_types())
        if pj == 0.0:
            return 0.0
        # cache the s-es
        s1 = ekomath.harmonic_S1(n)
        # compute the actual evolution kernel
        if self.is_singlet:
            ker = self.compute_singlet(n, s1)
            # select element of matrix
            k = 0 if self.mode[2] == "q" else 1
            l = 0 if self.mode[3] == "q" else 1
            ker = ker[k, l]
        else:
            ker = self.compute_ns(n, s1)
        # recombine everthing
        mellin_prefactor = np.complex(0.0, -1.0 / np.pi)
        return np.real(mellin_prefactor * ker * pj * jac)


class KernelDispatcher:
    """
    Does the common preparation for the kernel functions.

    Parameters
    ----------
        config : dict
            configuration
        interpol_dispatcher : InterpolatorDispatcher
            An instance of the InterpolatorDispatcher class
        constants : Constants
            An instance of the Constants class
    """

    def __init__(self, config, interpol_dispatcher, constants):
        # check
        order = int(config["order"])
        method = config["method"]
        if not method in [
            "iterate-exact",
            "iterate-expanded",
            "truncated",
            "ordered-truncated",
            "decompose-exact",
            "decompose-expanded",
            "perturbative-exact",
            "perturbative-expanded",
        ]:
            raise ValueError(f"Unknown evolution mode {method}")
        if order == 0 and method != "iterate-exact":
            logger.warning("Kernels: In LO we use the exact solution always!")
        self.config = config
        # set managers
        self.interpol_dispatcher = interpol_dispatcher
        self.constants = constants
        # init objects
        self.fnc = {}
        self.var = {}
        # integration stuff
        path, jac = mellin.get_path_Talbot()
        self.fnc["path"] = path
        self.fnc["jac"] = jac
        self.obj = IntegrationKernelObject(self)

    @classmethod
    def from_dict(cls, setup, interpol_dispatcher, constants):
        """
        Create the object from the theory dictionary.

        Read keys:

            - PTO : required, perturbative order
            - ModEv : optional, method to solve RGE, default=EXA=iterate-exact

        Parameters
        ----------
            setup : dict
                theory dictionary
            interpol_dispatcher : InterpolatorDispatcher
                An instance of the InterpolatorDispatcher class
            constants : Constants
                An instance of the Constants class

        Returns
        -------
            obj : cls
                created object
        """
        config = {}
        config["order"] = int(setup["PTO"])
        method = setup.get("ModEv", "EXA")
        mod_ev2method = {
            "EXA": "iterate-exact",
            "EXP": "iterate-expanded",
            "TRN": "truncated",
        }
        method = mod_ev2method.get(method, method)
        config["method"] = method
        config["ev_op_max_order"] = setup.get("ev_op_max_order", 10)
        config["ev_op_iterations"] = setup.get("ev_op_iterations", 10)
        return cls(config, interpol_dispatcher, constants)

    def init_loops(self, nf, a1, a0):
        """
        Called before the heavy grid-basis-functions-sectors loops.

        Parameters
        ----------
            nf : int
                number of flavors
            a1 : float
                strong coupling at target scale
            a0 : float
                strong coupling at initial scale
        """
        self.var["nf"] = nf
        beta_0 = sc.beta_0(nf, self.constants.CA, self.constants.CF, self.constants.TF)
        self.var["beta_0"] = beta_0
        self.var["a1"] = a1
        self.var["a0"] = a0
        self.var["j00"] = np.log(a1 / a0) / beta_0
        if self.config["order"] > 0:
            # TODO reshuffle inits by order
            beta_1 = sc.beta_1(
                nf, self.constants.CA, self.constants.CF, self.constants.TF
            )
            self.var["beta_1"] = beta_1
            b1 = beta_1 / beta_0
            if self.config["method"] in ["iterate-exact", "decompose-exact"]:
                self.var["j11"] = 1 / beta_1 * np.log((1 + a1 * b1) / (1 + a0 * b1))
            else:  # expanded and truncated
                self.var["j11"] = 1 / beta_0 * (a1 - a0)
            self.var["j01"] = self.var["j00"] - b1 * self.var["j11"]

    # def collect_kers(self, nf):
    #     r"""
    #     Returns the integration kernels
    #     :math:`\tilde {\mathbf E}_{ns}(a_s^1 \leftarrow a_s^0)`.

    #     Parameters
    #     ----------
    #         nf : int
    #             number of active flavors
    #         basis_function : callable
    #             accompaying basis function

    #     Returns
    #     -------
    #         ker : dict
    #             (physical) kernels, which will be further modified for the
    #             actual Mellin implementation
    #     """
    #     kers = {}
    #     CA = self.constants.CA
    #     CF = self.constants.CF
    #     beta_0 = sc.beta_0(nf, self.constants.CA, self.constants.CF, self.constants.TF)
    #     order = self.config["order"]
    #     method = self.config["method"]
    #     ev_op_max_order = self.config["ev_op_max_order"]
    #     ev_op_iterations = self.config["ev_op_iterations"]

    #     # provide the integrals
    #     j00 = lambda a1, a0: np.log(a1 / a0) / beta_0
    #     if order > 0:  # NLO constants and integrals
    #         beta_1 = sc.beta_1(
    #             nf, self.constants.CA, self.constants.CF, self.constants.TF
    #         )
    #         b1 = beta_1 / beta_0
    #         if method in ["iterate-exact", "decompose-exact"]:
    #             j11 = nb.njit(
    #                 lambda a1, a0: 1 / beta_1 * np.log((1 + a1 * b1) / (1 + a0 * b1))
    #             )
    #         else:  # expanded and truncated
    #             j11 = nb.njit(lambda a1, a0: 1 / beta_0 * (a1 - a0))
    #         j01 = nb.njit(lambda a1, a0: j00(a1, a0) - b1 * j11(a1, a0))

    #     # singlet kernels
    #     def get_ker_s(s_k, s_l):
    #         """getter for (k,l)-th element of singlet kernel matrix"""

    #         @nb.njit
    #         def ker_s(N, a1, a0, nf):
    #             """a singlet integration kernel"""
    #             # LO
    #             gamma_S_0 = ad_lo.gamma_singlet_0(N, nf, CA, CF)
    #             ln = gamma_S_0 * np.log(a1 / a0) / beta_0 #j00(a1, a0)
    #             e0, l_p, l_m, e_p, e_m = ad.exp_singlet(ln)
    #             #exp_r, r_p, r_m, f_p, f_m = ad.exp_singlet(gamma_S_0 / beta_0)
    #             e = e0
    #             # # NLO
    #             # if order > 0:
    #             #     gamma_S_1 = ad_nlo.gamma_singlet_1(N, nf, CA, CF)
    #             #     if method in ["decompose-exact", "decompose-expanded"]:
    #             #         ln = gamma_S_0 * j01(a1, a0) + gamma_S_1 * j11(a1, a0)
    #             #         e = ad.exp_singlet(ln)[0]
    #             #     elif method in ["iterate-exact", "iterate-expanded"]:
    #             #         delta_a = (a1 - a0) / ev_op_iterations
    #             #         e = np.identity(2, np.complex_)
    #             #         for kk in range(ev_op_iterations):
    #             #             a_half = a0 + (kk + 0.5) * delta_a
    #             #             ln = (
    #             #                 (gamma_S_0 * a_half + gamma_S_1 * a_half ** 2)
    #             #                 / (beta_0 * a_half ** 2 + beta_1 * a_half ** 3)
    #             #                 * delta_a
    #             #             )
    #             #             ek = ad.exp_singlet(ln)[0]
    #             #             e = ek @ e
    #             #     else:  # perturbative
    #             #         r1 = gamma_S_1 / beta_0 - b1 * gamma_S_0
    #             #         r_k = np.zeros(
    #             #             (ev_op_max_order, 2, 2), np.complex_
    #             #         )  # k = 1 .. max_order
    #             #         u_k = np.zeros(
    #             #             (ev_op_max_order + 1, 2, 2), np.complex_
    #             #         )  # k = 0 .. max_order
    #             #         # init with NLO
    #             #         r_k[1 - 1] = r1
    #             #         u_k[0] = np.identity(2, np.complex_)
    #             #         # fill R_k
    #             #         if method == "perturbative-exact":
    #             #             for kk in range(2, ev_op_max_order):
    #             #                 r_k[kk - 1] = -b1 * r_k[kk - 2]
    #             #         # compute R'_k and U_k (simultaneously)
    #             #         max_order = ev_op_max_order
    #             #         if method in ["truncated", "ordered-truncated"]:
    #             #             max_order = 1
    #             #         for kk in range(1, max_order + 1):
    #             #             # rp_k_elems = np.zeros((kk, 2, 2), np.complex_)
    #             #             # for ll in range(kk, 0, -1):
    #             #             rp_k = np.zeros((2, 2), np.complex_)
    #             #             for jj in range(kk):
    #             #                 rp_k += r_k[kk - jj - 1] @ u_k[jj]
    #             #             # rp_k = np.sum(rp_k_elems, 0)
    #             #             u_k[kk] = (
    #             #                 # (e_m @ rp_k @ e_m + e_p @ rp_k @ e_p) / kk
    #             #                 # + ((e_p @ rp_k @ e_m) / ((l_m - l_p)/lo_j - kk))
    #             #                 # + ((e_m @ rp_k @ e_p) / ((l_p - l_m)/lo_j - kk))
    #             #                 (f_m @ rp_k @ f_m + f_p @ rp_k @ f_p) / kk
    #             #                 + ((f_p @ rp_k @ f_m) / ((r_m - r_p) + kk))
    #             #                 + ((f_m @ rp_k @ f_p) / ((r_p - r_m) + kk))
    #             #             )
    #             #         if method in ["truncated", "ordered-truncated"]:
    #             #             u1 = u_k[1]
    #             #             delta_a = (a1 - a0) / ev_op_iterations
    #             #             e = np.identity(2, np.complex_)
    #             #             for kk in range(ev_op_iterations):
    #             #                 al = a0 + kk * delta_a
    #             #                 ah = al + delta_a
    #             #                 ek = e0 + ah * u1 @ e0 - al * e0 @ u1
    #             #                 e = ek @ e
    #             #         else:
    #             #             # U(a_s^1)
    #             #             uh = np.zeros((2, 2), np.complex_)  # k = 0 .. max_order
    #             #             a1power = 1
    #             #             for kk in range(ev_op_max_order + 1):
    #             #                 uh += a1power * u_k[kk]
    #             #                 a1power *= a1
    #             #             # U(a_s^0)
    #             #             ul = np.zeros((2, 2), np.complex_)  # k = 0 .. max_order
    #             #             a0power = 1
    #             #             for kk in range(ev_op_max_order + 1):
    #             #                 ul += a0power * u_k[kk]
    #             #                 a0power *= a0
    #             #             # inv(U(a_s^0))
    #             #             ul_inv = np.linalg.inv(ul)
    #             #             e = uh @ e0 @ ul_inv
    #             return e[s_k][s_l]

    #         return ker_s

    #     kers["S_qq"] = get_ker_s(0, 0)
    #     kers["S_qg"] = get_ker_s(0, 1)
    #     kers["S_gq"] = get_ker_s(1, 0)
    #     kers["S_gg"] = get_ker_s(1, 1)

    #     # non-singlet kernels
    #     def get_ker_ns(mode):
    #         """getter for mode-flavored non-singlet kernel"""

    #         @nb.njit
    #         def ker_ns(n, a1, a0, nf):
    #             """true non-siglet integration kernel"""
    #             # LO
    #             gamma_ns_0 = ad_lo.gamma_ns_0(n, nf, CA, CF)
    #             ln = gamma_ns_0 * np.log(a1 / a0) / beta_0
    #             do_exp = True
    #             # NLO
    #             if order > 0:
    #                 if mode == "p":
    #                     gamma_ns_1 = ad_nlo.gamma_nsp_1(n, nf, CA, CF)
    #                 elif mode == "m":
    #                     gamma_ns_1 = ad_nlo.gamma_nsm_1(n, nf, CA, CF)
    #                 if method == "truncated":
    #                     e = np.exp(ln) * (
    #                         1.0 + j11(a1, a0) * (gamma_ns_1 - b1 * gamma_ns_0)
    #                     )
    #                     do_exp = False
    #                 elif method == "ordered-truncated":
    #                     e = (
    #                         np.exp(ln)
    #                         * (1.0 + a1 / beta_0 * (gamma_ns_1 - b1 * gamma_ns_0))
    #                         / (1.0 + a0 / beta_0 * (gamma_ns_1 - b1 * gamma_ns_0))
    #                     )
    #                     do_exp = False
    #                 else:  # exact and expanded
    #                     ln = gamma_ns_0 * j01(a1, a0) + gamma_ns_1 * j11(a1, a0)
    #             # for exact and expanded we still need to exponentiate
    #             if do_exp:
    #                 e = np.exp(ln)
    #             return e

    #         return ker_ns

    #     # in LO: +=-=v
    #     kers["NS_p"] = get_ker_ns("p")
    #     if order > 0:  # in NLO: -=v
    #         kers["NS_m"] = get_ker_ns("m")
    #     return kers
