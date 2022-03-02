# -*- coding: utf-8 -*-
r"""
This module contains the central operator classes.

See :doc:`Operator overview </code/Operators>`.
"""

import logging
import os
import time
from multiprocessing import Pool

import numba as nb
import numpy as np
from scipy import integrate

from .. import anomalous_dimensions as ad
from .. import interpolation, mellin
from .. import scale_variations as sv
from ..basis_rotation import full_labels, singlet_labels
from ..kernels import non_singlet as ns
from ..kernels import singlet as s
from ..member import OpMember

logger = logging.getLogger(__name__)

sv_mode_dict = dict(
    zip(
        [None, "exponentiated", "expanded"],
        [sv.unvaried, sv.mode_exponentiated, sv.mode_expanded],
    )
)


@nb.njit("c16(c16[:,:],string)")
def select_singlet_element(ker, mode):
    """
    Select element of the singlet matrix

    Parameters
    ----------
        mode : str
            sector element
        ker : numpy.ndarray
            singlet integration kernel

    Returns
    -------
        ker : complex
            singlet integration kernel element
    """
    k = 0 if mode[2] == "q" else 1
    l = 0 if mode[3] == "q" else 1
    return ker[k, l]


spec = [
    ("is_singlet", nb.boolean),
    ("is_log", nb.boolean),
    ("logx", nb.float64),
    ("u", nb.float64),
]


@nb.experimental.jitclass(spec)
class QuadKerBase:
    """
    Manage the common part of Mellin inversion integral

    Parameters
    ----------
        u : float
            quad argument
        is_log : boolean
            is a logarithmic interpolation
        logx : float
            Mellin inversion point
        mode : str
            sector element
    """

    def __init__(self, u, is_log, logx, mode):
        self.is_singlet = mode[0] == "S"
        self.is_log = is_log
        self.u = u
        self.logx = logx

    @property
    def path(self):
        """Returns the associated instance of :class:`eko.mellin.Path`"""
        return mellin.Path(self.u, self.logx, self.is_singlet)

    @property
    def n(self):
        """Returns the Mellin moment N"""
        return self.path.n

    def integrand(
        self,
        areas,
    ):
        """
        Get transformation to Mellin space integral

        Parameters
        ----------
            areas : tuple
                basis function configuration

        Returns
        -------
            base_integrand: complex
                common mellin inversion intgrand
        """
        if self.logx == 0.0:
            return 0.0
        pj = interpolation.evaluate_grid(self.path.n, self.is_log, self.logx, areas)
        if pj == 0.0:
            return 0.0
        return self.path.prefactor * pj * self.path.jac


@nb.njit("f8(f8,u1,string,string,b1,f8,f8[:,:],f8,f8,f8,f8,u4,u1,u1)", cache=True)
def quad_ker(
    u,
    order,
    mode,
    method,
    is_log,
    logx,
    areas,
    a1,
    a0,
    nf,
    L,
    ev_op_iterations,
    ev_op_max_order,
    sv_mode,
):
    """
    Raw evolution kernel inside quad.

    Parameters
    ----------
        u : float
            quad argument
        order : int
            perturbation order
        method : str
            method
        mode : str
            sector element
        is_log : boolean
            is a logarithmic interpolation
        logx : float
            Mellin inversion point
        areas : tuple
            basis function configuration
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        L : float
            logarithm of the squared ratio of factorization and renormalization scale
        ev_op_iterations : int
            number of evolution steps
        ev_op_max_order : int
            perturbative expansion order of U
        sv_mode: int
            use scale variation mode 0: none, 1: exponentiated, 2: expanded

    Returns
    -------
        ker : float
            evaluated integration kernel
    """
    ker_base = QuadKerBase(u, is_log, logx, mode)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0

    # compute the actual evolution kernel
    if ker_base.is_singlet:
        gamma_singlet = ad.gamma_singlet(order, ker_base.n, nf)
        # scale var A is directly applied on gamma
        if sv_mode == sv.mode_exponentiated:
            gamma_singlet = sv.exponentiated.gamma_variation(
                gamma_singlet, order, nf, L
            )
        ker = s.dispatcher(
            order, method, gamma_singlet, a1, a0, nf, ev_op_iterations, ev_op_max_order
        )
        # scale var B is applied on the kernel
        if sv_mode == sv.mode_expanded:
            ker = np.ascontiguousarray(ker) @ np.ascontiguousarray(
                sv.expanded.singlet_variation(gamma_singlet, a1, order, nf, L)
            )
        ker = select_singlet_element(ker, mode)
    else:
        gamma_ns = ad.gamma_ns(order, mode[-1], ker_base.n, nf)
        if sv_mode == sv.mode_exponentiated:
            gamma_ns = sv.exponentiated.gamma_variation(gamma_ns, order, nf, L)
        ker = ns.dispatcher(
            order,
            method,
            gamma_ns,
            a1,
            a0,
            nf,
            ev_op_iterations,
        )
        if sv_mode == sv.mode_expanded:
            ker = ker * sv.expanded.non_singlet_variation(gamma_ns, a1, order, nf, L)

    # recombine everthing
    return np.real(ker * integrand)


class Operator:
    """
    Internal representation of a single EKO.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
        config : dict
            configuration
        managers : dict
            managers
        nf : int
            number of active flavors
        q2_from : float
            evolution source
        q2_to : float
            evolution target
        mellin_cut : float
            cut to the upper limit in the mellin inversion
    """

    def __init__(self, config, managers, nf, q2_from, q2_to=None, mellin_cut=5e-2):
        self.config = config
        self.managers = managers
        self.nf = nf
        self.q2_from = q2_from
        self.q2_to = q2_to
        # TODO make 'cut' external parameter?
        self._mellin_cut = mellin_cut
        self.op_members = {}

    @property
    def fact_to_ren(self):
        r"""Returns the factor :math:`(\mu_F/\mu_R)^2`"""
        return self.config["fact_to_ren"]

    @property
    def sv_mode(self):
        """Returns the scale variation mode"""
        return sv_mode_dict[self.config["ModSV"]]

    @property
    def int_disp(self):
        """Returns the interpolation dispatcher"""
        return self.managers["interpol_dispatcher"]

    @property
    def grid_size(self):
        """Returns the grid size"""
        return self.int_disp.xgrid.size

    @property
    def a_s(self):
        """Returns the computed values for :math:`a_s`"""
        sc = self.managers["strong_coupling"]
        a0 = sc.a_s(
            self.q2_from / self.fact_to_ren, fact_scale=self.q2_from, nf_to=self.nf
        )
        a1 = sc.a_s(self.q2_to / self.fact_to_ren, fact_scale=self.q2_to, nf_to=self.nf)
        return (a0, a1)

    @property
    def labels(self):
        """
        Compute necessary sector labels to compute.

        Returns
        -------
            labels : list(str)
                sector labels
        """
        order = self.config["order"]
        labels = []
        # the NS sector is dynamic
        if self.config["debug_skip_non_singlet"]:
            logger.warning("Evolution: skipping non-singlet sector")
        else:
            # add + as default
            labels.append("NS_p")
            if order >= 1:  # - becomes different starting from NLO
                labels.append("NS_m")
            if order >= 2:  # v also becomes different starting from NNLO
                labels.append("NS_v")
        # singlet sector is fixed
        if self.config["debug_skip_singlet"]:
            logger.warning("Evolution: skipping singlet sector")
        else:
            labels.extend(singlet_labels)
        return labels

    @property
    def quad_ker(self):
        """Integrand function"""
        return quad_ker

    def initialize_op_members(self):
        """Init all ops with identity or zeros if we skip them"""
        eye = OpMember(
            np.eye(self.grid_size), np.zeros((self.grid_size, self.grid_size))
        )
        zero = OpMember(*[np.zeros((self.grid_size, self.grid_size))] * 2)
        for n in full_labels:
            if n in self.labels:
                # off diag singlet are zero
                if n in ["S_qg", "S_gq"]:
                    self.op_members[n] = zero.copy()
                else:
                    self.op_members[n] = eye.copy()
            else:
                self.op_members[n] = zero.copy()

    def run_op_integration(
        self,
        log_grid,
    ):
        """
        Run the integration for each grid point

        Parameters
        ----------
            log_grid : tuple(k, logx)
                log grid point with relative index

        Returns
        -------
            column : list
                computed operators at the give grid point

        """
        column = []
        k, logx = log_grid
        start_time = time.perf_counter()
        # iterate basis functions
        for l, bf in enumerate(self.int_disp):
            if k == l and l == self.grid_size - 1:
                continue
            temp_dict = {}
            # iterate sectors
            for label in self.labels:
                res = integrate.quad(
                    self.quad_ker,
                    0.5,
                    1.0 - self._mellin_cut,
                    args=(
                        self.config["order"],
                        label,
                        self.config["method"],
                        self.int_disp.log,
                        logx,
                        bf.areas_representation,
                        self.a_s[1],
                        self.a_s[0],
                        self.nf,
                        np.log(self.fact_to_ren),
                        self.config["ev_op_iterations"],
                        self.config["ev_op_max_order"],
                        self.sv_mode,
                    ),
                    epsabs=1e-12,
                    epsrel=1e-5,
                    limit=100,
                    full_output=1,
                )
                temp_dict[label] = res[:2]
            column.append(temp_dict)

        print(
            f"Evolution: computing operators: - {k+1}/{self.grid_size} took: {(time.perf_counter() - start_time)} s"  # pylint: disable=line-too-long
        )
        return column

    def compute(self):
        """compute the actual operators (i.e. run the integrations)"""
        self.initialize_op_members()

        # skip computation ?
        if np.isclose(self.q2_from, self.q2_to):
            logger.info("Evolution: skipping unity operator at %e", self.q2_from)
            self.copy_ns_ops()
            return

        tot_start_time = time.perf_counter()
        logger.info(
            "Evolution: computing operators %e -> %e, nf=%d",
            self.q2_from,
            self.q2_to,
            self.nf,
        )
        logger.info(
            "Evolution: µ_R^2 distance: %e -> %e",
            self.q2_from / self.fact_to_ren,
            self.q2_to / self.fact_to_ren,
        )
        if self.sv_mode != 0:
            logger.info(
                "Scale Variation: (µ_F/µ_R)^2 = %e, mode: %s",
                self.fact_to_ren,
                "exponentiated" if self.sv_mode == 1 else "expanded",
            )
        logger.info("Evolution: a_s distance: %e -> %e", self.a_s[0], self.a_s[1])
        logger.info(
            "Evolution: order: %d, solution strategy: %s",
            self.config["order"],
            self.config["method"],
        )

        # run integration in parallel for each grid point
        with Pool(int(os.cpu_count() / 2)) as pool:
            res = pool.map(
                self.run_op_integration,
                enumerate(np.log(self.int_disp.xgrid_raw)),
            )

        # collect results
        for k, row in enumerate(res):
            for l, entry in enumerate(row):
                for label, (val, err) in entry.items():
                    self.op_members[label].value[k][l] = val
                    self.op_members[label].error[k][l] = err

        # closing comment
        logger.info("Evolution: Total time %f s", time.perf_counter() - tot_start_time)
        # copy non-singlet kernels, if necessary
        self.copy_ns_ops()

    def copy_ns_ops(self):
        """Copy non-singlet kernels, if necessary"""
        order = self.config["order"]
        if order == 0:  # in LO +=-=v
            for label in ["NS_v", "NS_m"]:
                self.op_members[label].value = self.op_members["NS_p"].value.copy()
                self.op_members[label].error = self.op_members["NS_p"].error.copy()
        elif order == 1:  # in NLO -=v
            self.op_members["NS_v"].value = self.op_members["NS_m"].value.copy()
            self.op_members["NS_v"].error = self.op_members["NS_m"].error.copy()
