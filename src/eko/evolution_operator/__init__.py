# -*- coding: utf-8 -*-
r"""
This module contains the central operator classes.

See :doc:`Operator overview </code/Operators>`.
"""

import functools
import logging
import os
import time
from multiprocessing import Pool

import numba as nb
import numpy as np
from scipy import integrate

from .. import anomalous_dimensions as ad
from .. import basis_rotation as br
from .. import interpolation, mellin
from .. import scale_variations as sv
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


@nb.njit("c16(c16[:,:],u2,u2)")
def select_singlet_element(ker, mode0, mode1):
    """
    Select element of the singlet matrix

    Parameters
    ----------
        ker : numpy.ndarray
            singlet integration kernel
        mode0 : int
            id for first sector element
        mode1 : int
            id for second sector element
    Returns
    -------
        ker : complex
            singlet integration kernel element
    """

    k = 0 if mode0 == 100 else 1
    l = 0 if mode1 == 100 else 1
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
        mode0 : str
            first sector element
    """

    def __init__(self, u, is_log, logx, mode0):
        self.is_singlet = mode0 in [100, 21, 90]
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


@nb.njit("f8(f8,u1,u2,u2,string,b1,f8,f8[:,:],f8,f8,f8,f8,u4,u1,u1)", cache=True)
def quad_ker(
    u,
    order,
    mode0,
    mode1,
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
        mode0: int
            pid for first sector element
        mode1 : int
            pid for second sector element
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
    ker_base = QuadKerBase(u, is_log, logx, mode0)
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
        ker = select_singlet_element(ker, mode0, mode1)
    else:
        gamma_ns = ad.gamma_ns(order, mode0, ker_base.n, nf)
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

    operator_type = "Evolution"
    n_pools = int(os.cpu_count() / 2)

    def __init__(self, config, managers, nf, q2_from, q2_to=None, mellin_cut=5e-2):
        self.config = config
        self.managers = managers
        self.nf = nf
        self.q2_from = q2_from
        self.q2_to = q2_to
        # TODO make 'cut' external parameter?
        self._mellin_cut = mellin_cut
        self.op_members = {}
        # TODO: temporary fix
        self.ome_members = {}

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
            logger.warning("%s: skipping non-singlet sector", self.operator_type)
        else:
            # add + as default
            labels.append(br.non_singlet_labels[1])
            if order >= 1:  # - becomes different starting from NLO
                labels.append(br.non_singlet_labels[0])
            if order >= 2:  # v also becomes different starting from NNLO
                labels.append(br.non_singlet_labels[2])
        # singlet sector is fixed
        if self.config["debug_skip_singlet"]:
            logger.warning("%s: skipping singlet sector", self.operator_type)
        else:
            labels.extend(br.singlet_labels)
        return labels

    def quad_ker(self, label, logx, areas):
        """
        Partially initialized integrand function

        Parameters
        ----------
            label: tuple
                operator element pids
            logx: float
                Mellin inversion point
            areas : tuple
                basis function configuration

        Returns
        -------
            quad_ker : functools.partial
                partially initialized intration kernel

        """
        return functools.partial(
            quad_ker,
            # TODO: implement N3LO evolution kernels
            order=self.config["order"] if self.config["order"] != 3 else 2,
            mode0=label[0],
            mode1=label[1],
            method=self.config["method"],
            is_log=self.int_disp.log,
            logx=logx,
            areas=areas,
            a1=self.a_s[1],
            a0=self.a_s[0],
            nf=self.nf,
            L=np.log(self.fact_to_ren),
            ev_op_iterations=self.config["ev_op_iterations"],
            ev_op_max_order=self.config["ev_op_max_order"],
            sv_mode=self.sv_mode,
        )

    def initialize_op_members(self):
        """Init all ops with identity or zeros if we skip them"""
        eye = OpMember(
            np.eye(self.grid_size), np.zeros((self.grid_size, self.grid_size))
        )
        zero = OpMember(*[np.zeros((self.grid_size, self.grid_size))] * 2)
        for n in br.full_labels:
            if n in self.labels:
                # off diag singlet are zero
                if n in br.singlet_labels and n[0] != n[1]:
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
                    self.quad_ker(label, logx, bf.areas_representation),
                    0.5,
                    1.0 - self._mellin_cut,
                    epsabs=1e-12,
                    epsrel=1e-5,
                    limit=100,
                    full_output=1,
                )
                temp_dict[label] = res[:2]
            column.append(temp_dict)
        print(
            f"{self.operator_type}: computing operators: - {k+1}/{self.grid_size} took: {(time.perf_counter() - start_time):6f} s"  # pylint: disable=line-too-long
        )
        return column

    def compute(self):
        """compute the actual operators (i.e. run the integrations)"""
        self.initialize_op_members()

        # skip computation ?
        if np.isclose(self.q2_from, self.q2_to):
            logger.info(
                "%s: skipping unity operator at %e", self.operator_type, self.q2_from
            )
            self.copy_ns_ops()
            return

        logger.info(
            "%s: computing operators %e -> %e, nf=%d",
            self.operator_type,
            self.q2_from,
            self.q2_to,
            self.nf,
        )
        logger.info(
            "%s: µ_R^2 distance: %e -> %e",
            self.operator_type,
            self.q2_from / self.fact_to_ren,
            self.q2_to / self.fact_to_ren,
        )
        if self.sv_mode != 0:
            logger.info(
                "Scale Variation: (µ_F/µ_R)^2 = %e, mode: %s",
                self.fact_to_ren,
                "exponentiated" if self.sv_mode == 1 else "expanded",
            )
        logger.info(
            "%s: a_s distance: %e -> %e", self.operator_type, self.a_s[0], self.a_s[1]
        )
        logger.info(
            "%s: order: %d, solution strategy: %s",
            self.operator_type,
            self.config["order"],
            self.config["method"],
        )

        self.integrate()
        # copy non-singlet kernels, if necessary
        self.copy_ns_ops()

    def integrate(
        self,
    ):
        """Run the integration"""
        tot_start_time = time.perf_counter()

        # run integration in parallel for each grid point
        with Pool(self.n_pools) as pool:
            res = pool.map(
                self.run_op_integration,
                enumerate(np.log(self.int_disp.xgrid_raw)),
            )

        # collect results
        for k, row in enumerate(res):
            for l, entry in enumerate(row):
                for label, (val, err) in entry.items():

                    # TODO: same as labels, promote ome_members to be op_members
                    if self.operator_type == "Evolution":
                        self.op_members[label].value[k][l] = val
                        self.op_members[label].error[k][l] = err
                    else:
                        self.ome_members[label].value[k][l] = val
                        self.ome_members[label].error[k][l] = err

        # closing comment
        logger.info(
            "%s: Total time %f s",
            self.operator_type,
            time.perf_counter() - tot_start_time,
        )

    def copy_ns_ops(self):
        """Copy non-singlet kernels, if necessary"""
        order = self.config["order"]
        if order == 0:  # in LO +=-=v
            for label in ["nsV", "ns-"]:
                self.op_members[
                    (br.non_singlet_pids_map[label], 0)
                ].value = self.op_members[
                    (br.non_singlet_pids_map["ns+"], 0)
                ].value.copy()
                self.op_members[
                    (br.non_singlet_pids_map[label], 0)
                ].error = self.op_members[
                    (br.non_singlet_pids_map["ns+"], 0)
                ].error.copy()
        elif order == 1:  # in NLO -=v
            self.op_members[
                (br.non_singlet_pids_map["nsV"], 0)
            ].value = self.op_members[(br.non_singlet_pids_map["ns-"], 0)].value.copy()
            self.op_members[
                (br.non_singlet_pids_map["nsV"], 0)
            ].error = self.op_members[(br.non_singlet_pids_map["ns-"], 0)].error.copy()
