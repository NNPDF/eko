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
from ..kernels import QEDnon_singlet as qed_ns
from ..kernels import QEDsinglet as qed_s
from ..kernels import QEDvalence as qed_v
from ..kernels import non_singlet as ns
from ..kernels import singlet as s
from ..member import OpMember

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
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


@nb.njit(cache=True)
def select_QEDsinglet_element(ker, mode0, mode1):
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
    k = 0
    if mode0 == 22:
        k = 1
    elif mode0 == 100:
        k = 2
    elif mode0 == 101:
        k = 3
    l = 0
    if mode1 == 22:
        l = 1
    elif mode1 == 100:
        l = 2
    elif mode1 == 101:
        l = 3
    return ker[k, l]


@nb.njit(cache=True)
def select_QEDvalence_element(ker, mode0, mode1):
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

    k = 0 if mode0 == 10200 else 1
    l = 0 if mode1 == 10200 else 1
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
        self.is_QEDsinglet = mode0 in [100, 101, 21, 22]
        self.is_QEDvalence = mode0 in [10200, 10204]
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
                common mellin inversion integrand
        """
        if self.logx == 0.0:
            return 0.0
        pj = interpolation.evaluate_grid(self.path.n, self.is_log, self.logx, areas)
        if pj == 0.0:
            return 0.0
        return self.path.prefactor * pj * self.path.jac


@nb.njit(cache=True)
def quad_ker(
    u,
    order,
    mode0,
    mode1,
    method,
    is_log,
    logx,
    areas,
    as1,
    as0,
    aem,
    nf,
    L,
    ev_op_iterations,
    ev_op_max_order,
    sv_mode,
    is_threshold,
):
    """Raw evolution kernel inside quad.

    Parameters
    ----------
    u : float
        quad argument
    order : int
        perturbation order
    mode0: int
        pid for first sector element
    mode1 : int
        pid for second sector element
    method : str
        method
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
    sv_mode: int, `enum.IntEnum`
        scale variation mode, see `eko.scale_variations.Modes`
    is_threshold : boolean
        is this an itermediate threshold operator?

    Returns
    -------
    float
        evaluated integration kernel
    """
    ker_base = QuadKerBase(u, is_log, logx, mode0)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0
    if order[1] == 0:
        # compute the actual evolution kernel
        if ker_base.is_singlet:
            gamma_singlet = ad.gamma_singlet(order, ker_base.n, nf)
            # scale var exponentiated is directly applied on gamma
            if sv_mode == sv.Modes.exponentiated:
                gamma_singlet = sv.exponentiated.gamma_variation(
                    gamma_singlet, order, nf, L
                )
            ker = s.dispatcher(
                order,
                method,
                gamma_singlet,
                as1,
                as0,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            )
            # scale var expanded is applied on the kernel
            if sv_mode == sv.Modes.expanded and not is_threshold:
                ker = np.ascontiguousarray(
                    sv.expanded.singlet_variation(gamma_singlet, as1, order, nf, L)
                ) @ np.ascontiguousarray(ker)
            ker = select_singlet_element(ker, mode0, mode1)
        else:
            gamma_ns = ad.gamma_ns(order, mode0, ker_base.n, nf)
            if sv_mode == sv.Modes.exponentiated:
                gamma_ns = sv.exponentiated.gamma_variation(gamma_ns, order, nf, L)
            ker = ns.dispatcher(
                order,
                method,
                gamma_ns,
                as1,
                as0,
                nf,
                ev_op_iterations,
            )
            if sv_mode == sv.Modes.expanded and not is_threshold:
                ker = (
                    sv.expanded.non_singlet_variation(gamma_ns, as1, order, nf, L) * ker
                )
    else:
        # compute the actual evolution kernel
        if ker_base.is_QEDsinglet:
            gamma_singlet = ad.gamma_4x4sector(order, ker_base.n, nf)
            # scale var exponentiated is directly applied on gamma
            if sv_mode == sv.Modes.exponentiated:
                gamma_singlet = sv.exponentiated.gamma_variation(
                    gamma_singlet, order, nf, L
                )
            ker = qed_s.dispatcher(
                order,
                method,
                gamma_singlet,
                as1,
                as0,
                aem,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            )
            # scale var expanded is applied on the kernel
            if sv_mode == sv.Modes.expanded and not is_threshold:
                ker = np.ascontiguousarray(ker) @ np.ascontiguousarray(
                    sv.expanded.singlet_variation(gamma_singlet, as1, order, nf, L)
                )
            ker = select_QEDsinglet_element(ker, mode0, mode1)
        elif ker_base.is_QEDvalence:
            gamma_v = ad.gamma_2x2sector(order, ker_base.n, nf)
            # scale var exponentiated is directly applied on gamma
            if sv_mode == sv.Modes.exponentiated:
                gamma_v = sv.exponentiated.gamma_variation(gamma_v, order, nf, L)
            ker = qed_v.dispatcher(
                order,
                method,
                gamma_v,
                as1,
                as0,
                aem,
                nf,
                ev_op_iterations,
                ev_op_max_order,
            )
            # scale var expanded is applied on the kernel
            if sv_mode == sv.Modes.expanded and not is_threshold:
                ker = np.ascontiguousarray(
                    sv.expanded.singlet_variation(gamma_v, as1, order, nf, L)
                ) @ np.ascontiguousarray(ker)
            ker = select_QEDvalence_element(ker, mode0, mode1)
        else:
            gamma_ns = ad.gamma_ns_qed(order, mode0, ker_base.n, nf)
            # scale var exponentiated is directly applied on gamma
            if sv_mode == sv.Modes.exponentiated:
                gamma_ns = sv.exponentiated.gamma_variation(gamma_ns, order, nf, L)
            ker = qed_ns.dispatcher(
                order,
                method,
                gamma_ns,
                as1,
                as0,
                aem,
                nf,
                ev_op_iterations,
            )
            if sv_mode == sv.Modes.expanded and not is_threshold:
                ker = (
                    sv.expanded.non_singlet_variation(gamma_ns, as1, order, nf, L) * ker
                )

    # recombine everything
    return np.real(ker * integrand)


class Operator(sv.ModeMixin):
    """Internal representation of a single EKO.

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
    is_threshold : bool
        is this an itermediate threshold operator?
    """

    log_label = "Evolution"
    # complete list of possible evolution operators labels
    full_labels = br.full_labels

    def __init__(
        self, config, managers, nf, q2_from, q2_to, mellin_cut=5e-2, is_threshold=False
    ):
        self.config = config
        self.managers = managers
        self.nf = nf
        self.q2_from = q2_from
        self.q2_to = q2_to
        # TODO make 'cut' external parameter?
        self._mellin_cut = mellin_cut
        self.is_threshold = is_threshold
        self.op_members = {}
        self.order = config["order"]

    @property
    def n_pools(self):
        n_pools = self.config["n_integration_cores"]
        if n_pools > 0:
            return n_pools
        # so we subtract from the maximum number
        return max(os.cpu_count() + n_pools, 1)

    @property
    def fact_to_ren(self):
        r"""Returns the factor :math:`(\mu_F/\mu_R)^2`"""
        return self.config["fact_to_ren"]

    @property
    def int_disp(self):
        """Returns the interpolation dispatcher"""
        return self.managers["interpol_dispatcher"]

    @property
    def grid_size(self):
        """Returns the grid size"""
        return self.int_disp.xgrid.size

    def mur2_shift(self, q2):
        """Computes shifted renormalization scale.

        Parameters
        ----------
        q2 : float
            factorization scale

        Returns
        -------
        float
            renormalization scale
        """
        if self.sv_mode == sv.Modes.exponentiated:
            return q2 / self.fact_to_ren
        return q2

    @property
    def a_s(self):
        """Returns the computed values for :math:`a_s`"""
        sc = self.managers["strong_coupling"]
        a0 = sc.a_s(
            self.mur2_shift(self.q2_from), fact_scale=self.q2_from, nf_to=self.nf
        )
        a1 = sc.a_s(self.mur2_shift(self.q2_to), fact_scale=self.q2_to, nf_to=self.nf)
        return (a0, a1)

    @property
    def labels(self):
        """Compute necessary sector labels to compute.

        Returns
        -------
        list(str)
            sector labels
        """
        labels = []
        # the NS sector is dynamic
        if self.order[1] == 0:
            if self.config["debug_skip_non_singlet"]:
                logger.warning("%s: skipping non-singlet sector", self.log_label)
            else:
                # add + as default
                labels.append(br.non_singlet_labels[1])
                if self.order[0] >= 2:  # - becomes different starting from NLO
                    labels.append(br.non_singlet_labels[0])
                if self.order[0] >= 3:  # v also becomes different starting from NNLO
                    labels.append(br.non_singlet_labels[2])
            # singlet sector is fixed
            if self.config["debug_skip_singlet"]:
                logger.warning("%s: skipping singlet sector", self.log_label)
            else:
                labels.extend(br.singlet_labels)
        else:
            # add +u and +d ad default
            labels.append(br.non_singlet_unified_labels[0])
            labels.append(br.non_singlet_unified_labels[2])
            # -u and -d become different starting from O(as1aem1) or O(aem2)
            if order[1] >= 2 or order[0] >= 1:
                labels.append(br.non_singlet_unified_labels[1])
                labels.append(br.non_singlet_unified_labels[3])
            labels.append(br.valence_unified_labels)
            labels.append(br.singlet_unified_labels)
        return labels

    def quad_ker(self, label, logx, areas):
        """Partially initialized integrand function.

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
        functools.partial
            partially initialized integration kernel

        """
        return functools.partial(
            quad_ker,
            order=self.order,
            mode0=label[0],
            mode1=label[1],
            method=self.config["method"],
            is_log=self.int_disp.log,
            logx=logx,
            areas=areas,
            as1=self.a_s[1],
            as0=self.a_s[0],
            aem=self.config["alphaem"] / 4 / np.pi,
            nf=self.nf,
            L=np.log(self.fact_to_ren),
            ev_op_iterations=self.config["ev_op_iterations"],
            ev_op_max_order=self.config["ev_op_max_order"],
            sv_mode=self.sv_mode,
            is_threshold=self.is_threshold,
        )

    def initialize_op_members(self):
        """Init all ops with identity or zeros if we skip them"""
        eye = OpMember(
            np.eye(self.grid_size), np.zeros((self.grid_size, self.grid_size))
        )
        zero = OpMember(*[np.zeros((self.grid_size, self.grid_size))] * 2)
        for n in self.full_labels:
            if n in self.labels:
                # non-singlet evolution and diagonal op are identities
                if n in br.non_singlet_labels or n[0] == n[1]:
                    self.op_members[n] = eye.copy()
                else:
                    self.op_members[n] = zero.copy()
            else:
                self.op_members[n] = zero.copy()

    def run_op_integration(
        self,
        log_grid,
    ):
        """Run the integration for each grid point

        Parameters
        ----------
        log_grid : tuple(k, logx)
            log grid point with relative index

        Returns
        -------
        list
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

        logger.info(
            "%s: computing operators - %u/%u took: %6f s",
            self.log_label,
            k + 1,
            self.grid_size,
            (time.perf_counter() - start_time),
        )
        return column

    def compute(self):
        """Compute the actual operators (i.e. run the integrations)."""
        self.initialize_op_members()

        # skip computation ?
        if np.isclose(self.q2_from, self.q2_to):
            logger.info(
                "%s: skipping unity operator at %e", self.log_label, self.q2_from
            )
            self.copy_ns_ops()
            return

        logger.info(
            "%s: computing operators %e -> %e, nf=%d",
            self.log_label,
            self.q2_from,
            self.q2_to,
            self.nf,
        )
        logger.info(
            "%s: µ_R^2 distance: %e -> %e",
            self.log_label,
            self.mur2_shift(self.q2_from),
            self.mur2_shift(self.q2_to),
        )
        if self.sv_mode.name != "unvaried":
            logger.info(
                "Scale Variation: (µ_F/µ_R)^2 = %e, mode: %s",
                self.fact_to_ren,
                self.sv_mode.name,
            )
        logger.info(
            "%s: a_s distance: %e -> %e", self.log_label, self.a_s[0], self.a_s[1]
        )
        logger.info(
            "%s: order: (%d, %d), solution strategy: %s",
            self.log_label,
            self.order[0],
            self.order[1],
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
        # or avoid opening a single pool
        args = (self.run_op_integration, enumerate(np.log(self.int_disp.xgrid_raw)))
        if self.n_pools == 1:
            res = map(*args)
        else:
            with Pool(self.n_pools) as pool:
                res = pool.map(*args)

        # collect results
        for k, row in enumerate(res):
            for l, entry in enumerate(row):
                for label, (val, err) in entry.items():
                    self.op_members[label].value[k][l] = val
                    self.op_members[label].error[k][l] = err

        # closing comment
        logger.info(
            "%s: Total time %f s",
            self.log_label,
            time.perf_counter() - tot_start_time,
        )

    def copy_ns_ops(self):
        """Copy non-singlet kernels, if necessary"""
        if self.order[0] == 1:  # in LO +=-=v
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
        elif self.order[0] == 2:  # in NLO -=v
            self.op_members[
                (br.non_singlet_pids_map["nsV"], 0)
            ].value = self.op_members[(br.non_singlet_pids_map["ns-"], 0)].value.copy()
            self.op_members[
                (br.non_singlet_pids_map["nsV"], 0)
            ].error = self.op_members[(br.non_singlet_pids_map["ns-"], 0)].error.copy()
