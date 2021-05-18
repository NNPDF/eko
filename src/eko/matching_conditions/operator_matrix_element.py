# -*- coding: utf-8 -*-
"""
This module defines the operator matrix elements for the non-trivial matching conditions in the
|VFNS| evolution.
"""

import time
import logging

import numpy as np
from scipy import integrate
import numba as nb

from .. import mellin
from .. import interpolation
from ..member import OpMember

from .nnlo import A_singlet_2, A_ns_2
from ..anomalous_dimensions import harmonics
from ..member import singlet_labels

# TODO: order might be removed if N3LO matching conditions will not be implemented

logger = logging.getLogger(__name__)


@nb.njit("f8(f8,string,b1,f8,f8[:,:],string,f8)", cache=True)
def quad_ker(
    u,
    # order,
    mode,
    is_log,
    logx,
    areas,
    backward_method,
    a_s,
):
    """
    Raw kernel inside quad

    Parameters
    ----------
        u : float
            quad argument
        mode : str
            element in the singlet sector
        is_log : boolean
            logarithmic interpolation
        logx : float
            Mellin inversion point
        areas : tuple
            basis function configuration
        backward_method : [exact, expanded or None]
            None or method for inverting the matching contidtion (exact or expanded)
        a_s : float
                strong coupling, needed only for the exact inverse

    Returns
    -------
        ker : float
            evaluated integration kernel
    """

    is_singlet = mode[0] == "S"
    # get transformation to N integral
    if is_singlet:
        r, o = 0.4 * 16.0 / (1.0 - logx), 1.0
    else:
        r, o = 0.5, 0.0

    n = mellin.Talbot_path(u, r, o)
    jac = mellin.Talbot_jac(u, r, o)
    # check PDF is active
    if is_log:
        pj = interpolation.log_evaluate_Nx(n, logx, areas)
    else:
        pj = interpolation.evaluate_Nx(n, logx, areas)

    if pj == 0.0:
        return 0.0

    # compute the harmonics
    sx = np.array(
        [harmonics.harmonic_S1(n), harmonics.harmonic_S2(n), harmonics.harmonic_S3(n)]
    )

    # compute the actual evolution kernel
    if is_singlet:
        ker = A_singlet_2(n, sx)
        if backward_method == "exact":
            ker_ns = np.array([[A_ns_2(n, sx), 0 + 0j], [0 + 0j, 0 + 0j]], np.complex_)
            ker = np.linalg.inv(np.eye(2) + a_s ** 2 * (ker_ns + ker))
        # select element of matrix
        k = 0 if mode[2] == "q" else 1
        l = 0 if mode[3] == "q" else 1
        ker = ker[k, l]
    else:
        ker = A_ns_2(n, sx)
        if backward_method == "exact":
            ker = 1 / (1 + a_s ** 2 * ker)

    # expanded inversion is the same for S and NS
    if backward_method == "expanded":
        ker = -ker

    # recombine everthing
    mellin_prefactor = complex(0.0, -1.0 / np.pi)
    return np.real(mellin_prefactor * ker * pj * jac)


class OperatorMatrixElement:
    """
    Internal representation of a single Operator Matrix Element.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
        config : dict
            configuration
        managers : dict
            managers
        mellin_cut : float
            cut to the upper limit in the mellin inversion
    """

    def __init__(self, config, managers, mellin_cut=1e-2):
        self.backward_method = config["backward_inversion"]
        if self.backward_method is None:
            self.backward_method = ""
        self.order = config["order"]
        self.int_disp = managers["interpol_dispatcher"]
        self._mellin_cut = mellin_cut
        self.ome_members = {}

    def compute(self, a_s=0.0):
        """compute the actual operators (i.e. run the integrations)

        Parameters
        ----------
            a_s : float, optional
                strong coupling, needed only for the exact inverse
        """

        # init all ops with zeros
        grid_size = len(self.int_disp.xgrid)
        labels = ["NS", *singlet_labels]
        for n in labels:
            self.ome_members[n] = OpMember(
                np.eye(grid_size), np.zeros((grid_size, grid_size))
            )

        # if LO and NLO no need to do anything
        if self.order <= 1:
            logger.info(
                "Matching: only trivial conditions are needed at PTO = %d", self.order
            )
            return

        tot_start_time = time.perf_counter()
        logger.info("Matching: computing operators - 0/%d", grid_size)
        # iterate output grid
        for k, logx in enumerate(np.log(self.int_disp.xgrid_raw)):
            start_time = time.perf_counter()
            # iterate basis functions
            for l, bf in enumerate(self.int_disp):
                if k == l and l == grid_size - 1:
                    continue
                # iterate sectors
                for label in labels:
                    # compute and set
                    res = integrate.quad(
                        quad_ker,
                        0.5,
                        1.0 - self._mellin_cut,
                        args=(
                            # self.order,
                            label,
                            self.int_disp.log,
                            logx,
                            bf.areas_representation,
                            self.backward_method,
                            a_s,
                        ),
                        epsabs=1e-12,
                        epsrel=1e-5,
                        limit=100,
                        full_output=1,
                    )
                    val, err = res[:2]
                    self.ome_members[label].value[k][l] = val
                    self.ome_members[label].error[k][l] = err

            logger.info(
                "Matching: computing operators - %d/%d took: %f s",
                k + 1,
                grid_size,
                time.perf_counter() - start_time,
            )

        # closing comment
        logger.info("Matching: Total time %f s", time.perf_counter() - tot_start_time)
