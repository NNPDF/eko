"""SCET I kernels"""

import copy
import functools
import logging

import numba as nb
import numpy as np

import ekore.scet_I as scet_I

from .. import basis_rotation as br
from .. import scale_variations as sv
from ..matchings import Segment
from . import Operator, QuadKerBase

logger = logging.getLogger(__name__)

@nb.njit(cache=True)
def quad_ker(
    u,
    order,
    space,
    mode0,
    mode1,
    is_log,
    logx,
    areas,
):
    r"""Raw kernel inside quad.

    Parameters
    ----------
    u : float
        quad argument
    order : tuple(int,int)
        perturbation matching order
    mode0 : int
        pid for first element in the singlet sector
    mode1 : int
        pid for second element in the singlet sector
    is_log : boolean
        logarithmic interpolation
    logx : float
        Mellin inversion point
    areas : tuple
        basis function configuration
    
    Returns
    -------
    ker : float
        evaluated integration kernel

    """
    ker_base = QuadKerBase(u, is_log, logx, mode0)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0
    indices = {21: 0, 1: 1, -1: 2, 2: 3, -2: 4}
    A = scet_I.SCET_I_entry(order, space, ker_base.n)
    # select the needed matrix element
    ker = A[indices[mode0], indices[mode1]]

    # recombine everything
    return np.real(ker * integrand)


class SCET_I(Operator):
    r"""
    Internal representation of a single |SCET1| mathcing kernel.

    The actual matrices are computed upon calling :meth:`compute`.

    Parameters
    ----------
    config : dict
        configuration
    managers : dict
        managers
    order: tuple (int, int)
        order in alpha_s and L
    """

    log_label = "Scet_I"
    full_labels = br.scet_labels

    def __init__(self, config, managers, order, space):
        super().__init__(config, managers, Segment(origin=1, target=1, nf=5))
        # order (alpha_s, L) of the SCET kernel
        self.order_scet = order
        self.space = space

    @property
    def labels(self):
        """Necessary sector labels to compute.

        Returns
        -------
        list(str)
            sector labels
        """
        labels = []
        
        
        labels.extend(
            [
                *br.scet_labels,
            ]
        )
            
        return labels

    def quad_ker(self, label, logx, areas):
        """Return partially initialized integrand function.

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
            order=self.order_scet,
            space=self.space,
            mode0=label[0],
            mode1=label[1],
            is_log=self.int_disp.log,
            logx=logx,
            areas=areas,
        )

    @property
    def a_s(self):
        """Return the computed values for :math:`a_s`.

        Note that here you need to use :math:`a_s^{n_f+1}`
        """
        sc = self.managers["couplings"]
        return sc.a_s(
            self.q2_from
            * (self.xif2 if self.sv_mode == sv.Modes.exponentiated else 1.0),
            nf_to=self.nf + 1,
        )

    def compute(self):
        """Compute the actual operators (i.e. run the integrations)."""
        self.initialize_op_members()
        self.integrate()
