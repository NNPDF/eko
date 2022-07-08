# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

from .. import beta
from . import evolution_integrals as ei
from . import non_singlet, utils


@nb.njit(cache=True)
def dispatcher(
    order, method, gamma_ns, a1, a0, nf, ev_op_iterations
):  # pylint: disable=too-many-return-statements
    """
    Determine used kernel and call it.

    In LO we always use the exact solution.

    Parameters
    ----------
        order : tuple(int,int)
            perturbation order
        method : str
            method
        gamma_ns : numpy.ndarray
            non-singlet anomalous dimensions
        a1 : float
            target coupling value
        a0 : float
            initial coupling value
        nf : int
            number of active flavors
        ev_op_iterations : int
            number of evolution steps

    Returns
    -------
        e_ns : complex
            non-singlet EKO
    """
    # use always exact in LO
    if order[1] == 0:
        return non_singlet.dispatcher(
            order, method, gamma_ns, a1, a0, nf, ev_op_iterations
        )
    raise NotImplementedError("Selected order is not implemented")
