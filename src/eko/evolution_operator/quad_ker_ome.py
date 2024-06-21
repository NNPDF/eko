r"""Integration kernels."""

import logging

import numba as nb
import numpy as np
from scipy import LowLevelCallable, integrate

from .. import basis_rotation as br
from .. import interpolation
from .. import scale_variations as sv
from ..io.types import InversionMethod
from ..kernels import non_singlet as ns
from ..kernels import singlet as s
from ..matchings import Segment
from ..member import OpMember

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def build_ome(A, matching_order, a_s, backward_method):
    r"""Construct the matching expansion in :math:`a_s` with the appropriate method.

    Parameters
    ----------
    A : numpy.ndarray
        list of |OME|
    matching_order : tuple(int,int)
        perturbation matching order
    a_s : float
        strong coupling, needed only for the exact inverse
    backward_method : InversionMethod or None
        empty or method for inverting the matching condition (exact or expanded)

    Returns
    -------
    ome : numpy.ndarray
        matching operator matrix

    """
    # to get the inverse one can use this FORM snippet
    # Symbol a;
    # NTensor c,d,e;
    # Local x=-(a*c+a**2* d + a**3 * e);
    # Local bi = 1+x+x**2+x**3;
    # Print;
    # .end
    ome = np.eye(len(A[0]), dtype=np.complex_)
    A = A[:, :, :]
    A = np.ascontiguousarray(A)

    if backward_method is InversionMethod.EXPANDED:
        # expended inverse
        if matching_order[0] >= 1:
            ome -= a_s * A[0]
        if matching_order[0] >= 2:
            ome += a_s**2 * (-A[1] + A[0] @ A[0])
        if matching_order[0] >= 3:
            ome += a_s**3 * (-A[2] + A[0] @ A[1] + A[1] @ A[0] - A[0] @ A[0] @ A[0])
    else:
        # forward or exact inverse
        if matching_order[0] >= 1:
            ome += a_s * A[0]
        if matching_order[0] >= 2:
            ome += a_s**2 * A[1]
        if matching_order[0] >= 3:
            ome += a_s**3 * A[2]
        # need inverse exact ?  so add the missing pieces
        if backward_method is InversionMethod.EXACT:
            ome = np.linalg.inv(ome)
    return ome


@nb.cfunc(
    nb.types.double(
        nb.types.CPointer(nb.types.double),  # re_gamma_ns_raw
        nb.types.CPointer(nb.types.double),  # im_gamma_ns_raw
        nb.types.double,  # re_n
        nb.types.double,  # im_n
        nb.types.double,  # re_jac
        nb.types.double,  # im_jac
        nb.types.uint,  # order_qcd
        nb.types.bool_,  # is_singlet
        nb.types.uint,  # mode0
        nb.types.uint,  # mode1
        nb.types.uint,  # nf
        nb.types.bool_,  # is_log
        nb.types.double,  # logx
        nb.types.CPointer(nb.types.double),  # areas_raw
        nb.types.uint,  # areas_x
        nb.types.uint,  # areas_y
        nb.types.double,  # L
        nb.types.uint,  # method_num
        nb.types.double,  # as1
        nb.types.double,  # as0
        nb.types.uint,  # ev_op_iterations
        nb.types.uint,  # ev_op_max_order_qcd
        nb.types.uint,  # sv_mode_num
        nb.types.bool_,  # is_threshold
    ),
    cache=True,
    nopython=True,
)
def cb_quad_ker_ome(
    re_gamma_raw,
    im_gamma_raw,
    re_n,
    im_n,
    re_jac,
    im_jac,
    order_qcd,
    is_singlet,
    mode0,
    mode1,
    nf,
    is_log,
    logx,
    areas_raw,
    areas_x,
    areas_y,
    L,
    _method_num,  # dummy variable
    as1,
    as0,
    ev_op_iterations,  # dummy variable
    ev_op_max_order_qcd,  # dummy variable
    _sv_mode_num,  # dummy variable
    is_threshold,  # dummy variable
):
    """C Callback inside integration kernel."""
    # recover complex variables
    n = re_n + im_n * 1j
    jac = re_jac + im_jac * 1j
    # compute basis functions
    areas = nb.carray(areas_raw, (areas_x, areas_y))
    pj = interpolation.evaluate_grid(n, is_log, logx, areas)
    # TODO recover parameters
    sv_mode = sv.Modes.exponentiated
    order = (order_qcd, 0)
    if is_singlet:
        indices = {21: 0, 100: 1, 90: 2}
        # reconstruct singlet matrices
        re_ome_singlet = nb.carray(re_ome_raw, (order_qcd, 3, 3))
        im_ome_singlet = nb.carray(im_ome_raw, (order_qcd, 3, 3))
        A = re_ome_singlet + im_ome_singlet * 1j
    else:
        indices = {200: 0, 91: 1}
        # construct non-singlet matrices
        re_ome_ns = nb.carray(re_ome_raw, (order_qcd, 2, 2))
        im_ome_ns = nb.carray(im_ome_raw, (order_qcd, 2, 2, 2))
        A = re_ome_ns + im_ome_ns * 1j

    # correct for scale variations
    if sv_mode == sv.Modes.exponentiated:
        A = sv.exponentiated.gamma_variation(A, order, nf, L)

    # TODO recover InversionMethod
    backward_method = "exact"

    # build the expansion in alpha_s depending on the strategy
    ker = build_ome(A, order, a_s, backward_method)

    # select the needed matrix element
    ker = ker[indices[mode0], indices[mode1]]

    # recombine everything
    res = ker * pj * jac
    return np.real(res)
