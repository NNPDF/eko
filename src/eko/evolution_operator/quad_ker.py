r"""Integration kernels."""

import logging

import numba as nb
import numpy as np

from .. import interpolation
from .. import scale_variations as sv
from ..io.types import InversionMethod
from ..kernels import non_singlet as ns
from ..kernels import non_singlet_qed as qed_ns
from ..kernels import singlet as s
from ..kernels import singlet_qed as qed_s
from ..kernels import valence_qed as qed_v
from ..matchings import lepton_number
from ..scale_variations import expanded as sv_expanded
from ..scale_variations import exponentiated as sv_exponentiated

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def select_singlet_element(ker, mode0, mode1):
    """Select element of the singlet matrix.

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
    complex
        singlet integration kernel element
    """
    j = 0 if mode0 == 100 else 1
    k = 0 if mode1 == 100 else 1
    return ker[j, k]


CB_SIGNATURE = nb.types.double(
    nb.types.CPointer(nb.types.double),  # re_*_raw
    nb.types.CPointer(nb.types.double),  # im_*_raw
    nb.types.double,  # re_n
    nb.types.double,  # im_n
    nb.types.double,  # re_jac
    nb.types.double,  # im_jac
    nb.types.uintc,  # order_qcd
    nb.types.uintc,  # order_qed
    nb.types.bool_,  # is_singlet
    nb.types.uintc,  # mode0
    nb.types.uintc,  # mode1
    nb.types.uintc,  # nf
    nb.types.bool_,  # is_log
    nb.types.double,  # logx
    nb.types.CPointer(nb.types.double),  # areas_raw
    nb.types.uintc,  # areas_x
    nb.types.uintc,  # areas_y
    nb.types.uintc,  # method_num
    nb.types.double,  # as1
    nb.types.double,  # as0
    nb.types.uintc,  # ev_op_iterations
    nb.types.uintc,  # ev_op_max_order_qcd
    nb.types.uintc,  # sv_mode_num
    nb.types.bool_,  # is_threshold
    nb.types.double,  # Lsv
    nb.types.CPointer(nb.types.double),  # as_list
    nb.types.uintc,  # as_list_len
    nb.types.double,  # mu2_from
    nb.types.double,  # mu2_to
    nb.types.CPointer(nb.types.double),  # a_half
    nb.types.uintc,  # a_half_x
    nb.types.uintc,  # a_half_y
    nb.types.bool_,  # alphaem_running
)


@nb.cfunc(
    CB_SIGNATURE,
    cache=True,
    nopython=True,
)
def cb_quad_ker_qcd(
    re_gamma_raw,
    im_gamma_raw,
    re_n,
    im_n,
    re_jac,
    im_jac,
    order_qcd,
    _order_qed,
    is_singlet,
    mode0,
    mode1,
    nf,
    is_log,
    logx,
    areas_raw,
    areas_x,
    areas_y,
    ev_method,
    as1,
    as0,
    ev_op_iterations,
    ev_op_max_order_qcd,
    sv_mode,
    is_threshold,
    Lsv,
    _as_list,
    _as_list_len,
    _mu2_from,
    _mu2_to,
    _a_half,
    _a_half_x,
    _a_half_y,
    _alphaem_running,
):
    """C Callback inside integration kernel."""
    # recover complex variables
    n = re_n + im_n * 1j
    jac = re_jac + im_jac * 1j
    # combute basis functions
    areas = nb.carray(areas_raw, (areas_x, areas_y))
    pj = interpolation.evaluate_grid(n, is_log, logx, areas)
    order = (order_qcd, 0)
    ev_op_max_order = (ev_op_max_order_qcd, 0)
    if is_singlet:
        # reconstruct singlet matrices
        re_gamma_singlet = nb.carray(re_gamma_raw, (order_qcd, 2, 2))
        im_gamma_singlet = nb.carray(im_gamma_raw, (order_qcd, 2, 2))
        gamma_singlet = re_gamma_singlet + im_gamma_singlet * 1j
        if sv_mode == sv.Modes.exponentiated:
            gamma_singlet = sv_exponentiated.gamma_variation(
                gamma_singlet, order, nf, Lsv
            )
        # construct eko
        ker = s.dispatcher(
            order,
            ev_method,
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
                sv_expanded.singlet_variation(gamma_singlet, as1, order, nf, Lsv, dim=2)
            ) @ np.ascontiguousarray(ker)
        ker = select_singlet_element(ker, mode0, mode1)
    else:
        # construct non-singlet matrices
        re_gamma_ns = nb.carray(re_gamma_raw, order_qcd)
        im_gamma_ns = nb.carray(im_gamma_raw, order_qcd)
        gamma_ns = re_gamma_ns + im_gamma_ns * 1j
        if sv_mode == sv.Modes.exponentiated:
            gamma_ns = sv_exponentiated.gamma_variation(gamma_ns, order, nf, Lsv)
        # construct eko
        ker = ns.dispatcher(
            order,
            ev_method,
            gamma_ns,
            as1,
            as0,
            nf,
        )
        if sv_mode == sv.Modes.expanded and not is_threshold:
            ker = sv_expanded.non_singlet_variation(gamma_ns, as1, order, nf, Lsv) * ker
    # recombine everything
    res = ker * pj * jac
    return np.real(res)


@nb.njit(cache=True)
def build_ome(A, matching_order, a_s, backward_method):
    r"""Construct the matching expansion in :math:`a_s` with the appropriate
    method.

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
    ome = np.eye(len(A[0]), dtype=np.complex128)
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
    CB_SIGNATURE,
    cache=True,
    nopython=True,
)
def cb_quad_ker_ome(
    re_ome_raw,
    im_ome_raw,
    re_n,
    im_n,
    re_jac,
    im_jac,
    order_qcd,
    _order_qed,
    is_singlet,
    mode0,
    mode1,
    nf,
    is_log,
    logx,
    areas_raw,
    areas_x,
    areas_y,
    backward_method,
    as1,
    _as0,
    _ev_op_iterations,
    _ev_op_max_order_qcd,
    sv_mode,
    _is_threshold,
    Lsv,
    _as_list,
    _as_list_len,
    _mu2_from,
    _mu2_to,
    _a_half,
    _a_half_x,
    _a_half_y,
    _alphaem_running,
):
    """C Callback inside integration kernel."""
    # recover complex variables
    n = re_n + im_n * 1j
    jac = re_jac + im_jac * 1j
    # compute basis functions
    areas = nb.carray(areas_raw, (areas_x, areas_y))
    pj = interpolation.evaluate_grid(n, is_log, logx, areas)
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
        im_ome_ns = nb.carray(im_ome_raw, (order_qcd, 2, 2))
        A = re_ome_ns + im_ome_ns * 1j

    # correct for scale variations
    if sv_mode == sv.Modes.exponentiated:
        A = sv.exponentiated.gamma_variation(A, order, nf, Lsv)

    # build the expansion in alpha_s depending on the strategy
    ker = build_ome(A, order, as1, backward_method)

    # select the needed matrix element
    ker = ker[indices[mode0], indices[mode1]]

    # recombine everything
    res = ker * pj * jac
    return np.real(res)


@nb.njit(cache=True)
def select_QEDsinglet_element(ker, mode0, mode1):
    """Select element of the QEDsinglet matrix.

    Parameters
    ----------
    ker : numpy.ndarray
        QEDsinglet integration kernel
    mode0 : int
        id for first sector element
    mode1 : int
        id for second sector element
    Returns
    -------
    ker : complex
        QEDsinglet integration kernel element
    """
    if mode0 == 21:
        index1 = 0
    elif mode0 == 22:
        index1 = 1
    elif mode0 == 100:
        index1 = 2
    else:
        index1 = 3
    if mode1 == 21:
        index2 = 0
    elif mode1 == 22:
        index2 = 1
    elif mode1 == 100:
        index2 = 2
    else:
        index2 = 3
    return ker[index1, index2]


@nb.njit(cache=True)
def select_QEDvalence_element(ker, mode0, mode1):
    """Select element of the QEDvalence matrix.

    Parameters
    ----------
    ker : numpy.ndarray
        QEDvalence integration kernel
    mode0 : int
        id for first sector element
    mode1 : int
        id for second sector element
    Returns
    -------
    ker : complex
        QEDvalence integration kernel element
    """
    index1 = 0 if mode0 == 10200 else 1
    index2 = 0 if mode1 == 10200 else 1
    return ker[index1, index2]


@nb.cfunc(
    CB_SIGNATURE,
    cache=True,
    nopython=True,
)
def cb_quad_ker_qed(
    re_gamma_raw,
    im_gamma_raw,
    re_n,
    im_n,
    re_jac,
    im_jac,
    order_qcd,
    order_qed,
    is_singlet,
    mode0,
    mode1,
    nf,
    is_log,
    logx,
    areas_raw,
    areas_x,
    areas_y,
    ev_method,
    _as1,
    _as0,
    ev_op_iterations,
    ev_op_max_order_qcd,
    sv_mode,
    is_threshold,
    Lsv,
    as_list_raw,
    as_list_len,
    mu2_from,
    mu2_to,
    a_half_raw,
    a_half_x,
    a_half_y,
    alphaem_running,
):
    """C Callback inside integration kernel."""
    # recover complex variables
    n = re_n + im_n * 1j
    jac = re_jac + im_jac * 1j
    # compute basis functions
    areas = nb.carray(areas_raw, (areas_x, areas_y))
    pj = interpolation.evaluate_grid(n, is_log, logx, areas)
    order = (order_qcd, order_qed)
    ev_op_max_order = (ev_op_max_order_qcd, order_qed)
    is_valence = (mode0 == 10200) or (mode0 == 10204)

    as_list = nb.carray(as_list_raw, as_list_len)
    a_half = nb.carray(a_half_raw, (a_half_x, a_half_y))

    if is_singlet:
        # reconstruct singlet matrices
        re_gamma_singlet = nb.carray(re_gamma_raw, (order_qcd + 1, order_qed + 1, 4, 4))
        im_gamma_singlet = nb.carray(im_gamma_raw, (order_qcd + 1, order_qed + 1, 4, 4))
        gamma_singlet = re_gamma_singlet + im_gamma_singlet * 1j

        # scale var exponentiated is directly applied on gamma
        if sv_mode == sv.Modes.exponentiated:
            gamma_singlet = sv.exponentiated.gamma_variation_qed(
                gamma_singlet, order, nf, lepton_number(mu2_to), Lsv, alphaem_running
            )

        ker = qed_s.dispatcher(
            order,
            ev_method,
            gamma_singlet,
            as_list,
            a_half,
            nf,
            ev_op_iterations,
            ev_op_max_order,
        )
        if sv_mode == sv.Modes.expanded and not is_threshold:
            ker = np.ascontiguousarray(
                sv.expanded.singlet_variation_qed(
                    gamma_singlet,
                    as_list[-1],
                    a_half[-1][1],
                    alphaem_running,
                    order,
                    nf,
                    Lsv,
                )
            ) @ np.ascontiguousarray(ker)
        ker = select_QEDsinglet_element(ker, mode0, mode1)

    elif is_valence:
        # reconstruct valence matrices
        re_gamma_valence = nb.carray(re_gamma_raw, (order_qcd + 1, order_qed + 1, 2, 2))
        im_gamma_valence = nb.carray(im_gamma_raw, (order_qcd + 1, order_qed + 1, 2, 2))
        gamma_valence = re_gamma_valence + im_gamma_valence * 1j

        if sv_mode == sv.Modes.exponentiated:
            gamma_valence = sv.exponentiated.gamma_variation_qed(
                gamma_valence, order, nf, lepton_number(mu2_to), Lsv, alphaem_running
            )
        ker = qed_v.dispatcher(
            order,
            ev_method,
            gamma_valence,
            as_list,
            a_half,
            nf,
            ev_op_iterations,
            ev_op_max_order,
        )
        # scale var expanded is applied on the kernel
        if sv_mode == sv.Modes.expanded and not is_threshold:
            ker = np.ascontiguousarray(
                sv.expanded.valence_variation_qed(
                    gamma_valence,
                    as_list[-1],
                    a_half[-1][1],
                    alphaem_running,
                    order,
                    nf,
                    Lsv,
                )
            ) @ np.ascontiguousarray(ker)
        ker = select_QEDvalence_element(ker, mode0, mode1)

    else:
        # construct non-singlet matrices
        re_gamma_ns = nb.carray(re_gamma_raw, (order_qcd + 1, order_qed + 1))
        im_gamma_ns = nb.carray(im_gamma_raw, (order_qcd + 1, order_qed + 1))
        gamma_ns = re_gamma_ns + im_gamma_ns * 1j
        if sv_mode == sv.Modes.exponentiated:
            gamma_ns = sv_exponentiated.gamma_variation_qed(
                gamma_ns, order, nf, lepton_number(mu2_to), Lsv, alphaem_running
            )
        # construct eko
        ker = qed_ns.dispatcher(
            order,
            ev_method,
            gamma_ns,
            as_list,
            a_half[:, 1],
            alphaem_running,
            nf,
            ev_op_iterations,
            mu2_from,
            mu2_to,
        )
        if sv_mode == sv.Modes.expanded and not is_threshold:
            ker = (
                sv_expanded.non_singlet_variation_qed(
                    gamma_ns,
                    as_list[-1],
                    a_half[-1][1],
                    alphaem_running,
                    order,
                    nf,
                    Lsv,
                )
                * ker
            )
    # recombine everything
    res = ker * pj * jac
    return np.real(res)
