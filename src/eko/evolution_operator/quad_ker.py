r"""Integration kernels."""

import logging

import numba as nb
import numpy as np

from .. import interpolation
from .. import scale_variations as sv
from ..kernels import non_singlet as ns
from ..kernels import singlet as s
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
def cb_quad_ker_qcd(
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
    _method_num,
    as1,
    as0,
    ev_op_iterations,
    ev_op_max_order_qcd,
    _sv_mode_num,
    is_threshold,
):
    """C Callback inside integration kernel."""
    # recover complex variables
    n = re_n + im_n * 1j
    jac = re_jac + im_jac * 1j
    # combute basis functions
    areas = nb.carray(areas_raw, (areas_x, areas_y))
    pj = interpolation.evaluate_grid(n, is_log, logx, areas)
    # TODO recover parameters
    method = "iterate-exact"
    sv_mode = sv.Modes.exponentiated
    order = (order_qcd, 0)
    ev_op_max_order = (ev_op_max_order_qcd, 0)
    if is_singlet:
        # reconstruct singlet matrices
        re_gamma_singlet = nb.carray(re_gamma_raw, (order_qcd, 2, 2))
        im_gamma_singlet = nb.carray(im_gamma_raw, (order_qcd, 2, 2))
        gamma_singlet = re_gamma_singlet + im_gamma_singlet * 1j
        if sv_mode == sv.Modes.exponentiated:
            gamma_singlet = sv_exponentiated.gamma_variation(
                gamma_singlet, order, nf, L
            )
        # construct eko
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
                sv_expanded.singlet_variation(gamma_singlet, as1, order, nf, L, dim=2)
            ) @ np.ascontiguousarray(ker)
        ker = select_singlet_element(ker, mode0, mode1)
    else:
        # construct non-singlet matrices
        re_gamma_ns = nb.carray(re_gamma_raw, order_qcd)
        im_gamma_ns = nb.carray(im_gamma_raw, order_qcd)
        gamma_ns = re_gamma_ns + im_gamma_ns * 1j
        if sv_mode == sv.Modes.exponentiated:
            gamma_ns = sv_exponentiated.gamma_variation(gamma_ns, order, nf, L)
        # construct eko
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
            ker = sv_expanded.non_singlet_variation(gamma_ns, as1, order, nf, L) * ker
    # recombine everything
    res = ker * pj * jac
    return np.real(res)


# from ..kernels import singlet_qed as qed_s
# from ..kernels import non_singlet_qed as qed_ns
# from ..kernels import valence_qed as qed_v

# @nb.njit(cache=True)
# def select_QEDsinglet_element(ker, mode0, mode1):
#     """Select element of the QEDsinglet matrix.

#     Parameters
#     ----------
#     ker : numpy.ndarray
#         QEDsinglet integration kernel
#     mode0 : int
#         id for first sector element
#     mode1 : int
#         id for second sector element
#     Returns
#     -------
#     ker : complex
#         QEDsinglet integration kernel element
#     """
#     if mode0 == 21:
#         index1 = 0
#     elif mode0 == 22:
#         index1 = 1
#     elif mode0 == 100:
#         index1 = 2
#     else:
#         index1 = 3
#     if mode1 == 21:
#         index2 = 0
#     elif mode1 == 22:
#         index2 = 1
#     elif mode1 == 100:
#         index2 = 2
#     else:
#         index2 = 3
#     return ker[index1, index2]


# @nb.njit(cache=True)
# def select_QEDvalence_element(ker, mode0, mode1):
#     """
#     Select element of the QEDvalence matrix.

#     Parameters
#     ----------
#     ker : numpy.ndarray
#         QEDvalence integration kernel
#     mode0 : int
#         id for first sector element
#     mode1 : int
#         id for second sector element
#     Returns
#     -------
#     ker : complex
#         QEDvalence integration kernel element
#     """
#     index1 = 0 if mode0 == 10200 else 1
#     index2 = 0 if mode1 == 10200 else 1
#     return ker[index1, index2]


# @nb.njit(cache=True)
# def quad_ker_qed(
#     ker_base,
#     order,
#     mode0,
#     mode1,
#     method,
#     as_list,
#     mu2_from,
#     mu2_to,
#     a_half,
#     alphaem_running,
#     nf,
#     L,
#     ev_op_iterations,
#     ev_op_max_order,
#     sv_mode,
#     is_threshold,
# ):
#     """Raw evolution kernel inside quad.

#     Parameters
#     ----------
#     ker_base : QuadKerBase
#         quad argument
#     order : int
#         perturbation order
#     mode0: int
#         pid for first sector element
#     mode1 : int
#         pid for second sector element
#     method : str
#         method
#     as1 : float
#         target coupling value
#     as0 : float
#         initial coupling value
#     mu2_from : float
#         initial value of mu2
#     mu2_from : float
#         final value of mu2
#     aem_list : list
#         list of electromagnetic coupling values
#     alphaem_running : bool
#         whether alphaem is running or not
#     nf : int
#         number of active flavors
#     L : float
#         logarithm of the squared ratio of factorization and renormalization scale
#     ev_op_iterations : int
#         number of evolution steps
#     ev_op_max_order : int
#         perturbative expansion order of U
#     sv_mode: int, `enum.IntEnum`
#         scale variation mode, see `eko.scale_variations.Modes`
#     is_threshold : boolean
#         is this an itermediate threshold operator?

#     Returns
#     -------
#     float
#         evaluated integration kernel
#     """
#     # compute the actual evolution kernel for QEDxQCD
#     if ker_base.is_QEDsinglet:
#         gamma_s = ad_us.gamma_singlet_qed(order, ker_base.n, nf)
#         # scale var exponentiated is directly applied on gamma
#         if sv_mode == sv.Modes.exponentiated:
#             gamma_s = sv.exponentiated.gamma_variation_qed(
#                 gamma_s, order, nf, L, alphaem_running
#             )
#         ker = qed_s.dispatcher(
#             order,
#             method,
#             gamma_s,
#             as_list,
#             a_half,
#             nf,
#             ev_op_iterations,
#             ev_op_max_order,
#         )
#         # scale var expanded is applied on the kernel
#         # TODO : in this way a_half[-1][1] is the aem value computed in
#         # the middle point of the last step. Instead we want aem computed in mu2_final.
#         # However the distance between the two is very small and affects only the running aem
#         if sv_mode == sv.Modes.expanded and not is_threshold:
#             ker = np.ascontiguousarray(
#                 sv.expanded.singlet_variation_qed(
#                     gamma_s, as_list[-1], a_half[-1][1], alphaem_running, order, nf, L
#                 )
#             ) @ np.ascontiguousarray(ker)
#         ker = select_QEDsinglet_element(ker, mode0, mode1)
#     elif ker_base.is_QEDvalence:
#         gamma_v = ad_us.gamma_valence_qed(order, ker_base.n, nf)
#         # scale var exponentiated is directly applied on gamma
#         if sv_mode == sv.Modes.exponentiated:
#             gamma_v = sv.exponentiated.gamma_variation_qed(
#                 gamma_v, order, nf, L, alphaem_running
#             )
#         ker = qed_v.dispatcher(
#             order,
#             method,
#             gamma_v,
#             as_list,
#             a_half,
#             nf,
#             ev_op_iterations,
#             ev_op_max_order,
#         )
#         # scale var expanded is applied on the kernel
#         if sv_mode == sv.Modes.expanded and not is_threshold:
#             ker = np.ascontiguousarray(
#                 sv.expanded.valence_variation_qed(
#                     gamma_v, as_list[-1], a_half[-1][1], alphaem_running, order, nf, L
#                 )
#             ) @ np.ascontiguousarray(ker)
#         ker = select_QEDvalence_element(ker, mode0, mode1)
#     else:
#         gamma_ns = ad_us.gamma_ns_qed(order, mode0, ker_base.n, nf)
#         # scale var exponentiated is directly applied on gamma
#         if sv_mode == sv.Modes.exponentiated:
#             gamma_ns = sv.exponentiated.gamma_variation_qed(
#                 gamma_ns, order, nf, L, alphaem_running
#             )
#         ker = qed_ns.dispatcher(
#             order,
#             method,
#             gamma_ns,
#             as_list,
#             a_half[:, 1],
#             alphaem_running,
#             nf,
#             ev_op_iterations,
#             mu2_from,
#             mu2_to,
#         )
#         if sv_mode == sv.Modes.expanded and not is_threshold:
#             ker = (
#                 sv.expanded.non_singlet_variation_qed(
#                     gamma_ns, as_list[-1], a_half[-1][1], alphaem_running, order, nf, L
#                 )
#                 * ker
#             )
#     return ker
