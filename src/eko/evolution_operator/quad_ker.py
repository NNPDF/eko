r"""Integration kernels."""

import enum
import logging

import numba as nb
import numpy as np

import ekore.anomalous_dimensions.polarized.space_like as ad_ps
import ekore.anomalous_dimensions.unpolarized.space_like as ad_us
import ekore.anomalous_dimensions.unpolarized.time_like as ad_ut
import ekore.operator_matrix_elements.polarized.space_like as ome_ps
import ekore.operator_matrix_elements.unpolarized.space_like as ome_us
import ekore.operator_matrix_elements.unpolarized.time_like as ome_ut

from .. import interpolation, mellin
from .. import scale_variations as sv
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


class MatchingMethods(enum.IntEnum):
    """Enumerate matching methods."""

    FORWARD = enum.auto()
    BACKWARD_EXACT = enum.auto()
    BACKWARD_EXPANDED = enum.auto()


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
    backward_method : MatchingMethods
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
    if backward_method is MatchingMethods.BACKWARD_EXPANDED:
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
        if backward_method is MatchingMethods.BACKWARD_EXACT:
            ome = np.linalg.inv(ome)
    return ome


spec = [
    ("is_singlet", nb.boolean),
    ("is_QEDsinglet", nb.boolean),
    ("is_QEDvalence", nb.boolean),
    ("is_log", nb.boolean),
    ("logx", nb.float64),
    ("u", nb.float64),
]


@nb.experimental.jitclass(spec)
class QuadKerBase:
    """Manage the common part of Mellin inversion integral.

    Parameters
    ----------
    u : float
        quad argument
    is_log : boolean
        is a logarithmic interpolation
    logx : float
        Mellin inversion point
    mode0 : int
        first sector element
    """

    def __init__(self, u, is_log, logx, mode0):
        self.is_singlet = mode0 in [100, 21, 90]
        self.is_QEDsinglet = mode0 in [21, 22, 100, 101, 90]
        self.is_QEDvalence = mode0 in [10200, 10204]
        self.is_log = is_log
        self.u = u
        self.logx = logx

    @property
    def path(self):
        """Return the associated instance of :class:`eko.mellin.Path`."""
        if self.is_singlet or self.is_QEDsinglet:
            return mellin.Path(self.u, self.logx, True)
        else:
            return mellin.Path(self.u, self.logx, False)

    @property
    def n(self):
        """Returns the Mellin moment :math:`N`."""
        return self.path.n

    def integrand(
        self,
        areas,
    ):
        """Get transformation to Mellin space integral.

        Parameters
        ----------
        areas : tuple
            basis function configuration

        Returns
        -------
        complex
            common mellin inversion integrand
        """
        if self.logx == 0.0:
            return 0.0
        pj = interpolation.evaluate_grid(self.path.n, self.is_log, self.logx, areas)
        if pj == 0.0:
            return 0.0
        return self.path.prefactor * pj * self.path.jac


@nb.njit(cache=True)
def quad_ker_ad(
    u,
    order,
    mode0,
    mode1,
    ev_method,
    is_log,
    logx,
    areas,
    as_list,
    mu2_from,
    mu2_to,
    a_half,
    alphaem_running,
    nf,
    Lsv,
    ev_op_iterations,
    ev_op_max_order,
    sv_mode,
    is_threshold,
    n3lo_ad_variation,
    is_polarized,
    is_time_like,
    use_fhmruvv,
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
    ev_method : str
        ev_method
    is_log : boolean
        is a logarithmic interpolation
    logx : float
        Mellin inversion point
    areas : tuple
        basis function configuration
    as_list : numpy.ndarray
        list of strong coupling values
    mu2_from : float
        initial value of mu2
    mu2_to : float
        final value of mu2
    aem_list : list
        list of electromagnetic coupling values
    alphaem_running : bool
        whether alphaem is running or not
    nf : int
        number of active flavors
    Lsv : float
        logarithm of the squared ratio of factorization and renormalization scale
    ev_op_iterations : int
        number of evolution steps
    ev_op_max_order : int
        perturbative expansion order of U
    sv_mode: int, `enum.IntEnum`
        scale variation mode, see `eko.scale_variations.Modes`
    is_threshold : boolean
        is this an intermediate threshold operator?
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    is_polarized : boolean
        is polarized evolution ?
    is_time_like : boolean
        is time-like evolution ?
    use_fhmruvv : bool
        if True use the |FHMRUVV| |N3LO| anomalous dimension

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
        ker = quad_ker_qcd(
            ker_base,
            order,
            mode0,
            mode1,
            ev_method,
            as_list[-1],
            as_list[0],
            nf,
            Lsv,
            ev_op_iterations,
            ev_op_max_order,
            sv_mode,
            is_threshold,
            is_polarized,
            is_time_like,
            n3lo_ad_variation,
            use_fhmruvv,
        )
    else:
        ker = quad_ker_qed(
            ker_base,
            order,
            mode0,
            mode1,
            ev_method,
            as_list,
            mu2_from,
            mu2_to,
            a_half,
            alphaem_running,
            nf,
            Lsv,
            ev_op_iterations,
            ev_op_max_order,
            sv_mode,
            is_threshold,
            n3lo_ad_variation,
            use_fhmruvv,
        )

    # recombine everything
    return np.real(ker * integrand)


@nb.njit(cache=True)
def quad_ker_qcd(
    ker_base,
    order,
    mode0,
    mode1,
    ev_method,
    as1,
    as0,
    nf,
    Lsv,
    ev_op_iterations,
    ev_op_max_order,
    sv_mode,
    is_threshold,
    is_polarized,
    is_time_like,
    n3lo_ad_variation,
    use_fhmruvv,
):
    """Raw evolution kernel inside quad.

    Parameters
    ----------
    quad_ker : float
        quad argument
    order : int
        perturbation order
    mode0: int
        pid for first sector element
    mode1 : int
        pid for second sector element
    ev_method : str
        ev_method
    as1 : float
        target coupling value
    as0 : float
        initial coupling value
    nf : int
        number of active flavors
    Lsv : float
        logarithm of the squared ratio of factorization and renormalization scale
    ev_op_iterations : int
        number of evolution steps
    ev_op_max_order : int
        perturbative expansion order of U
    sv_mode: int, `enum.IntEnum`
        scale variation mode, see `eko.scale_variations.Modes`
    is_threshold : boolean
        is this an itermediate threshold operator?
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv : bool
        if True use the |FHMRUVV| |N3LO| anomalous dimensions

    Returns
    -------
    float
        evaluated integration kernel
    """
    # compute the actual evolution kernel for pure QCD
    if ker_base.is_singlet:
        if is_polarized:
            if is_time_like:
                raise NotImplementedError("Polarized, time-like is not implemented")
            else:
                gamma_singlet = ad_ps.gamma_singlet(order, ker_base.n, nf)
        else:
            if is_time_like:
                gamma_singlet = ad_ut.gamma_singlet(order, ker_base.n, nf)
            else:
                gamma_singlet = ad_us.gamma_singlet(
                    order, ker_base.n, nf, n3lo_ad_variation, use_fhmruvv
                )
        # scale var exponentiated is directly applied on gamma
        if sv_mode == sv.Modes.exponentiated:
            gamma_singlet = sv_exponentiated.gamma_variation(
                gamma_singlet, order, nf, Lsv
            )
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
        if is_polarized:
            if is_time_like:
                raise NotImplementedError("Polarized, time-like is not implemented")
            else:
                gamma_ns = ad_ps.gamma_ns(order, mode0, ker_base.n, nf)
        else:
            if is_time_like:
                gamma_ns = ad_ut.gamma_ns(order, mode0, ker_base.n, nf)
            else:
                gamma_ns = ad_us.gamma_ns(
                    order, mode0, ker_base.n, nf, n3lo_ad_variation, use_fhmruvv
                )
        if sv_mode == sv.Modes.exponentiated:
            gamma_ns = sv_exponentiated.gamma_variation(gamma_ns, order, nf, Lsv)
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
    return ker


@nb.njit(cache=True)
def quad_ker_qed(
    ker_base,
    order,
    mode0,
    mode1,
    ev_method,
    as_list,
    mu2_from,
    mu2_to,
    a_half,
    alphaem_running,
    nf,
    Lsv,
    ev_op_iterations,
    ev_op_max_order,
    sv_mode,
    is_threshold,
    n3lo_ad_variation,
    use_fhmruvv,
):
    """Raw evolution kernel inside quad.

    Parameters
    ----------
    ker_base : QuadKerBase
        quad argument
    order : int
        perturbation order
    mode0: int
        pid for first sector element
    mode1 : int
        pid for second sector element
    ev_method : str
        ev_method
    as1 : float
        target coupling value
    as0 : float
        initial coupling value
    mu2_from : float
        initial value of mu2
    mu2_from : float
        final value of mu2
    aem_list : list
        list of electromagnetic coupling values
    alphaem_running : bool
        whether alphaem is running or not
    nf : int
        number of active flavors
    Lsv : float
        logarithm of the squared ratio of factorization and renormalization scale
    ev_op_iterations : int
        number of evolution steps
    ev_op_max_order : int
        perturbative expansion order of U
    sv_mode: int, `enum.IntEnum`
        scale variation mode, see `eko.scale_variations.Modes`
    is_threshold : boolean
        is this an itermediate threshold operator?
    n3lo_ad_variation : tuple
        |N3LO| anomalous dimension variation ``(gg, gq, qg, qq, nsp, nsm, nsv)``
    use_fhmruvv : bool
        if True use the |FHMRUVV| |N3LO| anomalous dimensions

    Returns
    -------
    float
        evaluated integration kernel
    """
    # compute the actual evolution kernel for QEDxQCD
    if ker_base.is_QEDsinglet:
        gamma_s = ad_us.gamma_singlet_qed(
            order, ker_base.n, nf, n3lo_ad_variation, use_fhmruvv
        )
        # scale var exponentiated is directly applied on gamma
        if sv_mode == sv.Modes.exponentiated:
            gamma_s = sv_exponentiated.gamma_variation_qed(
                gamma_s, order, nf, lepton_number(mu2_to), Lsv, alphaem_running
            )
        ker = qed_s.dispatcher(
            order,
            ev_method,
            gamma_s,
            as_list,
            a_half,
            nf,
            ev_op_iterations,
            ev_op_max_order,
        )
        # scale var expanded is applied on the kernel
        # TODO : in this way a_half[-1][1] is the aem value computed in
        # the middle point of the last step. Instead we want aem computed in mu2_final.
        # However the distance between the two is very small and affects only the running aem
        if sv_mode == sv.Modes.expanded and not is_threshold:
            ker = np.ascontiguousarray(
                sv_expanded.singlet_variation_qed(
                    gamma_s, as_list[-1], a_half[-1][1], alphaem_running, order, nf, Lsv
                )
            ) @ np.ascontiguousarray(ker)
        ker = select_QEDsinglet_element(ker, mode0, mode1)
    elif ker_base.is_QEDvalence:
        gamma_v = ad_us.gamma_valence_qed(
            order, ker_base.n, nf, n3lo_ad_variation, use_fhmruvv
        )
        # scale var exponentiated is directly applied on gamma
        if sv_mode == sv.Modes.exponentiated:
            gamma_v = sv_exponentiated.gamma_variation_qed(
                gamma_v, order, nf, lepton_number(mu2_to), Lsv, alphaem_running
            )
        ker = qed_v.dispatcher(
            order,
            ev_method,
            gamma_v,
            as_list,
            a_half,
            nf,
            ev_op_iterations,
            ev_op_max_order,
        )
        # scale var expanded is applied on the kernel
        if sv_mode == sv.Modes.expanded and not is_threshold:
            ker = np.ascontiguousarray(
                sv_expanded.valence_variation_qed(
                    gamma_v, as_list[-1], a_half[-1][1], alphaem_running, order, nf, Lsv
                )
            ) @ np.ascontiguousarray(ker)
        ker = select_QEDvalence_element(ker, mode0, mode1)
    else:
        gamma_ns = ad_us.gamma_ns_qed(
            order, mode0, ker_base.n, nf, n3lo_ad_variation, use_fhmruvv
        )
        # scale var exponentiated is directly applied on gamma
        if sv_mode == sv.Modes.exponentiated:
            gamma_ns = sv_exponentiated.gamma_variation_qed(
                gamma_ns, order, nf, lepton_number(mu2_to), Lsv, alphaem_running
            )
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
    return ker


@nb.njit(cache=True)
def quad_ker_ome(
    u,
    order,
    mode0,
    mode1,
    is_log,
    logx,
    areas,
    a_s,
    nf,
    L,
    sv_mode,
    Lsv,
    backward_method,
    is_msbar,
    is_polarized,
    is_time_like,
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
    a_s : float
        strong coupling, needed only for the exact inverse
    nf: int
        number of active flavor below threshold
    L : float
        :math:``\ln(\mu_F^2 / m_h^2)``
    backward_method : MatchingMethods
        empty or method for inverting the matching condition (exact or expanded)
    is_msbar: bool
        add the |MSbar| contribution
    is_polarized : boolean
        is polarized evolution ?
    is_time_like : boolean
        is time-like evolution ?

    Returns
    -------
    ker : float
        evaluated integration kernel
    """
    ker_base = QuadKerBase(u, is_log, logx, mode0)
    integrand = ker_base.integrand(areas)
    if integrand == 0.0:
        return 0.0
    # compute the ome
    if ker_base.is_singlet or ker_base.is_QEDsinglet:
        indices = {21: 0, 100: 1, 90: 2}
        if is_polarized:
            if is_time_like:
                raise NotImplementedError("Polarized, time-like is not implemented")
            A = ome_ps.A_singlet(order, ker_base.n, nf, L)
        else:
            if is_time_like:
                A = ome_ut.A_singlet(order, ker_base.n, L)
            else:
                A = ome_us.A_singlet(order, ker_base.n, nf, L, is_msbar)
    else:
        indices = {200: 0, 91: 1}
        if is_polarized:
            if is_time_like:
                raise NotImplementedError("Polarized, time-like is not implemented")
            A = ome_ps.A_non_singlet(order, ker_base.n, L)
        else:
            if is_time_like:
                A = ome_ut.A_non_singlet(order, ker_base.n, L)
            else:
                A = ome_us.A_non_singlet(order, ker_base.n, nf, L)

    # correct for scale variations
    if sv_mode == sv.Modes.exponentiated:
        A = sv_exponentiated.gamma_variation(A, order, nf, Lsv)

    # build the expansion in alpha_s depending on the strategy
    ker = build_ome(A, order, a_s, backward_method)

    # select the needed matrix element
    ker = ker[indices[mode0], indices[mode1]]

    # recombine everything
    return np.real(ker * integrand)
