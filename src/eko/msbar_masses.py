r"""|RGE| for the |MSbar| masses."""

from typing import List

import numba as nb
import numpy as np
from scipy import integrate, optimize

from .basis_rotation import quark_names
from .beta import b_qcd, beta_qcd
from .couplings import Couplings, invert_matching_coeffs
from .gamma import gamma
from .io.types import FlavorsNumber, Order
from .matchings import Atlas, flavor_shift, is_downward_path
from .quantities.couplings import CouplingEvolutionMethod, CouplingsInfo
from .quantities.heavy_quarks import HeavyQuarkMasses, QuarkMassRef, QuarkMassScheme


def ker_exact(a0, a1, order, nf):
    r"""Provide exact |MSbar| |RGE| kernel.

    Parameters
    ----------
    a0 : float
        strong coupling at the initial scale
    a1 : float
        strong coupling at the final scale
    oreder : tuple(int,int)
        perturbative order
    nf : int
        number of active flavours

    Returns
    -------
    float
        Exact |MSbar| kernel:

        .. math::
            k_{exact} = e^{-\int_{a_s(\mu_{h,0}^2)}^{a_s(\mu^2)}\gamma_m(a_s)/ \beta(a_s)da_s}
    """
    b_vec = [beta_qcd((2, 0), nf)]
    g_vec = [gamma(1, nf)]
    if order[0] >= 2:
        b_vec.append(beta_qcd((3, 0), nf))
        g_vec.append(gamma(2, nf))
    if order[0] >= 3:
        b_vec.append(beta_qcd((4, 0), nf))
        g_vec.append(gamma(3, nf))
    if order[0] >= 4:
        b_vec.append(beta_qcd((5, 0), nf))
        g_vec.append(gamma(4, nf))

    # quad ker
    def integrand(a, b_vec, g_vec):
        # minus sign goes away
        fgamma = np.sum([a**k * b for k, b in enumerate(g_vec)])
        fbeta = a * np.sum([a**k * b for k, b in enumerate(b_vec)])
        return fgamma / fbeta

    res = integrate.quad(
        integrand,
        a0,
        a1,
        args=(b_vec, g_vec),
        epsabs=1e-12,
        epsrel=1e-5,
        limit=100,
        full_output=1,
    )
    val, _ = res[:2]
    return np.exp(val)


@nb.njit(cache=True)
def ker_expanded(a0, a1, order, nf):
    r"""Provide expanded |MSbar| |RGE| kernel.

    Parameters
    ----------
    a0 : float
        strong coupling at the initial scale
    a1 : float
        strong coupling at the final scale
    order : tuple(int,int)
        perturbative order
    nf : int
        number of active flavours

    Returns
    -------
    float
        Expanded |MSbar| kernel:

        .. math::
            k_{expanded} &= \left (\frac{a_s(\mu^2)}{a_s(\mu_{h,0}^2)} \right )^{c_0}
            \frac{j_{exp}(a_s(\mu^2))}{j_{exp}(a_s(\mu_{h,0}^2))} \\
            j_{exp}(a_s) &= 1 + a_s \left [ c_1 - b_1 c_0 \right ] \\
            & + \frac{a_s^2}{2}
            \left [c_2 - c_1 b_1 - b_2 c_0 + b_1^2 c_0 + (c_1 - b_1 c_0)^2 \right ] \\
            & + \frac{a_s^3}{6} [ -2 b_3 c_0 - b_1^3 c_0 (1 + c_0) (2 + c_0) - 2 b_2 c_1 \\
            & - 3 b_2 c_0 c_1 + b_1^2 (2 + 3 c_0 (2 + c_0)) c_1 + c_1^3 + 3 c_1 c_2 \\
            & + b_1 (b_2 c_0 (4 + 3 c_0) - 3 (1 + c_0) c_1^2 - (2 + 3 c_0) c_2) + 2 c_3 ]
    """
    b0 = beta_qcd((2, 0), nf)
    c0 = gamma(1, nf) / b0
    ev_mass = np.power(a1 / a0, c0)
    num = 1.0
    den = 1.0
    if order[0] >= 2:
        b1 = b_qcd((3, 0), nf)
        c1 = gamma(2, nf) / b0
        u = c1 - b1 * c0
        num += a1 * u
        den += a0 * u
    if order[0] >= 3:
        b2 = b_qcd((4, 0), nf)
        c2 = gamma(3, nf) / b0
        u = (c2 - c1 * b1 - b2 * c0 + b1**2 * c0 + (c1 - b1 * c0) ** 2) / 2.0
        num += a1**2 * u
        den += a0**2 * u
    if order[0] >= 4:
        b3 = b_qcd((5, 0), nf)
        c3 = gamma(4, nf) / b0
        u = (
            1
            / 6
            * (
                -2 * b3 * c0
                - b1**3 * c0 * (1 + c0) * (2 + c0)
                - 2 * b2 * c1
                - 3 * b2 * c0 * c1
                + b1**2 * (2 + 3 * c0 * (2 + c0)) * c1
                + c1**3
                + 3 * c1 * c2
                + b1
                * (b2 * c0 * (4 + 3 * c0) - 3 * (1 + c0) * c1**2 - (2 + 3 * c0) * c2)
                + 2 * c3
            )
        )
        num += a1**3 * u
        den += a0**3 * u
    return ev_mass * num / den


def ker_dispatcher(q2_to, q2m_ref, strong_coupling, xif2, nf):
    r"""Select the |MSbar| kernel and compute the strong coupling values.

    Parameters
    ----------
    q2_to : float
        final scale
    q2m_ref : float
        initial scale
    strong_coupling : eko.strong_coupling.StrongCoupling
        Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
        any q
    xif2 : float
        factorization to renormalization scale ratio
    nf : int
        number of active flavours

    Returns
    -------
    ker :
        Expanded or exact |MSbar| kernel
    """
    a0 = strong_coupling.a(q2m_ref * xif2, nf)[0]
    a1 = strong_coupling.a(q2_to * xif2, nf)[0]
    method = strong_coupling.method
    order = strong_coupling.order
    if method == "expanded":
        return ker_expanded(a0, float(a1), order, nf)
    return ker_exact(a0, a1, order, nf)


def compute_matching_coeffs_up(nf: int):
    """Upward |MSbar| matching coefficients.

    Used at threshold when moving to a regime with *more* flavors
    :cite:`Liu:2015fxa`.

    Note :cite:`Liu:2015fxa` (eq 5.) reports the backward relation,
    multiplied by a factor of 4 (and 4^2 ...)

    Parameters
    ----------
    nf : int
        number of active flavors in the lower patch

    Returns
    -------
    dict
        downward matching coefficient matrix
    """
    matching_coeffs_up = np.zeros((4, 4))
    matching_coeffs_up[2, 0] = -89.0 / 27.0
    matching_coeffs_up[2, 1] = 20.0 / 9.0
    matching_coeffs_up[2, 2] = -4.0 / 3.0

    matching_coeffs_up[3, 0] = -118.248 - 1.58257 * nf
    matching_coeffs_up[3, 1] = 71.7887 + 7.85185 * nf
    matching_coeffs_up[3, 2] = -700.0 / 27.0
    matching_coeffs_up[3, 3] = -232.0 / 27.0 + 16.0 / 27.0 * nf

    return matching_coeffs_up


def compute_matching_coeffs_down(nf: int):
    """Downward |MSbar| matching coefficients.

    Used at threshold when moving to a regime with *less* flavors
    :cite:`Liu:2015fxa` :eqref:`5`.

    Parameters
    ----------
    nf : int
        number of active flavors in the lower patch

    Returns
    -------
    dict
        downward matching coefficient matrix
    """
    c_up = compute_matching_coeffs_up(nf)
    return invert_matching_coeffs(c_up)


def solve(m2_ref, q2m_ref, strong_coupling, nf_ref, xif2):
    r"""Compute the |MSbar| masses.

    Solves the equation :math:`m_{\overline{MS}}(m) = m` for a fixed number of
    nf.

    Parameters
    ----------
    m2_ref : float
        squared initial mass reference
    q2m_ref : float
        squared initial scale
    strong_coupling : eko.strong_coupling.StrongCoupling
        Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
        any q
    nf_ref : int
        number of active flavours at the scale q2m_ref, where the solution is searched
    xif2 : float
        :math:`\mu_F^2/\mu_R^2`

    Returns
    -------
    m2 : float
        :math:`m_{\overline{MS}}(\mu_2)^2`
    """

    def rge(m2, q2m_ref, strong_coupling, xif2, nf_ref):
        return (
            m2_ref * ker_dispatcher(m2, q2m_ref, strong_coupling, xif2, nf_ref) ** 2
            - m2
        )

    msbar_mass = optimize.fsolve(
        rge, q2m_ref, args=(q2m_ref, strong_coupling, xif2, nf_ref)
    )
    return float(msbar_mass)


def evolve(
    m2_ref,
    q2m_ref,
    strong_coupling,
    thresholds_ratios,
    xif2,
    q2_to,
    nf_ref=None,
    nf_to=None,
):
    r"""Perform the |MSbar| mass evolution up to given scale.

    It allows for different number of active flavors.

    Parameters
    ----------
    m2_ref : float
        squared initial mass reference
    q2m_ref : float
        squared initial scale
    strong_coupling : eko.strong_coupling.StrongCoupling
        Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
        any q
    xif2 : float
        :math:`\mu_F^2/\mu_R^2`
    q2_to : float
        scale at which the mass is computed
    nf_ref : int
        number of flavor active at the reference scale
    nf_to : int
        number of flavor active at the target scale

    Returns
    -------
    m2 : float
        :math:`m_{\overline{MS}}(\mu_2)^2`
    """
    matching_scales = np.array(strong_coupling.atlas.walls)[1:-1] * np.array(
        thresholds_ratios
    )
    atlas = Atlas(matching_scales.tolist(), (q2m_ref, nf_ref))
    path = atlas.path((q2_to, nf_to))
    is_downward = is_downward_path(path)
    shift = flavor_shift(is_downward)

    ev_mass = 1.0
    for k, seg in enumerate(path):
        # skip a very short segment, but keep the matching
        ker_evol = 1.0
        if not np.isclose(seg.origin, seg.target):
            ker_evol = (
                ker_dispatcher(seg.target, seg.origin, strong_coupling, xif2, seg.nf)
                ** 2
            )
        # apply matching condition
        if k < len(path) - 1:
            L = np.log(thresholds_ratios[seg.nf - shift])
            m_coeffs = (
                compute_matching_coeffs_down(seg.nf - 1)
                if is_downward
                else compute_matching_coeffs_up(seg.nf)
            )
            matching = 1.0
            for pto in range(1, strong_coupling.order[0]):
                # 0**0=1, from NNLO there is a matching also in this case
                for logpow in range(pto + 1):
                    as_thr = strong_coupling.a(seg.target * xif2, seg.nf - shift + 4)[0]
                    matching += as_thr**pto * L**logpow * m_coeffs[pto, logpow]
            ker_evol *= matching
        ev_mass *= ker_evol
    return m2_ref * ev_mass


def compute(
    masses_ref: HeavyQuarkMasses,
    couplings: CouplingsInfo,
    order: Order,
    evmeth: CouplingEvolutionMethod,
    matching: List[float],
    xif2: float = 1.0,
):
    r"""Compute the |MSbar| masses.

    Computation is performed solving the equation :math:`m_{\bar{MS}}(\mu) =
    \mu` for each heavy quark and consistent boundary contitions.

    Parameters
    ----------
    masses_ref :
        reference scale squared (a.k.a. :math:`Q_{ref}`)
    couplings :
        couplings configuration
    order :
        perturbative order
    evmeth :
        evolution method
    matching :
        threshold matching scale ratios
    xif2 :
        squared ratio of factorization to central scale

    Returns
    -------
    masses : list
        list of |MSbar| masses squared
    """
    # TODO: sketch in the docs how the MSbar computation works with a figure.
    mu2_ref = couplings.ref[0] ** 2
    nf_ref: FlavorsNumber = couplings.ref[1]
    masses = np.concatenate((np.zeros(nf_ref - 3), np.full(6 - nf_ref, np.inf)))

    def sc(thr_masses):
        return Couplings(
            couplings,
            order=order,
            method=evmeth,
            masses=thr_masses,
            thresholds_ratios=(np.array(matching) * xif2).tolist(),
            hqm_scheme=QuarkMassScheme.MSBAR,
        )

    # First you need to look for the thr around the given as_ref
    heavy_quarks = quark_names[3:]
    hq_idxs = np.arange(0, 3)
    if nf_ref > 4:
        heavy_quarks = "".join(reversed(heavy_quarks))
        hq_idxs = reversed(hq_idxs)

    # loop on heavy quarks and compute the msbar masses
    for q_idx, hq in zip(hq_idxs, heavy_quarks):
        mhq: QuarkMassRef = getattr(masses_ref, hq)
        q2m_ref = mhq.scale**2
        m2_ref = mhq.value**2

        # check if mass is already given at the pole -> done
        if q2m_ref == m2_ref:
            masses[q_idx] = m2_ref
            continue

        # update the alphas thr scales
        nf_target = q_idx + 3
        shift = -1

        # check that alphas is given with a consistent number of flavors
        if q_idx + 4 == nf_ref and q2m_ref > mu2_ref:
            raise ValueError(
                f"In MSbar scheme, Qm{hq} should be lower than Qref, "
                f"if alpha_s is given with nfref={nf_ref} at scale Qref={mu2_ref}"
            )
        if q_idx + 4 == nf_ref + 1 and q2m_ref < mu2_ref:
            raise ValueError(
                f"In MSbar scheme, Qm{hq} should be greater than Qref, "
                f"if alpha_s is given with nfref={nf_ref} at scale Qref={mu2_ref}"
            )

        # check that for higher patches you do forward running
        # with consistent conditions
        if q_idx + 3 >= nf_ref and q2m_ref >= m2_ref:
            raise ValueError(
                f"In MSbar scheme, Qm{hq} should be lower than m{hq} "
                f"if alpha_s is given with nfref={nf_ref} at scale Qref={mu2_ref}"
            )

        # check that for lower patches you do backward running
        # with consistent conditions
        if q_idx + 3 < nf_ref:
            if q2m_ref < m2_ref:
                raise ValueError(
                    f"In MSbar scheme, Qm{hq} should be greater than m{hq}"
                    f"if alpha_s is given with nfref={nf_ref} at scale Qref={mu2_ref}"
                )
            nf_target += 1
            shift = 1

        # if the initial condition is not in the target patch,
        # you need to evolve it until nf_target patch wall is reached:
        #   for backward you reach the higher, for forward the lower.
        # len(masses[q2m_ref > masses]) + 3 is the nf at the given reference scale
        nf_ref_cur = len(masses[q2m_ref > masses]) + 3
        if nf_target != nf_ref_cur:
            q2_to = masses[q_idx + shift]
            m2_ref = evolve(
                m2_ref,
                q2m_ref,
                sc(masses),
                matching,
                xif2,
                q2_to,
                nf_ref=nf_ref_cur,
                nf_to=nf_target,
            )
            q2m_ref = q2_to

        # now solve the RGE
        masses[q_idx] = solve(
            m2_ref,
            q2m_ref,
            sc(masses),
            nf_target,
            xif2,
        )

    # Check the msbar ordering
    if not np.allclose(masses, np.sort(masses)):
        raise ValueError("MSbar masses are not to be sorted")
    return np.sort(masses)
