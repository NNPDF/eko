# -*- coding: utf-8 -*-
r"""
This module contains the |RGE| for the |MSbar| masses
"""
import numpy as np
from scipy import integrate, optimize

from .beta import b, beta
from .gamma import gamma
from .strong_coupling import StrongCoupling


def msbar_ker_exact(a0, a1, order, nf):
    r"""
    Exact |MSbar| |RGE| kernel

    Parameters
    ----------
        a0: float
            strong coupling at the initial scale
        a1: float
            strong coupling at the final scale
        oreder: int
            perturbative order
        nf: int
            number of active flavours

    Returns
    -------
        ker: float
            Exact |MSbar| kernel:

            .. math::
                k_{exact} = e^{- \int_{a_s(\mu_{h,0}^2)}^{a_s(\mu^2)} \gamma_m(a_s) / \beta(a_s) da_s}
    """ # pylint:disable=line-too-long
    b_vec = [beta(0, nf)]
    g_vec = [gamma(0, nf)]
    if order >= 1:
        b_vec.append(beta(1, nf))
        g_vec.append(gamma(1, nf))
    if order >= 2:
        b_vec.append(beta(2, nf))
        g_vec.append(gamma(2, nf))
    if order >= 3:
        b_vec.append(beta(3, nf))
        g_vec.append(gamma(3, nf))

    # quad ker
    def integrand(a, b_vec, g_vec):
        # minus sign goes away
        fgamma = np.sum([a ** k * b for k, b in enumerate(g_vec)])
        fbeta = a * np.sum([a ** k * b for k, b in enumerate(b_vec)])
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


def msbar_ker_expanded(a0, a1, order, nf):
    r"""
    Expanded |MSbar| |RGE| kernel

    Parameters
    ----------
        a0: float
            strong coupling at the initial scale
        a1: float
            strong coupling at the final scale
        oreder: int
            perturbative order
        nf: int
            number of active flavours

    Returns
    -------
        ker: float
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
    b0 = beta(0, nf)
    c0 = gamma(0, nf) / b0
    ev_mass = np.power(a1 / a0, c0)
    num = 1.0
    den = 1.0
    if order >= 1:
        b1 = b(1, nf)
        c1 = gamma(1, nf) / b0
        u = c1 - b1 * c0
        num += a1 * u
        den += a0 * u
    if order >= 2:
        b2 = b(2, nf)
        c2 = gamma(2, nf) / b0
        u = (c2 - c1 * b1 - b2 * c0 + b1 ** 2 * c0 + (c1 - b1 * c0) ** 2) / 2.0
        num += a1 ** 2 * u
        den += a0 ** 2 * u
    if order >= 3:
        b3 = b(3, nf)
        c3 = gamma(3, nf) / b0
        u = (
            1
            / 6
            * (
                -2 * b3 * c0
                - b1 ** 3 * c0 * (1 + c0) * (2 + c0)
                - 2 * b2 * c1
                - 3 * b2 * c0 * c1
                + b1 ** 2 * (2 + 3 * c0 * (2 + c0)) * c1
                + c1 ** 3
                + 3 * c1 * c2
                + b1
                * (b2 * c0 * (4 + 3 * c0) - 3 * (1 + c0) * c1 ** 2 - (2 + 3 * c0) * c2)
                + 2 * c3
            )
        )
        num += a1 ** 3 * u
        den += a0 ** 3 * u
    return ev_mass * num / den


def msbar_ker_dispatcher(q2_to, q2m_ref, strong_coupling, fact_to_ren, nf):
    r"""
    Select the |MSbar| kernel and compute the strong coupling values

    Parameters
    ----------
        q2_to: float
            final scale
        q2m_ref: float
            initial scale
        strong_coupling: eko.strong_coupling.StrongCoupling
            Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
            any q
        fact_to_ren: float
            factorization to renormalization scale ratio
        nf: int
            number of active flavours

    Returns
    -------
        ker:
            Expanded or exact |MSbar| kernel
    """
    a0 = strong_coupling.a_s(q2m_ref / fact_to_ren, q2m_ref)
    a1 = strong_coupling.a_s(q2_to / fact_to_ren, q2_to)
    method = strong_coupling.method
    order = strong_coupling.order
    if method == "expanded":
        return msbar_ker_expanded(a0, a1, order, nf)
    return msbar_ker_exact(a0, a1, order, nf)


def evolve_msbar_mass(
    m2_ref,
    q2m_ref,
    nf_ref=None,
    config=None,
    strong_coupling=None,
    q2_to=None,
):
    r"""
    Compute the |MSbar| mass.
    If the final scale is not given it solves the equation :math:`m_{\overline{MS}}(m) = m`
    for a fixed number of nf
    Else perform the mass evolution.

    Parameters
    ----------
        m2_ref: float
            squared initial mass reference
        q2m_ref: float
            squared initial scale
        nf_ref: int, optional (not used when q2_to is given)
            number of active flavours at the scale q2m_ref, where the solution is searched
        config: dict
            |MSbar| configuration dictionary
        strong_coupling: eko.strong_coupling.StrongCoupling
            Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
            any q
        q2_to: float, optional
            scale at which the mass is computed

    Returns
    -------
        m2 : float
            :math:`m_{\overline{MS}}(\mu_2)^2`
    """
    # set the missing information if needed
    fact_to_ren = config["fact_to_ren"]
    if strong_coupling is None:
        strong_coupling = StrongCoupling(
            config["as_ref"],
            config["q2a_ref"],
            config["thr_masses"],
            thresholds_ratios=[1, 1, 1],
            order=config["order"],
            method=config["method"],
            nf_ref=config["nfref"],
        )

    if q2_to is None:

        def rge(m2, q2m_ref, strong_coupling, fact_to_ren, nf_ref):
            return (
                m2_ref
                * msbar_ker_dispatcher(
                    m2, q2m_ref, strong_coupling, fact_to_ren, nf_ref
                )
                ** 2
                - m2
            )

        msbar_mass = optimize.fsolve(
            rge, q2m_ref, args=(q2m_ref, strong_coupling, fact_to_ren, nf_ref)
        )
        return float(msbar_mass)
    else:

        # evolution might involve different number of active flavors.
        # Find out the evolution path (always sorted)
        q2_low, q2_high = sorted([q2m_ref, q2_to])
        area_walls = np.array(strong_coupling.thresholds.area_walls)
        path = np.concatenate(
            (
                [q2_low],
                area_walls[(q2_low < area_walls) & (area_walls < q2_high)],
                [q2_high],
            )
        )
        nf_init = len(area_walls[q2_low >= area_walls]) + 2
        nf_final = len(area_walls[q2_high > area_walls]) + 2

        ev_mass = 1.0
        for i, n in enumerate(np.arange(nf_init, nf_final + 1)):
            q2_init = path[i]
            q2_final = path[i + 1]
            # if you are going backward
            # need to reverse the evolution in each path segment
            if q2m_ref > q2_to:
                q2_init, q2_final = q2_final, q2_init
            ev_mass *= msbar_ker_dispatcher(
                q2_final, q2_init, strong_coupling, fact_to_ren, n
            )
        return m2_ref * ev_mass ** 2
