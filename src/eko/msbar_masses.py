# -*- coding: utf-8 -*-
r"""
This module contains the RGE for the ms bar masses
"""
import numpy as np
import scipy.integrate as integrate
from scipy import optimize

from .beta import beta, b
from .gamma import gamma


def msbar_ker_exact(a0, a1, order, nf):
    r"""
    Exact MSBar RGE kernel

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
            Exact MSBar kernel:

            ..math:
                k_{exact} = e^{\int_{a_0}^{a} \gamma(a) / \beta(a) da}
    """
    b_vec = [beta(0, nf)]
    g_vec = [gamma(0, nf)]
    if order >= 1:
        b_vec.append(beta(1, nf))
        g_vec.append(gamma(1, nf))
    if order >= 2:
        b_vec.append(beta(2, nf))
        g_vec.append(gamma(2, nf))

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
    Expanded MSBar RGE kernel

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
            Expaned MSBar kernel:

            ..math:
                k_{expanded} &= (\frac{a}{a_0})^{c_0} \frac{j_{exp}(a)}{j_{exp}(a_0)} \\
                j_{exp}(a) &= 1 + a \left [ c_1 - b_1 c_0 \right ]
                + a^2 \left [c_2 - c_1 b_1 - b_2 c_0 + b_1^2 c_0 + (c_1 - b_1 c_0)^2) / 2.0 \right]
    """
    b0 = beta(0, nf)
    c0 = gamma(0, nf) / b0
    ev_mass = np.power(a1 / a0, c0)
    num = 1.0
    den = 1.0
    if order >= 1:
        b1 = b(1, nf)
        c1 = gamma(1, nf) / b0
        r = c1 - b1 * c0
        num += a1 * r
        den += a0 * r
    if order >= 2:
        b2 = b(2, nf)
        c2 = gamma(2, nf) / b0
        r = (c2 - c1 * b1 - b2 * c0 + b1 ** 2 * c0 + (c1 - b1 * c0) ** 2) / 2.0
        num += a1 ** 2 * r
        den += a0 ** 2 * r
    return ev_mass * num / den


def msbar_ker_dispatcher(q2_to, q2m_ref, strong_coupling, fact_to_ren, nf):
    """
    Select the MSbar kernel and compute the strong coupling values

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
            Expaned or exact MSBar kernel
    """
    a0 = strong_coupling.a_s(q2m_ref / fact_to_ren, q2m_ref)
    a1 = strong_coupling.a_s(q2_to / fact_to_ren, q2_to)
    method = strong_coupling.method
    order = strong_coupling.order
    if method == "expanded":
        return msbar_ker_expanded(a0, a1, order, nf)
    return msbar_ker_exact(a0, a1, order, nf)


def evolve_msbar_mass(m2_ref, q2m_ref, strong_coupling, fact_to_ren, nf, q2_to=None):
    """
    Compute the MSbar mass.
    If the final scale is not gven it solves the equation :math:`m_{\bar{MS}}(m) = m`

    Parameters
    ----------
        m2_ref: float
            squared intial mass reference
        q2m_ref: float
            squared intial scale
        strong_coupling: strong_coupling: eko.strong_coupling.StrongCoupling
            Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
            any q
        fact_to_ren: float
            factorization to renormalization scale ratio
        nf: int
            number of active flavours
        q2_to: float
            scale at which the mass is computed.
            If not given it solves the equation :math:`m_{\bar{MS}}(m) = m`
    Returns
    -------
        m2 : float
            :math:`m_{\bar{MS}}(q2)`
    """

    if q2_to is None:
        # check if mass is already given at the pole
        if q2m_ref == m2_ref:
            return m2_ref

        def rge(m2, q2m_ref, strong_coupling, fact_to_ren, nf):
            return (
                m2_ref
                * msbar_ker_dispatcher(m2, q2m_ref, strong_coupling, fact_to_ren, nf)
                ** 2
                - m2
            )

        msbar_mass = optimize.fsolve(
            rge, q2m_ref, args=(q2m_ref, strong_coupling, fact_to_ren, nf)
        )
        return float(msbar_mass)
    else:
        ev_mass = msbar_ker_dispatcher(q2_to, q2m_ref, strong_coupling, fact_to_ren, nf)
        return m2_ref * ev_mass ** 2
