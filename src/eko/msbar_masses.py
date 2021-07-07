# -*- coding: utf-8 -*-
r"""
This module contains the RGE for the ms bar masses
"""
import numpy as np
from scipy import optimize, integrate

from .beta import beta, b
from .gamma import gamma
from .strong_coupling import StrongCoupling


def msbar_ker_exact(a0, a1, order, nf):
    r"""
    Exact :math:`\overline{MS}` RGE kernel

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
            Exact :math:`\overline{MS}` kernel:

            ..math:
                k_{exact} = e^{\int_{a_s(\mu_{h,0}^2)}^{a_s(\mu^2)} \gamma(a_s) / \beta(a_s) da_s}
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
    Expanded :math:`\overline{MS}` RGE kernel

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
            Expanded :math:`\overline{MS}` kernel:

            ..math:
                k_{expanded} &= \left (\frac{a_s(\mu^2)}{a_s(\mu_{h,0}^2)} \right )^{c_0}
                \frac{j_{exp}(a_s(\mu^2))}{j_{exp}(a_s(\mu_{h,0}^2))} \\
                j_{exp}(a_s) &= 1 + a_s \left [ c_1 - b_1 c_0 \right ]
                + \frac{a_s^2}{2}
                \left [c_2 - c_1 b_1 - b_2 c_0 + b_1^2 c_0 + (c_1 - b_1 c_0)^2 right]
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
    r"""
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
            Expanded or exact :math:`\overline{MS}` kernel
    """
    a0 = strong_coupling.a_s(q2m_ref / fact_to_ren, q2m_ref)
    a1 = strong_coupling.a_s(q2_to / fact_to_ren, q2_to)
    method = strong_coupling.method
    order = strong_coupling.order
    if method == "expanded":
        return msbar_ker_expanded(a0, a1, order, nf)
    return msbar_ker_exact(a0, a1, order, nf)


def evolve_msbar_mass(
    m2_ref, q2m_ref, nf, config=None, strong_coupling=None, q2_to=None
):
    r"""
    Compute the MSbar mass.
    If the final scale is not gven it solves the equation :math:`m_{\overline{MS}}(m) = m`

    Parameters
    ----------
        m2_ref: float
            squared intial mass reference
        q2m_ref: float
            squared intial scale
        nf: int
            number of active flavours
        config: dict
            msbar configuration dictionary
        strong_coupling: eko.strong_coupling.StrongCoupling
            Instance of :class:`~eko.strong_coupling.StrongCoupling` able to generate a_s for
            any q
        q2_to: float
            scale at which the mass is computed

    Returns
    -------
        m2 : float
            :math:`m_{\overline{MS}}(\mu_2)`
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
        )

    if q2_to is None:

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


def compute_msbar_mass(theory_card):
    r"""
    Compute the :math:`\overline{MS}` masses solving the equation :math:`m_{\bar{MS}}(m) = m`

    Parameters
    ----------
        theory_card: dict
            theory run card

    Returns
    -------
        msbar_masses: list
            list of msbar masses
    """
    heavy_flavors = "cbt"
    msbar_masses = np.full(3, np.inf)
    shift = 3
    q2m = np.power([theory_card[f"Qm{q}"] for q in heavy_flavors], 2)
    m2 = np.power([theory_card[f"m{q}"] for q in heavy_flavors], 2)
    config = {
        "q2m_ref": np.power([theory_card[f"Qm{q}"] for q in heavy_flavors], 2),
        "as_ref": theory_card["alphas"],
        "q2a_ref": np.power(theory_card["Qref"], 2),
        "order": theory_card["PTO"],
        "fact_to_ren": theory_card["fact_to_ren_scale_ratio"] ** 2,
    }
    for nf in [3, 4, 5]:
        q2m_ref = q2m[nf - shift]
        m2_ref = m2[nf - shift]
        # check if mass is already given at the pole
        if q2m_ref == m2_ref:
            msbar_masses[nf - shift] = m2_ref
            continue
        # if self.q2_ref > q2m_ref:
        #     raise ValueError("In MSBAR scheme Q0 must be lower than any Qm")
        # if q2m_ref > m2_ref:
        #     raise ValueError("In MSBAR scheme each heavy quark \
        #         mass reference scale must be smaller or equal than \
        #             the value of the mass itself"
        #     )
        config["thr_masses"] = msbar_masses
        msbar_masses[nf - shift] = evolve_msbar_mass(m2_ref, q2m_ref, nf, config=config)

    # Check the msbar ordering
    for nf in [4, 5]:
        q2m_ref = q2m[nf - shift]
        m2_ref = m2[nf - shift]
        m2_msbar = msbar_masses[nf - shift - 1]
        config["thr_masses"] = msbar_masses
        # check that m_msbar_hq < msbar_hq+1 (m_msbar_hq)
        m2_test = evolve_msbar_mass(m2_ref, q2m_ref, nf, config=config, q2_to=m2_msbar)
        if m2_msbar > m2_test:
            raise ValueError(
                "The MSBAR masses do not preserve the correct ordering,\
                     check the inital reference values"
            )
    return msbar_masses
