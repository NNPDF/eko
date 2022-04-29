# -*- coding: utf-8 -*-
r"""
This file contains the QCD beta function coefficients and the handling of the running
coupling :math:`\alpha_s`.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import logging

import numba as nb
import numpy as np
import scipy

from . import constants, thresholds
from .beta import b_qcd, b_qed, beta_qcd, beta_qed

logger = logging.getLogger(__name__)


def strong_coupling_mod_ev(mod_ev):
    """Map ModEv key to the available strong coupling evolution methods"""
    if mod_ev in ["EXA", "iterate-exact", "decompose-exact", "perturbative-exact"]:
        return "exact"
    if mod_ev in [
        "TRN",
        "truncated",
        "ordered-truncated",
        "EXP",
        "iterate-expanded",
        "decompose-expanded",
        "perturbative-expanded",
    ]:
        return "expanded"
    raise ValueError(f"Unknown evolution mode {mod_ev}")


@nb.njit("f8[:](UniTuple(u1, 2),f8[:],u1,f8,f8)", cache=True)
def couplings_expanded(order, couplings_ref, nf, scale_from, scale_to):
    """
    Compute expanded expression.

    Parameters
    ----------
        order: tuple(int, int)
            perturbation order
        couplings_ref: np array
            reference alpha_s and alpha
        nf: int
            value of nf for computing the couplings
        scale_from: float
            reference scale
        scale_to : float
            target scale

    Returns
    -------
        a : np array
            couplings at target scale :math:`a(Q^2)`
    """
    # common vars
    lmu = np.log(scale_to / scale_from)

    def expanded_lo(ref, beta0, lmu):
        den = 1.0 + beta0 * ref * lmu
        return ref / den

    def expanded_nlo(ref, beta0, beta1, lmu):
        den = 1.0 + beta0 * ref * lmu
        b1 = beta1 / beta0
        as_LO = expanded_lo(ref, beta0, lmu)
        as_NLO = as_LO * (1 - b1 * as_LO * np.log(den))
        return as_NLO

    if order[1] == 0:
        beta_qcd0 = beta_qcd((0, 0), nf)
        # QCD LO
        as_LO = expanded_lo(couplings_ref[0], beta_qcd0, lmu)
        res_as = as_LO
        beta_qed0 = beta_qed((0, 0), nf)
        # QED LO
        aem_LO = expanded_lo(couplings_ref[1], beta_qed0, lmu)
        res_aem = aem_LO
        # NLO
        if order[0] >= 1:
            b1 = b_qcd((1, 0), nf)
            as_NLO = expanded_nlo(couplings_ref[0], beta_qcd0, b1 * beta_qcd0, lmu)
            res_as = as_NLO
            # NNLO
            if order[0] >= 2:
                b2 = b_qcd((2, 0), nf)
                res_as = as_LO * (
                    1.0
                    + as_LO * (as_LO - couplings_ref[0]) * (b2 - b1**2)
                    + as_NLO * b1 * np.log(as_NLO / couplings_ref[0])
                )
                # N3LO expansion is taken from Luca Rottoli
                if order[0] >= 3:
                    b3 = b_qcd((3, 0), nf)
                    log_fact = np.log(as_LO)
                    res_as += (
                        as_LO**4
                        / (2 * beta_qcd0**3)
                        * (
                            -2 * b1**3 * np.log(couplings_ref[0]) ** 3
                            + 5 * b1**3 * log_fact**2
                            + 2 * b1**3 * log_fact**3
                            + b1**3
                            * np.log(couplings_ref[0]) ** 2
                            * (5 + 6 * log_fact)
                            + 2
                            * beta_qcd0
                            * b1
                            * log_fact
                            * (
                                b2
                                + 2
                                * (b1**2 - beta_qcd0 * b2)
                                * lmu
                                * couplings_ref[0]
                            )
                            - beta_qcd0**2
                            * lmu
                            * couplings_ref[0]
                            * (
                                -2 * b1 * b2
                                + 2 * beta_qcd0 * b3
                                + (
                                    b1**3
                                    - 2 * beta_qcd0 * b1 * b2
                                    + beta_qcd0**2 * b3
                                )
                                * lmu
                                * couplings_ref[0]
                            )
                            - 2
                            * b1
                            * np.log(couplings_ref[0])
                            * (
                                5 * b1**2 * log_fact
                                + 3 * b1**2 * log_fact**2
                                + beta_qcd0
                                * (
                                    b2
                                    + 2
                                    * (b1**2 - beta_qcd0 * b2)
                                    * lmu
                                    * couplings_ref[0]
                                )
                            )
                        )
                    )
    elif order[1] == 1:
        beta_qcd0 = beta_qcd((0, 0), nf)
        # QCD LO
        as_LO = expanded_lo(couplings_ref[0], beta_qcd0, lmu)
        res_as = as_LO
        beta_qed0 = beta_qed((0, 0), nf)
        # QED NLO
        b_qed1 = b_qed((0, 1), nf)
        aem_LO = expanded_lo(couplings_ref[1], beta_qed0, lmu)
        aem_NLO = expanded_nlo(couplings_ref[1], beta_qed0, beta_qed0 * b_qed1, lmu)
        res_aem = aem_NLO
        if order[0] >= 1:
            # TODO find and implement expanded solution at order (1,1)
            raise ValueError("Order (1,1) is not implemented (yet)!")
    return np.array([res_as, res_aem])


class Couplings:
    r"""
        Computes the strong and electromagnetic coupling constants :math:`a_s,a`.

        Note that

        - all scale parameters (``scale_ref`` and ``scale_to``),
          have to be given as squared values, i.e. in units of :math:`\text{GeV}^2`
        - although, we only provide methods for
          :math:`a_i = \frac{\alpha_i(\mu^2)}{4\pi}` the reference value has to be
          given in terms of :math:`\alpha_i(\mu_0^2)` due to legacy reasons
        - the ``order`` refers to the perturbative order of the beta function, thus
          ``order=(0,0)`` means leading order beta function, means evolution with :math:`\beta_as_2`,
          means running at 1-loop - so there is a natural mismatch between ``order`` and the
          number of loops by one unit

        Normalization is given by :cite:`Herzog:2017ohr`:

        .. math::
            \frac{da_s(\mu^2)}{d\ln\mu^2} = \beta(a_s) \
            = - \sum\limits_{n=0} \beta_n a_s^{n+2}(\mu^2) \quad
            \text{with}~ a_s = \frac{\alpha_s(\mu^2)}{4\pi}

        See :doc:`pQCD ingredients </theory/pQCD>`.

        Parameters
        ----------
            couplings_ref : np array
                alpha_s and \alpha(!) at the reference scale :math:`\alpha_s(\mu_0^2),\alpha(\mu_0^2)`
            scale_ref : float
                reference scale :math:`\mu_0^2`
            masses : list(float)
                list with quark masses squared
            thresholds_ratios : list(float)
                list with ratios between the mass and the thresholds squared
            order: tuple(int,int)
                Evaluated order of the beta function: ``0`` = LO, ...
            method : ["expanded", "exact"]
                Applied method to solve the beta function
            nf_ref : int
                if given, the number of flavors at the reference scale
            max_nf : int
                if given, the maximum number of flavors
    """

    def __init__(
        self,
        couplings_ref,
        scale_ref,
        masses,
        thresholds_ratios,
        order=(0, 0),
        method="exact",
        nf_ref=None,
        max_nf=None,
        hqm_scheme="POLE",
    ):
        # Sanity checks
        if couplings_ref[0] <= 0:
            raise ValueError(f"alpha_s_ref has to be positive - got {couplings_ref[0]}")
        if couplings_ref[1] <= 0:
            raise ValueError(
                f"alpha_em_ref has to be positive - got {couplings_ref[1]}"
            )
        if scale_ref <= 0:
            raise ValueError(f"scale_ref has to be positive - got {scale_ref}")
        if order[0] not in [0, 1, 2, 3]:
            raise NotImplementedError("a_s beyond N3LO is not implemented")
        if order[1] not in [0, 1]:
            raise NotImplementedError("a_em beyond NLO is not implemented")
        self.order = order
        if method not in ["expanded", "exact"]:
            raise ValueError(f"Unknown method {method}")
        self.method = method

        # create new threshold object
        self.a_ref = couplings_ref / 4.0 / np.pi  # convert to a_s and a_em
        self.thresholds = thresholds.ThresholdsAtlas(
            masses,
            scale_ref,
            nf_ref,
            thresholds_ratios=thresholds_ratios,
            max_nf=max_nf,
        )
        self.hqm_scheme = hqm_scheme
        logger.info(
            "Strong Coupling: a_s(µ_R^2=%f)%s=%f=%f/(4π)\nElectromagnetic Coupling: a_em(µ_R^2=%f)%s=%f=%f/(4π)",
            self.q2_ref,
            f"^(nf={nf_ref})" if nf_ref else "",
            self.a_ref[0],
            self.a_ref[0] * 4 * np.pi,
            self.q2_ref,
            f"^(nf={nf_ref})" if nf_ref else "",
            self.a_ref[1],
            self.a_ref[1] * 4 * np.pi,
        )
        # cache
        self.cache = {}

    @property
    def q2_ref(self):
        """reference scale"""
        return self.thresholds.q2_ref

    @classmethod
    def from_dict(cls, theory_card, masses=None):
        r"""
        Create object from theory dictionary.

        Parameters
        ----------
            theory_card : dict
                theory dictionary
            masses: list
                list of |MSbar| masses squared or None if POLE masses are used

        Returns
        -------
            cls : Couplings
                created object
        """
        # read my values
        # TODO cast to a_s here
        alphas_ref = theory_card["alphas"]
        alphaem_ref = theory_card["alphaem"]
        couplings_ref = np.array([alphas_ref, alphaem_ref])
        nf_ref = theory_card["nfref"]
        q2_alpha = pow(theory_card["Qref"], 2)
        order = theory_card["orders"]
        method = strong_coupling_mod_ev(theory_card["ModEv"])
        hqm_scheme = theory_card["HQ"]
        if hqm_scheme not in ["MSBAR", "POLE"]:
            raise ValueError(f"{hqm_scheme} is not implemented, choose POLE or MSBAR")
        # adjust factorization scale / renormalization scale
        fact_to_ren = theory_card["fact_to_ren_scale_ratio"]
        heavy_flavors = "cbt"
        if masses is None:
            masses = np.power(
                [theory_card[f"m{q}"] / fact_to_ren for q in heavy_flavors], 2
            )
        else:
            masses = masses / fact_to_ren**2
        thresholds_ratios = np.power(
            [theory_card[f"k{q}Thr"] for q in heavy_flavors], 2
        )
        max_nf = theory_card["MaxNfAs"]
        return cls(
            couplings_ref,
            q2_alpha,
            masses,
            thresholds_ratios,
            order,
            method,
            nf_ref,
            max_nf,
            hqm_scheme,
        )

    def compute_exact(self, a_ref, nf, scale_from, scale_to):
        """
        Compute via |RGE|.

        Parameters
        ----------
            as_ref: np array
                reference alpha_s and alpha
            nf: int
                value of nf for computing alpha_i
            scale_from: float
                reference scale
            scale_to : float
                target scale

        Returns
        -------
            a : np array
                couplings at target scale :math:`a(Q^2)`
        """
        # in LO fallback to expanded, as this is the full solution
        u = np.log(scale_to / scale_from)

        def unidimensional_exact(beta0, b_vec, u, a_ref, method, rtol):
            def rge(_t, a, b_vec):
                rge = -(a**2) * (np.sum([a**k * b for k, b in enumerate(b_vec)]))
                return rge

            res = scipy.integrate.solve_ivp(
                rge,
                (0, beta0 * u),
                (a_ref,),
                args=[b_vec],
                method=method,
                rtol=rtol,
            )
            return res.y[0][-1]

        if self.order == (0, 0):
            return couplings_expanded(
                self.order, a_ref, nf, scale_from, float(scale_to)
            )
        if self.order[0] == 0:
            # return expanded solution for a_s and exact for a_em
            a_s = couplings_expanded(
                self.order, a_ref, nf, scale_from, float(scale_to)
            )[0]
            beta0_qed = beta_qed((0, 0), nf)
            b_qed_vec = [1.0]
            # NLO
            if self.order[1] >= 1:
                b_qed_vec.append(b_qed((0, 1), nf))
            a_em = unidimensional_exact(
                beta0_qed, b_qed_vec, u, a_ref[1], "Radau", 1e-6
            )
            return np.array([a_s, a_em])
        if self.order[1] == 0:
            # return expanded solution for a_em and exact for a_s
            a_em = couplings_expanded(
                self.order, a_ref, nf, scale_from, float(scale_to)
            )[1]
            beta0_qcd = beta_qcd((0, 0), nf)
            b_qcd_vec = [1.0]
            # NLO
            if self.order[0] >= 1:
                b_qcd_vec.append(b_qcd((1, 0), nf))
                # NNLO
                if self.order[0] >= 2:
                    b_qcd_vec.append(b_qcd((2, 0), nf))
                    # N3LO
                    if self.order[0] >= 3:
                        b_qcd_vec.append(b_qcd((3, 0), nf))
            a_s = unidimensional_exact(beta0_qcd, b_qcd_vec, u, a_ref[0], "Radau", 1e-6)
            return np.array([a_s, a_em])
        # otherwise rescale the RGE to run in terms of
        # u = ln(scale_to/scale_from)
        beta_qcd_vec = [beta_qcd((0, 0), nf)]
        beta_qcd_mix = 0
        # NLO
        if self.order[0] >= 1:
            beta_qcd_vec.append(beta_qcd((1, 0), nf))
            # NNLO
            if self.order[0] >= 2:
                beta_qcd_vec.append(beta_qcd((2, 0), nf))
                # N3LO
                if self.order[0] >= 3:
                    beta_qcd_vec.append(beta_qcd((3, 0), nf))
        if self.order[1] >= 1:
            beta_qcd_mix = beta_qcd((0, 1), nf)
        beta_qed_vec = [beta_qed((0, 0), nf)]
        beta_qed_mix = 0
        # NLO
        if self.order[1] >= 1:
            beta_qed_vec.append(beta_qed((0, 1), nf))
            # NNLO
        #            if self.order[0] >= 2:
        #                b_qed_vec.append(b_qed((2,0), nf))
        #                # N3LO
        #                if self.order[0] >= 3:
        #                    b_qed_vec.append(b_qed((3,0), nf))
        if self.order[0] >= 1:
            beta_qed_mix = beta_qed((1, 0), nf)
        # integration kernel
        def rge(_t, a, beta_qcd_vec, beta_qed_vec):
            rge_qcd = -(a[0] ** 2) * (
                np.sum([a[0] ** k * b for k, b in enumerate(beta_qcd_vec)])
                - a[1] * beta_qcd_mix
            )
            rge_qed = -(a[1] ** 2) * (
                np.sum([a[1] ** k * b for k, b in enumerate(beta_qed_vec)])
                - a[0] * beta_qed_mix
            )
            res = np.array([rge_qcd, rge_qed])
            return res

        # let scipy solve
        res = scipy.integrate.solve_ivp(
            rge,
            (0, u),
            a_ref,
            args=[beta_qcd_vec, beta_qed_vec],
            method="Radau",
            rtol=1e-6,
        )
        return np.array([res.y[0][-1], res.y[1][-1]])

    def compute(self, a_ref, nf, scale_from, scale_to):
        """
        Wrapper in order to pass the computation to the corresponding
        method (depending on the calculation method).

        Parameters
        ----------
            a_ref: np array
                reference a
            nf: int
                value of nf for computing alpha
            scale_from: float
                reference scale
            scale_to : float
                target scale

        Returns
        -------
            a : np array
                couplings at target scale :math:`a(Q^2)`
        """
        key = (float(a_ref[0]), float(a_ref[1]), nf, scale_from, float(scale_to))
        try:
            return self.cache[key].copy()
        except KeyError:
            # at the moment everything is expanded - and type has been checked in the constructor
            if self.method == "exact":
                a_new = self.compute_exact(
                    a_ref.astype(float), nf, scale_from, scale_to
                )
            else:
                a_new = couplings_expanded(
                    self.order, a_ref.astype(float), nf, scale_from, float(scale_to)
                )
            self.cache[key] = a_new.copy()
            return a_new

    def a(
        self,
        scale_to,
        fact_scale=None,
        nf_to=None,
    ):
        r"""
        Computes couplings :math:`a_i(\mu_R^2) = \frac{\alpha_i(\mu_R^2)}{4\pi}`.

        Parameters
        ----------
            scale_to : float
                final scale to evolve to :math:`\mu_R^2`
            fact_scale : float
                factorization scale (if different from final scale)

        Returns
        -------
            a : np array
                couplings :math:`a_i(\mu_R^2) = \frac{\alpha_i(\mu_R^2)}{4\pi}`
        """
        # Set up the path to follow in order to go from q2_0 to q2_ref
        final_a = self.a_ref
        path = self.thresholds.path(scale_to, nf_to)
        is_downward = thresholds.is_downward_path(path)
        shift = thresholds.flavor_shift(is_downward)

        # as a default assume mu_F^2 = mu_R^2
        if fact_scale is None:
            fact_scale = scale_to
        for k, seg in enumerate(path):
            # skip a very short segment, but keep the matching
            if not np.isclose(seg.q2_from, seg.q2_to):
                new_a = self.compute(final_a, seg.nf, seg.q2_from, seg.q2_to)
            else:
                new_a = final_a
            # apply matching conditions: see hep-ph/9706430
            # - if there is yet a step to go
            if k < len(path) - 1:
                # q2_to is the threshold value
                L = np.log(scale_to / fact_scale) + np.log(
                    self.thresholds.thresholds_ratios[seg.nf - shift]
                )
                m_coeffs = (
                    compute_matching_coeffs_down(self.hqm_scheme, seg.nf - 1)
                    if is_downward
                    else compute_matching_coeffs_up(self.hqm_scheme, seg.nf)
                )
                fact = 1.0
                # shift
                for n in range(1, self.order[0] + 1):
                    for l in range(n + 1):
                        fact += new_a[0] ** n * L**l * m_coeffs[n, l]
                new_a[0] *= fact
            final_a = new_a
        return final_a


def compute_matching_coeffs_up(mass_scheme, nf):
    r"""
    Matching coefficients :cite:`Schroder:2005hy,Chetyrkin:2005ia,Vogt:2004ns`
    at threshold when moving to a regime with *more* flavors.

    We follow notation of :cite:`Vogt:2004ns` (eq 2.43) for POLE masses

    The *inverse* |MSbar| values instead are given in :cite:`Schroder:2005hy` (eq 3.1)
    multiplied by a factor of 4 (and 4^2 ...)

    .. math::
        a_s^{(n_l+1)} = a_s^{(n_l)} + \sum\limits_{n=1} (a_s^{(n_l)})^n
                                \sum\limits_{k=0}^n c_{nl} \log(\mu_R^2/\mu_F^2)

    Parameters
    ----------
        mass_scheme:
            Heavy quark mass scheme: "POLE" or "MSBAR"
        nf:
            number of active flavors in the lower patch

    Returns
    -------
        matching_coeffs_down:
            forward matching coefficient matrix
    """
    matching_coeffs_up = np.zeros((4, 4))
    if mass_scheme == "MSBAR":
        matching_coeffs_up[2, 0] = -22.0 / 9.0
        matching_coeffs_up[2, 1] = 22.0 / 3.0

        # c30 = -d30
        matching_coeffs_up[3, 0] = -62.2116 + 5.4177 * nf
        # c31 = -d31 + 5 c11 * c20
        matching_coeffs_up[3, 1] = 365.0 / 3.0 - 67.0 / 9.0 * nf
        matching_coeffs_up[3, 2] = 109.0 / 3.0 + 16.0 / 9.0 * nf

    elif mass_scheme == "POLE":
        matching_coeffs_up[2, 0] = 14.0 / 3.0
        matching_coeffs_up[2, 1] = 38.0 / 3.0
        matching_coeffs_up[3, 0] = 340.729 - 16.7981 * nf
        matching_coeffs_up[3, 1] = 8941.0 / 27.0 - 409.0 / 27.0 * nf
        matching_coeffs_up[3, 2] = 511.0 / 9.0

    matching_coeffs_up[1, 1] = 4.0 / 3.0 * constants.TR
    matching_coeffs_up[2, 2] = 4.0 / 9.0
    matching_coeffs_up[3, 3] = 8.0 / 27.0

    return matching_coeffs_up


# inversion of the matching coefficients
def compute_matching_coeffs_down(mass_scheme, nf):
    """
    Matching coefficients :cite:`Schroder:2005hy,Chetyrkin:2005ia` at threshold
    when moving to a regime with *less* flavors.

    Parameters
    ----------
        mass_scheme:
            Heavy quark mass scheme: "POLE" or "MSBAR"
        nf:
            number of active flavors in the lower patch

    Returns
    -------
        matching_coeffs_down:
            downward matching coefficient matrix
    """
    c_up = compute_matching_coeffs_up(mass_scheme, nf)
    return invert_matching_coeffs(c_up)


def invert_matching_coeffs(c_up):
    """
    This is the perturbative inverse of :data:`matching_coeffs_up` and has been obtained via

    .. code-block:: Mathematica

        Module[{f, g, l, sol},
            f[a_] := a + Sum[d[n, k]*L^k*a^(1 + n), {n, 3}, {k, 0, n}];
            g[a_] := a + Sum[c[n, k]*L^k*a^(1 + n), {n, 3}, {k, 0, n}] /. {c[1, 0] -> 0};
            l = CoefficientList[Normal@Series[f[g[a]], {a, 0, 5}], {a, L}];
            sol = First@
                Solve[{l[[3]] == 0, l[[4]] == 0, l[[5]] == 0},
                Flatten@Table[d[n, k], {n, 3}, {k, 0, n}]];
            Do[Print@r, {r, sol}];
            Print@Series[f[g[a]] /. sol, {a, 0, 5}];
            Print@Series[g[f[a]] /. sol, {a, 0, 5}];
        ]

    Parameters
    ----------
        c_up : numpy.ndarray
            forward matching coefficient matrix

    Returns
    -------
        matching_coeffs_down:
            downward matching coefficient matrix
    """
    matching_coeffs_down = np.zeros_like(c_up)
    matching_coeffs_down[1, 1] = -c_up[1, 1]
    matching_coeffs_down[2, 0] = -c_up[2, 0]
    matching_coeffs_down[2, 1] = -c_up[2, 1]
    matching_coeffs_down[2, 2] = 2.0 * c_up[1, 1] ** 2 - c_up[2, 2]
    matching_coeffs_down[3, 0] = -c_up[3, 0]
    matching_coeffs_down[3, 1] = 5 * c_up[1, 1] * c_up[2, 0] - c_up[3, 1]
    matching_coeffs_down[3, 2] = 5 * c_up[1, 1] * c_up[2, 1] - c_up[3, 2]
    matching_coeffs_down[3, 3] = (
        -5 * c_up[1, 1] ** 3 + 5 * c_up[1, 1] * c_up[2, 2] - c_up[3, 3]
    )
    return matching_coeffs_down
