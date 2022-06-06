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

from . import constants
from . import scale_variations as sv
from . import thresholds
from .beta import b as beta_b
from .beta import beta

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


@nb.njit(cache=True)
def as_expanded(order, as_ref, nf, scale_from, scale_to):
    """
    Compute expanded expression.

    Parameters
    ----------
        order: int
            perturbation order
        as_ref: float
            reference alpha_s
        nf: int
            value of nf for computing alpha_s
        scale_from: float
            reference scale
        scale_to : float
            target scale

    Returns
    -------
        a_s : float
            coupling at target scale :math:`a_s(Q^2)`
    """
    # common vars
    beta0 = beta(0, nf)
    lmu = np.log(scale_to / scale_from)
    den = 1.0 + beta0 * as_ref * lmu
    # LO
    as_LO = as_ref / den
    res = as_LO
    # NLO
    if order >= 1:
        b1 = beta_b(1, nf)
        as_NLO = as_LO * (1 - b1 * as_LO * np.log(den))
        res = as_NLO
        # NNLO
        if order >= 2:
            b2 = beta_b(2, nf)
            res = as_LO * (
                1.0
                + as_LO * (as_LO - as_ref) * (b2 - b1**2)
                + as_NLO * b1 * np.log(as_NLO / as_ref)
            )
            # N3LO expansion is taken from Luca Rottoli
            if order >= 3:
                b3 = beta_b(3, nf)
                log_fact = np.log(as_LO)
                res += (
                    as_LO**4
                    / (2 * beta0**3)
                    * (
                        -2 * b1**3 * np.log(as_ref) ** 3
                        + 5 * b1**3 * log_fact**2
                        + 2 * b1**3 * log_fact**3
                        + b1**3 * np.log(as_ref) ** 2 * (5 + 6 * log_fact)
                        + 2
                        * beta0
                        * b1
                        * log_fact
                        * (b2 + 2 * (b1**2 - beta0 * b2) * lmu * as_ref)
                        - beta0**2
                        * lmu
                        * as_ref
                        * (
                            -2 * b1 * b2
                            + 2 * beta0 * b3
                            + (b1**3 - 2 * beta0 * b1 * b2 + beta0**2 * b3)
                            * lmu
                            * as_ref
                        )
                        - 2
                        * b1
                        * np.log(as_ref)
                        * (
                            5 * b1**2 * log_fact
                            + 3 * b1**2 * log_fact**2
                            + beta0 * (b2 + 2 * (b1**2 - beta0 * b2) * lmu * as_ref)
                        )
                    )
                )
    return res


class StrongCoupling:
    r"""
        Computes the strong coupling constant :math:`a_s`.

        Note that

        - all scale parameters (``scale_ref`` and ``scale_to``),
          have to be given as squared values, i.e. in units of :math:`\text{GeV}^2`
        - although, we only provide methods for
          :math:`a_s = \frac{\alpha_s(\mu^2)}{4\pi}` the reference value has to be
          given in terms of :math:`\alpha_s(\mu_0^2)` due to legacy reasons
        - the ``order`` refers to the perturbative order of the beta function, thus
          ``order=0`` means leading order beta function, means evolution with :math:`\beta_0`,
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
            alpha_s_ref : float
                alpha_s(!) at the reference scale :math:`\alpha_s(\mu_0^2)`
            scale_ref : float
                reference scale :math:`\mu_0^2`
            masses : list(float)
                list with quark masses squared
            thresholds_ratios : list(float)
                list with ratios between the mass and the matching scales squared
            order: int
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
        alpha_s_ref,
        scale_ref,
        masses,
        thresholds_ratios,
        order=0,
        method="exact",
        nf_ref=None,
        max_nf=None,
        hqm_scheme="POLE",
    ):
        # Sanity checks
        if alpha_s_ref <= 0:
            raise ValueError(f"alpha_s_ref has to be positive - got {alpha_s_ref}")
        if scale_ref <= 0:
            raise ValueError(f"scale_ref has to be positive - got {scale_ref}")
        if order not in [0, 1, 2, 3]:
            raise NotImplementedError("a_s beyond N3LO is not implemented")
        self.order = order
        if method not in ["expanded", "exact"]:
            raise ValueError(f"Unknown method {method}")
        self.method = method

        # create new threshold object
        self.as_ref = alpha_s_ref / 4.0 / np.pi  # convert to a_s
        self.thresholds = thresholds.ThresholdsAtlas(
            masses,
            scale_ref,
            nf_ref,
            thresholds_ratios=thresholds_ratios,
            max_nf=max_nf,
        )
        self.hqm_scheme = hqm_scheme
        logger.info(
            "Strong Coupling: a_s(µ_R^2=%f)%s=%f=%f/(4π)",
            self.q2_ref,
            f"^(nf={nf_ref})" if nf_ref else "",
            self.as_ref,
            self.as_ref * 4 * np.pi,
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
            cls : StrongCoupling
                created object
        """
        # read my values
        # TODO cast to a_s here
        alpha_ref = theory_card["alphas"]
        nf_ref = theory_card["nfref"]
        q2_alpha = pow(theory_card["Qref"], 2)
        order = theory_card["PTO"]
        method = strong_coupling_mod_ev(theory_card["ModEv"])
        hqm_scheme = theory_card["HQ"]
        if hqm_scheme not in ["MSBAR", "POLE"]:
            raise ValueError(f"{hqm_scheme} is not implemented, choose POLE or MSBAR")
        # adjust factorization scale / renormalization scale
        fact_to_ren = theory_card["fact_to_ren_scale_ratio"]
        if sv.sv_mode(theory_card["ModSV"]) is not sv.Modes.exponentiated:
            fact_to_ren = 1.0
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
            alpha_ref,
            q2_alpha,
            masses,
            thresholds_ratios,
            order,
            method,
            nf_ref,
            max_nf,
            hqm_scheme,
        )

    def compute_exact(self, as_ref, nf, scale_from, scale_to):
        """
        Compute via |RGE|.

        Parameters
        ----------
            as_ref: float
                reference alpha_s
            nf: int
                value of nf for computing alpha_s
            scale_from: float
                reference scale
            scale_to : float
                target scale

        Returns
        -------
            a_s : float
                strong coupling at target scale :math:`a_s(Q^2)`
        """
        # in LO fallback to expanded, as this is the full solution
        if self.order == 0:
            return as_expanded(self.order, as_ref, nf, scale_from, float(scale_to))
        # otherwise rescale the RGE to run in terms of
        # u = beta0 * ln(scale_to/scale_from)
        beta0 = beta(0, nf)
        u = beta0 * np.log(scale_to / scale_from)
        b_vec = [1]
        # NLO
        if self.order >= 1:
            b_vec.append(beta_b(1, nf))
            # NNLO
            if self.order >= 2:
                b_vec.append(beta_b(2, nf))
                # N3LO
                if self.order >= 3:
                    b_vec.append(beta_b(3, nf))
        # integration kernel
        def rge(_t, a, b_vec):
            return -(a**2) * np.sum([a**k * b for k, b in enumerate(b_vec)])

        # let scipy solve
        res = scipy.integrate.solve_ivp(
            rge, (0, u), (as_ref,), args=[b_vec], method="Radau", rtol=1e-6
        )
        return res.y[0][-1]

    def compute(self, as_ref, nf, scale_from, scale_to):
        """
        Wrapper in order to pass the computation to the corresponding
        method (depending on the calculation method).

        Parameters
        ----------
            as_ref: float
                reference alpha_s
            nf: int
                value of nf for computing alpha_s
            scale_from: float
                reference scale
            scale_to : float
                target scale

        Returns
        -------
            a_s : float
                strong coupling at target scale :math:`a_s(Q^2)`
        """
        key = (float(as_ref), nf, scale_from, float(scale_to))
        try:
            return self.cache[key]
        except KeyError:
            # at the moment everything is expanded - and type has been checked in the constructor
            if self.method == "exact":
                as_new = self.compute_exact(float(as_ref), nf, scale_from, scale_to)
            else:
                as_new = as_expanded(
                    self.order, float(as_ref), nf, scale_from, float(scale_to)
                )
            self.cache[key] = as_new
            return as_new

    def a_s(
        self,
        scale_to,
        fact_scale=None,
        nf_to=None,
    ):
        r"""
        Computes strong coupling :math:`a_s(\mu_R^2) = \frac{\alpha_s(\mu_R^2)}{4\pi}`.

        Parameters
        ----------
            scale_to : float
                final scale to evolve to :math:`\mu_R^2`
            fact_scale : float
                factorization scale (if different from final scale)

        Returns
        -------
            a_s : float
                strong coupling :math:`a_s(\mu_R^2) = \frac{\alpha_s(\mu_R^2)}{4\pi}`
        """
        # Set up the path to follow in order to go from q2_0 to q2_ref
        final_as = self.as_ref
        path = self.thresholds.path(scale_to, nf_to)
        is_downward = thresholds.is_downward_path(path)
        shift = thresholds.flavor_shift(is_downward)

        # as a default assume mu_F^2 = mu_R^2
        if fact_scale is None:
            fact_scale = scale_to
        for k, seg in enumerate(path):
            # skip a very short segment, but keep the matching
            if not np.isclose(seg.q2_from, seg.q2_to):
                new_as = self.compute(final_as, seg.nf, seg.q2_from, seg.q2_to)
            else:
                new_as = final_as
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
                for n in range(1, self.order + 1):
                    for l in range(n + 1):
                        fact += new_as**n * L**l * m_coeffs[n, l]
                new_as *= fact
            final_as = new_as
        return final_as


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
