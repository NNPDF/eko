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
from .beta import beta

logger = logging.getLogger(__name__)


@nb.njit("f8(u1,f8,u1,f8,f8)", cache=True)
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
        beta1 = beta(1, nf)
        b1 = beta1 / beta0
        as_NLO = as_LO * (1 - b1 * as_LO * np.log(den))
        res = as_NLO
        # NNLO
        if order == 2:
            beta2 = beta(2, nf)
            b2 = beta2 / beta0
            res = as_LO * (
                1.0
                + as_LO * (as_LO - as_ref) * (b2 - b1 ** 2)
                + as_NLO * b1 * np.log(as_NLO / as_ref)
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
            threshold_holder : eko.thresholds.ThresholdsAtlas
                An instance of :class:`~eko.thresholds.ThresholdsAtlas`
            order: int
                Evaluated order of the beta function: ``0`` = LO, ...
            method : ["expanded", "exact"]
                Applied method to solve the beta function

        Examples
        --------
            >>> alpha_ref = 0.35
            >>> scale_ref = 2
            >>> threshold_holder = ThresholdsAtlas( ... )
            >>> sc = StrongCoupling(alpha_ref, scale_ref, threshold_holder)
            >>> q2 = 91.1**2
            >>> sc.a_s(q2)
            0.118
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
    ):
        # Sanity checks
        if alpha_s_ref <= 0:
            raise ValueError(f"alpha_s_ref has to be positive - got {alpha_s_ref}")
        if scale_ref <= 0:
            raise ValueError(f"scale_ref has to be positive - got {scale_ref}")
        if order not in [0, 1, 2]:
            raise NotImplementedError("a_s beyond NNLO is not implemented")
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
        logger.info(
            "Strong Coupling: a_s(µ_R^2=%f)%s=%f=%f/(4π)",
            self.q2_ref,
            "^(nf=%d)" % nf_ref if nf_ref else "",
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
    def from_dict(cls, theory_card):
        """
        Create object from theory dictionary.

        Read keys:

            - alphas : required, reference value for  alpha_s (!)
            - Qref : required, reference value in GeV (!)
            - PTO : required, perturbative order
            - ModEv : optional, method to solve RGE, default=EXA

        Parameters
        ----------
            theory_card : dict
                theory dictionary

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
        mod_ev = theory_card["ModEv"]
        if mod_ev in ["EXA", "iterate-exact", "decompose-exact", "perturbative-exact"]:
            method = "exact"
        elif mod_ev in [
            "TRN",
            "truncated",
            "ordered-truncated",
            "EXP",
            "iterate-expanded",
            "decompose-expanded",
            "perturbative-expanded",
        ]:
            method = "expanded"
        else:
            raise ValueError(f"Unknown evolution mode {mod_ev}")
        # adjust factorization scale / renormalization scale
        fact_to_ren = theory_card["fact_to_ren_scale_ratio"]
        heavy_flavors = "cbt"
        masses = np.power(
            [theory_card[f"m{q}"] / fact_to_ren for q in heavy_flavors], 2
        )
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
        )

    def compute_exact(self, as_ref, nf, scale_from, scale_to):
        """
        Compute via RGE.

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
            return as_expanded(self.order, as_ref, nf, scale_from, scale_to)
        # otherwise rescale the RGE to run in terms of
        # u = beta0 * ln(scale_to/scale_from)
        beta0 = beta(0, nf)
        u = beta0 * np.log(scale_to / scale_from)
        b_vec = [1]
        # NLO
        if self.order >= 1:
            beta1 = beta(1, nf)
            b1 = beta1 / beta0
            b_vec.append(b1)
            # NNLO
            if self.order >= 2:
                beta2 = beta(2, nf)
                b2 = beta2 / beta0
                b_vec.append(b2)
        # integration kernel
        def rge(_t, a, b_vec):
            return -(a ** 2) * np.sum([a ** k * b for k, b in enumerate(b_vec)])

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
        key = (as_ref, nf, scale_from, scale_to)
        try:
            return self.cache[key]
        except KeyError:
            # at the moment everything is expanded - and type has been checked in the constructor
            if self.method == "exact":
                as_new = self.compute_exact(as_ref, nf, scale_from, scale_to)
            else:
                as_new = as_expanded(self.order, as_ref, nf, scale_from, scale_to)
            self.cache[key] = as_new
            return as_new

    def a_s(self, scale_to, fact_scale=None, nf_to=None):
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
        is_downward_path = False
        if len(path) > 1:
            is_downward_path = path[1].nf < path[0].nf
        shift = 3 if not is_downward_path else 4

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
                    matching_coeffs_down if is_downward_path else matching_coeffs_up
                )
                fact = 1.0
                # shift
                for n in range(1, self.order + 1):
                    for l in range(n + 1):
                        fact += new_as ** n * L ** l * m_coeffs[n, l]
                # shift
                new_as *= fact
            final_as = new_as
        return final_as


matching_coeffs_up = np.zeros((3, 3))
r"""
Matching coefficients :cite:`Schroder:2005hy,Chetyrkin:2005ia,Vogt:2004ns` at threshold
when moving to a regime with *more* flavors.

.. math::
    a_s^{(n_l+1)} = a_s^{(n_l)} + \sum\limits_{n=1} (a_s^{(n_l)})^n
                            \sum\limits_{k=0}^n c_{nl} \log(\mu_R^2/\mu_F^2)
"""
matching_coeffs_up[1, 1] = 4.0 / 3.0 * constants.TR
matching_coeffs_up[2, 0] = 14.0 / 3.0
matching_coeffs_up[2, 1] = 38.0 / 3.0
matching_coeffs_up[2, 2] = 4.0 / 9.0

# inversion of the matching coefficients
_c = matching_coeffs_up

matching_coeffs_down = np.zeros_like(matching_coeffs_up)
"""
Matching coefficients :cite:`Schroder:2005hy` :cite:`Chetyrkin:2005ia` at threshold
when moving to a regime with *less* flavors.

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
"""

matching_coeffs_down[1, 1] = -_c[1, 1]
matching_coeffs_down[2, 0] = -_c[2, 0]
matching_coeffs_down[2, 1] = -_c[2, 1]
matching_coeffs_down[2, 2] = 2.0 * _c[1, 1] ** 2 - _c[2, 2]
