r"""Manage running (and fixed) couplings.

Manage QCD coupling :math:`\alpha_s` and QED coupling :math:`\alpha`. We
provide an interface to access them simultaneously and provide several
strategies to solve the associated |RGE|.

See :doc:`pQCD ingredients </theory/pQCD>`.
"""

import logging
from typing import Dict, Iterable, List, Tuple

import numba as nb
import numpy as np
import numpy.typing as npt
import scipy

from . import constants, matchings
from .beta import b_qcd, b_qed, beta_qcd, beta_qed
from .io.types import EvolutionMethod, Order, SquaredScale
from .quantities.couplings import CouplingEvolutionMethod, CouplingsInfo
from .quantities.heavy_quarks import QuarkMassScheme

logger = logging.getLogger(__name__)


def couplings_mod_ev(mod_ev: EvolutionMethod) -> CouplingEvolutionMethod:
    """Map ModEv key to the available strong coupling evolution methods.

    Parameters
    ----------
    mod_ev : str
        evolution mode

    Returns
    -------
    str
        coupling mode
    """
    if mod_ev.value in [
        "EXA",
        "iterate-exact",
        "decompose-exact",
        "perturbative-exact",
    ]:
        return CouplingEvolutionMethod.EXACT
    if mod_ev.value in [
        "TRN",
        "truncated",
        "ordered-truncated",
        "EXP",
        "iterate-expanded",
        "decompose-expanded",
        "perturbative-expanded",
    ]:
        return CouplingEvolutionMethod.EXPANDED
    raise ValueError(f"Unknown evolution mode {mod_ev}")


@nb.njit(cache=True)
def exact_lo(ref, beta0, lmu):
    r"""Compute expanded solution at |LO|.

    Parameters
    ----------
    ref : float
        reference value of the coupling
    beta0 : float
        first coefficient of the beta function
    lmu : float
        logarithm of the ratio between target and reference scales

    Returns
    -------
    float
        coupling at target scale :math:`a(\mu_R^2)`
    """
    den = 1.0 + beta0 * ref * lmu
    return ref / den


@nb.njit(cache=True)
def expanded_nlo(ref, beta0, b1, lmu):
    r"""Compute expanded solution at |NLO|.

    Implement the default expression for |NLO| expanded solution, e.g. the one
    implemented in |APFEL|.

    Parameters
    ----------
    ref : float
        reference value of the coupling
    beta0 : float
        first coefficient of the beta function
    b1 : float
        second coefficient of the b function
    lmu : float
        logarithm of the ratio between target and reference scales

    Returns
    -------
    float
        coupling at target scale :math:`a(\mu_R^2)`
    """
    den = 1.0 + beta0 * ref * lmu
    a_LO = exact_lo(ref, beta0, lmu)
    as_NLO = a_LO * (1 - b1 * a_LO * np.log(den))
    return as_NLO


@nb.njit(cache=True)
def expanded_nnlo(ref, beta0, b1, b2, lmu):
    r"""Compute expanded solution at |NNLO|.

    Implement the |NNLO| expanded solution from |APFEL| (not the default expression)

    Parameters
    ----------
    ref : float
        reference value of the coupling
    beta0 : float
        first coefficient of the beta function
    b1 : float
        second coefficient of the b function
    b2 : float
        third coefficient of the b function
    lmu : float
        logarithm of the ratio between target and reference scales

    Returns
    -------
    float
        coupling at target scale :math:`a(\mu_R^2)`
    """
    a_LO = exact_lo(ref, beta0, lmu)
    a_NLO = expanded_nlo(ref, beta0, b1, lmu)
    res = a_LO * (
        1.0 + a_LO * (a_LO - ref) * (b2 - b1**2) + a_NLO * b1 * np.log(a_NLO / ref)
    )
    return res


@nb.njit(cache=True)
def expanded_n3lo(ref, beta0, b1, b2, b3, lmu):
    r"""Compute expanded solution at |N3LO|.

    Implement the |N3LO| expanded solution obtained via iterated solution of the RGE :cite:`Rottoli`

    Parameters
    ----------
    ref : float
        reference value of the coupling
    beta0 : float
        first coefficient of the beta function
    b1 : float
        second coefficient of the b function
    b2 : float
        third coefficient of the b function
    b3 : float
        fourth coefficient of the b function
    lmu : float
        logarithm of the ratio between target and reference scales

    Returns
    -------
    float
        coupling at target scale :math:`a(\mu_R^2)`
    """
    a_LO = exact_lo(ref, beta0, lmu)
    log_fact = np.log(a_LO)
    res = expanded_nnlo(ref, beta0, b1, b2, lmu)
    res += (
        a_LO**4
        / (2 * beta0**3)
        * (
            -2 * b1**3 * np.log(ref) ** 3
            + 5 * b1**3 * log_fact**2
            + 2 * b1**3 * log_fact**3
            + b1**3 * np.log(ref) ** 2 * (5 + 6 * log_fact)
            + 2 * beta0 * b1 * log_fact * (b2 + 2 * (b1**2 - beta0 * b2) * lmu * ref)
            - beta0**2
            * lmu
            * ref
            * (
                -2 * b1 * b2
                + 2 * beta0 * b3
                + (b1**3 - 2 * beta0 * b1 * b2 + beta0**2 * b3) * lmu * ref
            )
            - 2
            * b1
            * np.log(ref)
            * (
                5 * b1**2 * log_fact
                + 3 * b1**2 * log_fact**2
                + beta0 * (b2 + 2 * (b1**2 - beta0 * b2) * lmu * ref)
            )
        )
    )
    return res


@nb.njit(cache=True)
def expanded_qcd(ref, order, beta0, b_vec, lmu):
    r"""Compute QCD expanded solution at a given order.

    Parameters
    ----------
    ref : float
        reference value of the strong coupling
    order : int
        QCD order
    beta0 : float
        first coefficient of the beta function
    b_vec : list
        list of b function coefficients (including b0)
    lmu : float
        logarithm of the ratio between target and reference scales

    Returns
    -------
    float
        strong coupling at target scale :math:`a_s(\mu_R^2)`
    """
    res_as = ref
    # LO
    if order == 1:
        res_as = exact_lo(ref, beta0, lmu)
    # NLO
    if order == 2:
        res_as = expanded_nlo(ref, beta0, b_vec[1], lmu)
    # NNLO
    if order == 3:
        res_as = expanded_nnlo(ref, beta0, b_vec[1], b_vec[2], lmu)
    # N3LO
    if order == 4:
        res_as = expanded_n3lo(
            ref,
            beta0,
            b_vec[1],
            b_vec[2],
            b_vec[3],
            lmu,
        )
    return res_as


@nb.njit(cache=True)
def expanded_qed(ref, order, beta0, b_vec, lmu):
    r"""Compute QED expanded solution at a given order.

    Parameters
    ----------
    ref : float
        reference value of the QED coupling
    order : int
        QED order
    beta0 : float
        first coefficient of the beta function
    b_vec : list
        list of b function coefficients (including b0)
    lmu : float
            logarithm of the ratio between target and reference scales

    Returns
    -------
    float
        QED coupling at target scale :math:`a_em(\mu_R^2)`
    """
    res_aem = ref
    # LO
    if order == 1:
        res_aem = exact_lo(ref, beta0, lmu)
    # NLO
    if order == 2:
        res_aem = expanded_nlo(ref, beta0, b_vec[1], lmu)
    return res_aem


@nb.njit(cache=True)
def couplings_expanded_alphaem_running(
    order, couplings_ref, nf, nl, scale_from, scale_to, decoupled_running
):
    r"""Compute coupled expanded expression of the couplings for running
    alphaem.

    Implement Eqs. (17-18) from :cite:`Surguladze:1996hx`

    Parameters
    ----------
    order : tuple(int, int)
        perturbation order
    couplings_ref : numpy.ndarray
        reference alpha_s and alpha
    nf : int
        value of nf for computing the couplings
    nl : int
        number of leptons partecipating to alphaem running
    scale_from : float
        reference scale
    scale_to : float
        target scale
    decoupled_running : bool
        whether the running of the couplings is decoupled or not

    Returns
    -------
    numpy.ndarray
        couplings at target scale :math:`a(\mu_R^2)`
    """
    # common vars
    lmu = np.log(scale_to / scale_from)
    beta0_qcd = beta_qcd((2, 0), nf)
    b_vec_qcd = [b_qcd((i + 2, 0), nf) for i in range(order[0])]
    res_as = expanded_qcd(couplings_ref[0], order[0], beta0_qcd, b_vec_qcd, lmu)
    beta0_qed = beta_qed((0, 2), nf, nl)
    b_vec_qed = [b_qed((0, i + 2), nf, nl) for i in range(order[1])]
    res_aem = expanded_qed(couplings_ref[1], order[1], beta0_qed, b_vec_qed, lmu)
    # if order[0] >= 1 and order[1] >= 1:
    # order[0] is always >=1
    if not decoupled_running:
        if order[1] >= 1:
            res_as += (
                -(couplings_ref[0] ** 2)
                * b_qcd((2, 1), nf)
                * np.log(1 + beta0_qcd * couplings_ref[1] * lmu)
            )
            res_aem += (
                -(couplings_ref[1] ** 2)
                * b_qed((1, 2), nf, nl)
                * np.log(1 + beta0_qed * couplings_ref[0] * lmu)
            )
    return np.array([res_as, res_aem])


@nb.njit(cache=True)
def couplings_expanded_fixed_alphaem(order, couplings_ref, nf, scale_from, scale_to):
    r"""Compute coupled expanded expression of the couplings for fixed alphaem.

    Parameters
    ----------
    order : tuple(int, int)
        perturbation order
    couplings_ref : numpy.ndarray
        reference alpha_s and alpha
    nf : int
        value of nf for computing the couplings
    scale_from : float
        reference scale
    scale_to : float
        target scale

    Returns
    -------
    numpy.ndarray
        couplings at target scale :math:`a(\mu_R^2)`
    """
    # common vars
    lmu = np.log(scale_to / scale_from)
    res_as = couplings_ref[0]
    aem = couplings_ref[1]
    beta_qcd0 = beta_qcd((2, 0), nf)
    if order[1] >= 1:
        beta_qcd0 += aem * beta_qcd((2, 1), nf)
    b_vec = [beta_qcd((i + 2, 0), nf) / beta_qcd0 for i in range(order[0])]
    # LO
    if order[0] == 1:
        res_as = exact_lo(couplings_ref[0], beta_qcd0, lmu)
    # NLO
    if order[0] == 2:
        res_as = expanded_nlo(couplings_ref[0], beta_qcd0, b_vec[1], lmu)
    # NNLO
    if order[0] == 3:
        res_as = expanded_nnlo(couplings_ref[0], beta_qcd0, b_vec[1], b_vec[2], lmu)
    # N3LO
    if order[0] == 4:
        res_as = expanded_n3lo(
            couplings_ref[0],
            beta_qcd0,
            b_vec[1],
            b_vec[2],
            b_vec[3],
            lmu,
        )
    return np.array([res_as, aem])


_CouplingsCacheKey = Tuple[float, float, int, float, float]
"""Cache key containing (a0, a1, nf, scale_from, scale_to)."""


class Couplings:
    r"""Compute the strong and electromagnetic coupling constants :math:`a_s,
    a_{em}`.

    Note that

    - although, we only provide methods for
      :math:`a_i = \frac{\alpha_i(\mu^2)}{4\pi}` the reference value has to be
      given in terms of :math:`\alpha_i(\mu_0^2)` due to legacy reasons
    - the ``order`` refers to the perturbative order of the beta functions, i.e.
      the number of loops for QCD and QED respectively. QCD is always running with
      at least 1 loop, while QED might not run at all (so 0 loop).

    Normalization is given by :cite:`Herzog:2017ohr`:

    .. math::
        \frac{da_s(\mu^2)}{d\ln\mu^2} = \beta(a_s) \
        = - \sum\limits_{n=0} \beta_n a_s^{n+2}(\mu^2) \quad
        \text{with}~ a_s = \frac{\alpha_s(\mu^2)}{4\pi}

    See :doc:`pQCD ingredients </theory/pQCD>`.

    Parameters
    ----------
    couplings :
        reference configuration
    order :
        Number of loops in beta functions (QCD, QED)
    method :
        Applied method to solve the beta functions
    masses :
        list with quark masses squared
    hqm_scheme :
        heavy quark mass scheme
    thresholds_ratios :
        list with ratios between the matching scales and the mass squared
    """

    def __init__(
        self,
        couplings: CouplingsInfo,
        order: Order,
        method: CouplingEvolutionMethod,
        masses: List[SquaredScale],
        hqm_scheme: QuarkMassScheme,
        thresholds_ratios: Iterable[float],
    ):
        # Sanity checks
        def assert_positive(name, var):
            if var <= 0:
                raise ValueError(f"{name} has to be positive - got: {var}")

        assert_positive("alpha_s_ref", couplings.alphas)
        assert_positive("alpha_em_ref", couplings.alphaem)
        assert_positive("scale_ref", couplings.ref[0])
        if order[0] not in [1, 2, 3, 4]:
            raise NotImplementedError(
                "QCD order has to be an integer between 1 (LO) and 4 (N3LO)"
            )
        if order[1] not in [0, 1, 2]:
            raise NotImplementedError(
                "QED order has to be an integer between 0 (no running) and 2 (NLO)"
            )
        self.order = tuple(order)
        if method.value not in ["expanded", "exact"]:
            raise ValueError(f"Unknown method {method.value}")
        self.method = method.value

        nf_ref = couplings.ref[1]
        scheme_name = hqm_scheme.name
        self.alphaem_running = couplings.em_running
        self.decoupled_running = False

        # create new threshold object
        self.a_ref = np.array(couplings.values) / 4.0 / np.pi  # convert to a_s and a_em
        matching_scales = (np.array(masses) * np.array(thresholds_ratios)).tolist()
        self.thresholds_ratios = list(thresholds_ratios)
        self.atlas = matchings.Atlas(matching_scales, (couplings.ref[0] ** 2, nf_ref))
        self.hqm_scheme = scheme_name
        logger.info(
            "Strong Coupling: a_s(µ_R^2=%f)%s=%f=%f/(4π)",
            self.mu2_ref,
            f"^(nf={nf_ref})" if nf_ref else "",
            self.a_ref[0],
            self.a_ref[0] * 4 * np.pi,
        )
        if self.order[1] > 0:
            logger.info(
                "Electromagnetic Coupling: a_em(µ_R^2=%f)%s=%f=%f/(4π)\nalphaem"
                " running: %r\ndecoupled running: %r",
                self.mu2_ref,
                f"^(nf={nf_ref})" if nf_ref else "",
                self.a_ref[1],
                self.a_ref[1] * 4 * np.pi,
                self.alphaem_running,
                self.decoupled_running,
            )
        # cache
        self.cache: Dict[_CouplingsCacheKey, npt.NDArray] = {}

    @property
    def mu2_ref(self):
        """Return reference scale."""
        return self.atlas.origin[0]

    def unidimensional_exact(self, beta0, b_vec, u, a_ref, method, rtol):
        """Compute single coupling via decoupled |RGE|.

        Parameters
        ----------
        beta0 : float
            first coefficient of the beta function
        b_vec : list
            list of b function coefficients (including b0)
        u : float
            :math:`log(scale_to / scale_from)`
        a_ref : float
            reference alpha_s or alpha
        method : string
            method for solving the RGE
        rtol : float
            relative acuracy of the solution

        Returns
        -------
        float
            coupling at target scale :math:`a(Q^2)`
        """
        if len(b_vec) == 1:
            return exact_lo(a_ref, beta0, u)

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

    def compute_exact_alphaem_running(self, a_ref, nf, nl, scale_from, scale_to):
        """Compute couplings via |RGE| with running alphaem.

        Parameters
        ----------
        a_ref : numpy.ndarray
            reference alpha_s and alpha
        nf : int
            value of nf for computing alpha_i
        nl : int
            number of leptons partecipating to alphaem running
        scale_from : float
            reference scale
        scale_to : float
            target scale

        Returns
        -------
        numpy.ndarray
            couplings at target scale :math:`a(Q^2)`
        """
        # in LO fallback to expanded, as this is the full solution
        u = np.log(scale_to / scale_from)

        if self.order == (1, 0):
            return couplings_expanded_fixed_alphaem(
                self.order, a_ref, nf, scale_from, float(scale_to)
            )

        beta_qcd_vec = [beta_qcd((2, 0), nf)]
        beta_qcd_mix = 0
        beta_qed_mix = 0
        # NLO
        if self.order[0] >= 2:
            beta_qcd_vec.append(beta_qcd((3, 0), nf))
            # NNLO
            if self.order[0] >= 3:
                beta_qcd_vec.append(beta_qcd((4, 0), nf))
                # N3LO
                if self.order[0] >= 4:
                    beta_qcd_vec.append(beta_qcd((5, 0), nf))
        if self.order[1] == 0:
            b_qcd_vec = [
                beta_qcd_vec[i] / beta_qcd_vec[0] for i in range(self.order[0])
            ]
            rge_qcd = self.unidimensional_exact(
                beta_qcd_vec[0],
                b_qcd_vec,
                u,
                a_ref[0],
                method="Radau",
                rtol=1e-6,
            )
            # for order = (qcd, 0) with qcd > 1 we return the exact solution for the QCD RGE
            # while aem is constant
            return np.array([rge_qcd, a_ref[1]])
        if self.order[1] >= 1:
            beta_qed_vec = [beta_qed((0, 2), nf, nl)]
            if not self.decoupled_running:
                beta_qcd_mix = beta_qcd((2, 1), nf)
                beta_qed_mix = beta_qed((1, 2), nf, nl)  # order[0] is always at least 1
            if self.order[1] >= 2:
                beta_qed_vec.append(beta_qed((0, 3), nf, nl))

        # integration kernel
        def rge(_t, a, beta_qcd_vec, beta_qcd_mix, beta_qed_vec, beta_qed_mix):
            rge_qcd = -(a[0] ** 2) * (
                np.sum([a[0] ** k * b for k, b in enumerate(beta_qcd_vec)])
                + a[1] * beta_qcd_mix
            )
            rge_qed = -(a[1] ** 2) * (
                np.sum([a[1] ** k * b for k, b in enumerate(beta_qed_vec)])
                + a[0] * beta_qed_mix
            )
            res = np.array([rge_qcd, rge_qed])
            return res

        # let scipy solve
        res = scipy.integrate.solve_ivp(
            rge,
            (0, u),
            a_ref,
            args=[beta_qcd_vec, beta_qcd_mix, beta_qed_vec, beta_qed_mix],
            method="Radau",
            rtol=1e-6,
        )
        return np.array([res.y[0][-1], res.y[1][-1]])

    def compute_exact_fixed_alphaem(self, a_ref, nf, scale_from, scale_to):
        """Compute couplings via |RGE| with fixed alphaem.

        Parameters
        ----------
        a_ref : numpy.ndarray
            reference alpha_s and alpha
        nf : int
            value of nf for computing alpha_i
        scale_from : float
            reference scale
        scale_to : float
            target scale

        Returns
        -------
        numpy.ndarray
            couplings at target scale :math:`a(Q^2)`
        """
        u = np.log(scale_to / scale_from)

        if self.order in [(1, 0), (1, 1)]:
            return couplings_expanded_fixed_alphaem(
                self.order, a_ref, nf, scale_from, float(scale_to)
            )

        beta_vec = [beta_qcd((2, 0), nf)]
        # NLO
        if self.order[0] >= 2:
            beta_vec.append(beta_qcd((3, 0), nf))
            # NNLO
            if self.order[0] >= 3:
                beta_vec.append(beta_qcd((4, 0), nf))
                # N3LO
                if self.order[0] >= 4:
                    beta_vec.append(beta_qcd((5, 0), nf))
        if self.order[1] >= 1:
            beta_vec[0] += a_ref[1] * beta_qcd((2, 1), nf)
        b_vec = [i / beta_vec[0] for i in beta_vec]
        rge_qcd = self.unidimensional_exact(
            beta_vec[0],
            b_vec,
            u,
            a_ref[0],
            method="Radau",
            rtol=1e-6,
        )
        return np.array([rge_qcd, a_ref[1]])

    def compute(self, a_ref, nf, nl, scale_from, scale_to):
        """Compute actual couplings.

        This is a wrapper in order to pass the computation to the corresponding method
        (depending on the calculation method).

        Parameters
        ----------
        a_ref : numpy.ndarray
            reference a
        nf : int
            value of nf for computing alpha
        nl : int
            number of leptons partecipating to alphaem running
        scale_from : float
            reference scale
        scale_to : float
            target scale

        Returns
        -------
        numpy.ndarray
            couplings at target scale :math:`a(Q^2)`
        """
        key = (float(a_ref[0]), float(a_ref[1]), nf, nl, scale_from, float(scale_to))
        try:
            return self.cache[key].copy()
        except KeyError:
            # at the moment everything is expanded - and type has been checked in the constructor
            if self.method == "exact":
                if self.alphaem_running:
                    a_new = self.compute_exact_alphaem_running(
                        a_ref.astype(float), nf, nl, scale_from, scale_to
                    )
                else:
                    a_new = self.compute_exact_fixed_alphaem(
                        a_ref.astype(float), nf, scale_from, scale_to
                    )
            else:
                if self.alphaem_running:
                    a_new = couplings_expanded_alphaem_running(
                        self.order,
                        a_ref.astype(float),
                        nf,
                        nl,
                        scale_from,
                        float(scale_to),
                        self.decoupled_running,
                    )
                else:
                    a_new = couplings_expanded_fixed_alphaem(
                        self.order, a_ref.astype(float), nf, scale_from, float(scale_to)
                    )
            self.cache[key] = a_new.copy()
            return a_new

    def a(
        self,
        scale_to,
        nf_to=None,
    ):
        r"""Compute couplings :math:`a_i(\mu_R^2) =
        \frac{\alpha_i(\mu_R^2)}{4\pi}`.

        Parameters
        ----------
        scale_to : float
            final scale to evolve to :math:`\mu_R^2`
        nf_to : int
            final nf value

        Returns
        -------
        numpy.ndarray
            couplings :math:`a_i(\mu_R^2) = \frac{\alpha_i(\mu_R^2)}{4\pi}`
        """
        # Set up the path to follow in order to go from mu2_0 to mu2_ref
        final_a = self.a_ref.copy()
        path = self.atlas.path((scale_to, nf_to))
        is_downward = matchings.is_downward_path(path)
        shift = matchings.flavor_shift(is_downward)

        for k, seg in enumerate(path):
            # skip a very short segment, but keep the matching
            if not np.isclose(seg.origin, seg.target):
                nli = matchings.lepton_number(seg.origin)
                nlf = matchings.lepton_number(seg.target)
                if self.order[1] != 0 and nli != nlf:
                    # it means that MTAU is between origin and target:
                    # first we evolve from origin to MTAU with nli leptons
                    a_tmp = self.compute(
                        final_a, seg.nf, nli, seg.origin, constants.MTAU**2
                    )
                    # then from MTAU to target with nlf leptons
                    new_a = self.compute(
                        a_tmp, seg.nf, nlf, constants.MTAU**2, seg.target
                    )
                else:
                    new_a = self.compute(final_a, seg.nf, nli, seg.origin, seg.target)
            else:
                new_a = final_a
            # apply matching conditions: see hep-ph/9706430
            # - if there is yet a step to go
            if k < len(path) - 1:
                L = np.log(self.thresholds_ratios[seg.nf - shift])
                m_coeffs = (
                    compute_matching_coeffs_down(self.hqm_scheme, seg.nf - 1)
                    if is_downward
                    else compute_matching_coeffs_up(self.hqm_scheme, seg.nf)
                )
                fact = 1.0
                # shift
                for n in range(1, self.order[0]):
                    for l_pow in range(n + 1):
                        fact += new_a[0] ** n * L**l_pow * m_coeffs[n, l_pow]
                new_a[0] *= fact
            final_a = new_a
        return final_a

    def a_s(self, scale_to, nf_to=None):
        r"""Compute strong coupling.

        The strong oupling uses the normalization :math:`a_s(\mu_R^2) =
        \frac{\alpha_s(\mu_R^2)}{4\pi}`.

        Parameters
        ----------
        scale_to : float
            final scale to evolve to :math:`\mu_R^2`
        nf_to : int
            final nf value

        Returns
        -------
        a_s : float
            couplings :math:`a_s(\mu_R^2) = \frac{\alpha_s(\mu_R^2)}{4\pi}`
        """
        return self.a(scale_to, nf_to)[0]

    def a_em(self, scale_to, nf_to=None):
        r"""Compute electromagnetic coupling.

        The electromagnetic oupling uses the normalization :math:`a_em(\mu_R^2)
        = \frac{\alpha_em(\mu_R^2)}{4\pi}`.

        Parameters
        ----------
        scale_to : float
            final scale to evolve to :math:`\mu_R^2`
        nf_to : int
            final nf value

        Returns
        -------
        a_em : float
            couplings :math:`a_em(\mu_R^2) = \frac{\alpha_em(\mu_R^2)}{4\pi}`
        """
        return self.a(scale_to, nf_to)[1]


def compute_matching_coeffs_up(mass_scheme, nf):
    r"""Compute the upward matching coefficients.

    The matching coefficients :cite:`Schroder:2005hy,Chetyrkin:2005ia,Vogt:2004ns`
    are normalized at threshold when moving to a regime with *more* flavors.

    We follow notation of :cite:`Vogt:2004ns` (eq 2.43) for POLE masses.

    The *inverse* |MSbar| values instead are given in :cite:`Schroder:2005hy` (eq 3.1)
    multiplied by a factor of 4 (and 4^2 ...).

    .. math::
        a_s^{(n_l+1)} = a_s^{(n_l)} + \sum\limits_{n=1} (a_s^{(n_l)})^n
                                \sum\limits_{k=0}^n c_{nl} \log(\mu_R^2/\mu_F^2)

    Parameters
    ----------
    mass_scheme : str
        Heavy quark mass scheme: "POLE" or "MSBAR"
    nf : int
        number of active flavors in the lower patch

    Returns
    -------
    numpy.ndarray
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
    """Compute the downward matching coefficients.

    This is the inverse function to :meth:`compute_matching_coeffs_up`.

    Parameters
    ----------
    mass_scheme : str
        Heavy quark mass scheme: "POLE" or "MSBAR"
    nf : int
        number of active flavors in the lower patch

    Returns
    -------
    numpy.ndarray
        downward matching coefficient matrix
    """
    c_up = compute_matching_coeffs_up(mass_scheme, nf)
    return invert_matching_coeffs(c_up)


def invert_matching_coeffs(c_up):
    """Compute the perturbative inverse of the matching conditions.

    They can be obtained e.g. via Mathematica by:

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
    numpy.ndarray
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
