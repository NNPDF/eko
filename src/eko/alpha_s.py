# -*- coding: utf-8 -*-
r"""
    This file contains the QCD beta function coefficients and the handling of the running
    coupling :math:`\alpha_s`.

    Normalization is given by :cite:`Herzog:2017ohr`

    .. math::
        \frac{da_s}{d\ln\mu^2} = \beta(a_s) \
        = - \sum\limits_{n=0} \beta_n a_s^{n+2} \quad
        \text{with}~ a_s = \frac{\alpha_s(\mu^2)}{4\pi}

    References
    ----------
        The 5-loop references are :cite:`Herzog:2017ohr`, :cite:`Luthe:2016ima`,
        :cite:`Baikov:2016tgj` which also include the lower order results.
        We use the Herzog paper :cite:`Herzog:2017ohr` as our main reference.
"""

import numpy as np
import numba as nb

from eko import t_float
from eko.constants import Constants


@nb.njit
def beta_0(
    nf: int, CA: t_float, CF: t_float, TF: t_float
):  # pylint: disable=unused-argument
    """
        Computes the first coefficient of the QCD beta function

        Implements Eq. (3.1) of :cite:`Herzog:2017ohr`.
        For the sake of unification we keep a unique function signature for *all* coefficients.

        Parameters
        ----------
            nf : int
                number of active flavours
            CA : t_float
                Casimir constant of adjoint representation
            CF : t_float
                Casimir constant of fundamental representation (which is actually not used here)
            TF : t_float
                fundamental normalization factor

        Returns
        -------
            beta_0 : t_float
                first coefficient of the QCD beta function :math:`\\beta_0^{n_f}`
    """
    beta_0 = 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * nf
    return beta_0


@nb.njit
def beta_1(nf: int, CA: t_float, CF: t_float, TF: t_float):
    """
        Computes the second coefficient of the QCD beta function

        Implements Eq. (3.2) of :cite:`Herzog:2017ohr`.

        Parameters
        ----------
            nf : int
                number of active flavours
            CA : t_float
                Casimir constant of adjoint representation
            CF : t_float
                Casimir constant of fundamental representation
            TF : t_float
                fundamental normalization factor

        Returns
        -------
        beta_1 : t_float
            second coefficient of the QCD beta function :math:`\\beta_1^{n_f}`
    """
    b_ca2 = 34.0 / 3.0 * CA * CA
    b_ca = -20.0 / 3.0 * CA * TF * nf
    b_cf = -4.0 * CF * TF * nf
    beta_1 = b_ca2 + b_ca + b_cf
    return beta_1


@nb.njit
def beta_2(nf: int, CA: t_float, CF: t_float, TF: t_float):
    """
        Computes the third coefficient of the QCD beta function

        Implements Eq. (3.3) of :cite:`Herzog:2017ohr`.

        Parameters
        ----------
        nf : int
            number of active flavours.
        CA : t_float
            Casimir constant of adjoint representation.
        CF : t_float
            Casimir constant of fundamental representation.
        TF : t_float
            fundamental normalization factor.

        Returns
        -------
        beta_2 : t_float
            third coefficient of the QCD beta function :math:`\\beta_2^{n_f}`
    """
    beta_2 = (
        2857.0 / 54.0 * CA * CA * CA
        - 1415.0 / 27.0 * CA * CA * TF * nf
        - 205.0 / 9.0 * CF * CA * TF * nf
        + 2.0 * CF * CF * TF * nf
        + 44.0 / 9.0 * CF * TF * TF * nf * nf
        + 158.0 / 27.0 * CA * TF * TF * nf * nf
    )
    return beta_2


class StrongCoupling:
    r"""
        Computes strong coupling constant :math:`a_s`.

        Note that all scale parameters, `scale_ref`, `scale_to`, `thresholds`,
        have to be given as squared values. Although we only provide methods for
        :math:`a_s = \frac{\alpha_s(\mu^2)}{4\pi}` the reference value has to be
        given in terms of :math:`\alpha_s(\mu_0^2)`.

        Normalization is given by :cite:`Herzog:2017ohr`:

        .. math::
            \frac{da_s}{d\ln\mu^2} = \beta(a_s) \
            = - \sum\limits_{n=0} \beta_n a_s^{n+2} \quad
            \text{with}~ a_s = \frac{\alpha_s(\mu^2)}{4\pi}


        Parameters
        ----------
            constants: Constants
                An instance of the Constants class
            alpha_s_ref : t_float
                alpha_s(!) at the reference scale :math:`\\alpha_s(\\mu_0^2)`
            scale_ref : t_float
                reference scale :math:`\\mu_0^2`
            order: int
                Evaluated order of the beta function
            number_scheme : {"FFNS", "VFNS"}
                number scheme
            nf : int
                Number of active flavours, if FFNS
            thresholds : list
                list of quark thresholds with {mc2,mb2,mt2}, if VFNS
            method : {"analytic"}
                Applied method to solve the beta function
    """

    def __init__(
        self,
        constants,
        alpha_s_ref,
        scale_ref,
        order,
        number_scheme,
        nf=None,
        thresholds=None,
        method="analytic",
    ):
        self._constants = constants
        self._scale_ref = scale_ref
        self._as_ref = alpha_s_ref / 4.0 / np.pi  # convert to a_s
        if order not in [0]:
            raise NotImplementedError("a_s beyond LO is not implemented")
        self._order = order
        if method not in ["analytic"]:
            raise ValueError(f"Unknown method {method}")
        self._method = method
        # setup number scheme
        self._set_number_scheme(number_scheme, nf, thresholds)

    def _set_number_scheme(self, number_scheme, nf, thresholds):
        """
            Sets the necessary configurations for the number scheme.

            Parameters
            ----------
                number_scheme: {"FFNS", "VFNS"}
                    number scheme
                nf : int
                    number of flavors (if necessary)
                threshold : array
                    threshold list (if necessary)
        """
        # FFNS -> one for all
        if number_scheme == "FFNS":
            if not isinstance(nf, int) or nf < 3 or nf > 6:
                raise ValueError(f"Needs nf in [3..6] for FFNS - got {nf}")
            self._configs = [
                {
                    "mu2min": 0,
                    "mu2max": np.inf,
                    "as_ref": self._as_ref,
                    "scale_ref": self._scale_ref,
                    "nf": nf,
                }
            ]
        # VFNS -> need to resort
        elif number_scheme == "VFNS":
            if not isinstance(thresholds, list) or len(thresholds) != 3:
                raise ValueError(
                    "For VFNS needs list with thresholds with exact 3 entries"
                )
            # update threshold list
            thresh_p = [0] + thresholds + [np.inf]
            # find initial step
            self._configs = []
            for k, __ in enumerate(thresh_p):
                if thresh_p[k] <= self._scale_ref <= thresh_p[k + 1]:
                    c = {
                        "mu2min": thresh_p[k],
                        "mu2max": thresh_p[k + 1],
                        "as_ref": self._as_ref,
                        "scale_ref": self._scale_ref,
                        "nf": 3 + k,
                    }
                    self._configs.append(c)
                    break
            if len(self._configs) == 0:
                raise ValueError(
                    "Couldn't find a correct matching of the reference values"
                )
            # fill upstairs
            for k in range(self._configs[0]["nf"] - 2, 4):
                low = self._configs[-1]
                as_thres = self._compute(low, thresh_p[k])
                c = {
                    "mu2min": thresh_p[k],
                    "mu2max": thresh_p[k + 1],
                    "as_ref": as_thres,
                    "scale_ref": thresh_p[k],
                    "nf": 3 + k,
                }
                self._configs.append(c)
            # fill downstairs
            for k in range(0, self._configs[0]["nf"] - 3):
                high = self._configs[0]
                as_thres = self._compute(high, thresh_p[k + 1])
                c = {
                    "mu2min": thresh_p[k],
                    "mu2max": thresh_p[k + 1],
                    "as_ref": as_thres,
                    "scale_ref": thresh_p[k + 1],
                    "nf": 3 + k,
                }
                self._configs = [c] + self._configs
        else:
            raise ValueError(f"Unknown number scheme {number_scheme}")

    def _compute_analytic(self, conf, scale_to):
        """
            Compute via analytic expression.

            Parameters
            ----------
                conf : dict
                    configuration
                scale_to : t_float
                    target scale

            Returns
            -------
                a_s : t_float
                    coupling at target scale
        """
        beta0 = beta_0(
            conf["nf"], self._constants.CA, self._constants.CF, self._constants.TF
        )
        L = np.log(scale_to / conf["scale_ref"])
        as_ref = conf["as_ref"]
        result = as_ref / (1.0 + beta0 * as_ref * L)
        # add higher orders ...
        return result

    def _compute(self, conf, scale_to):
        """
            Computes  for a given config according to method.

            Parameters
            ----------
                conf : dict
                    active configuration
                scale_to : t_float
                    final scale to evolve to :math:`Q^2`

            Returns
            -------
                a_s : t_float
                    strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`
        """
        if self._method == "analytic":
            return self._compute_analytic(conf, scale_to)
        raise ValueError(f"Unknown method {self._method}")

    def a_s(self, scale_to):
        """
            Computes strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`.

            Parameters
            ----------
                scale_to : t_float
                    final scale to evolve to :math:`Q^2`

            Returns
            -------
                a_s : t_float
                    strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`
        """
        # find configuration
        conf = None
        for c in self._configs:
            if c["mu2min"] <= scale_to <= c["mu2max"]:
                conf = c
                break
        if conf is None:
            raise ValueError(f"Couldn't find a valid configuration for {scale_to}")
        # compute
        return self._compute(conf, scale_to)

    def t(self, scale_to):
        """
            Computes evolution parameter :math:`t(Q^2) = \\log(1/a_s(Q^2))`.

            Parameters
            ----------
                scale_to : t_float
                    final scale to evolve to :math:`Q^2`

            Returns
            -------
                t : t_float
                    evolution parameter :math:`t(Q^2) = \\log(1/a_s(Q^2))`
        """
        return np.log(1.0 / self.a_s(scale_to))


def alpha_s_generator(
    c: Constants,
    alpha_s_ref: t_float,
    scale_ref: t_float,
    nf: int,
    method: str,  # pylint: disable=unused-argument
):
    """
        Generates the :math:`a_s` functions for a given configuration.

        Note that all scale parameters, :math:`\\mu_0^2` and :math:`Q^2`,
        have to be given as squared values.

        Parameters
        ----------
            constants : Constants
                physical constants
            alpha_s_ref : t_float
                alpha_s(!) at the reference scale :math:`\\alpha_s(\\mu_0^2)`
            scale_ref : t_float
                reference scale :math:`\\mu_0^2`
            nf : int
                Number of active flavours (is passed to the beta function)
            method : {"analytic"}
                Applied method to solve the beta function

        Returns
        -------
            a_s: function
                function(order, scale_to) which computes a_s for a given order at a given scale
    """
    # TODO implement more complex runnings (we may take a glimpse into LHAPDF)
    beta0 = beta_0(nf, c.CA, c.CF, c.TF)

    @nb.njit
    def a_s(order: int, scale_to: t_float):
        """
            Evalute :math:`a_s`.

            Parameters
            ----------
                order : int
                    evaluated order of beta function
                scale_to : t_float
                    final scale to evolve to :math:`Q^2`

            Returns
            -------
                a_s : t_float
                    strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`
        """
        L = np.log(scale_to / scale_ref)
        if order == 0:
            return alpha_s_ref / (4.0 * np.pi + beta0 * alpha_s_ref * L)
        else:
            raise NotImplementedError("Alpha_s beyond LO not implemented")

    return a_s


def get_evolution_params(
    setup: dict,
    constants: Constants,
    nf: t_float,
    mu2init: t_float,
    mu2final: t_float,
    mu2step=None,
):
    """
        Compute evolution parameters

        Parameters
        ----------
            setup: dict
                a dictionary with the theory parameters for the evolution
            constants : Constants
                physical constants
            nf : int
                number of active flavours
            mu2init : float
                initial scale
            mu2final : flaot
                final scale

        Returns
        -------
            delta_t : t_float
                scale difference
    """
    # setup params
    qref2 = setup["Qref"] ** 2
    pto = setup["PTO"]
    alphas = setup["alphas"]
    # Generate the alpha_s functions
    a_s = alpha_s_generator(constants, alphas, qref2, nf, "analytic")
    print("mu2init=", mu2init, "mu2final=", mu2final, "mu2step=", mu2step)
    print("nf=", nf)
    if False and mu2step is not None:
        a_s_low = alpha_s_generator(constants, alphas, qref2, nf - 1, "analytic")
        a_ref_low = a_s_low(pto, mu2step)
        print("a_ref_low = ", a_ref_low)
        print("a0_old = ", a_s(pto, mu2init))
        print("a1_old = ", a_s(pto, mu2final))
        a_s = alpha_s_generator(
            constants, a_ref_low * 4.0 * np.pi, mu2step, nf, "analytic"
        )
        print("a0_new = ", a_s(pto, mu2init))
        print("a1_new = ", a_s(pto, mu2final))
    a0 = a_s(pto, mu2init)
    a1 = a_s(pto, mu2final)
    # as0 = 0.23171656287356338/4/np.pi
    # as1 = 0.18837996737403412/4/np.pi
    # print("a0 = ",a0,"alpha_s_0 = ",a0*4*np.pi)
    # print("a0 = ",a1,"alpha_s_0 = ",a1*4*np.pi)
    # evolution parameters
    t0 = np.log(1.0 / a0)
    t1 = np.log(1.0 / a1)
    return t1 - t0
