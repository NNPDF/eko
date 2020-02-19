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
            threshold_holder : eko.thresholds.Threshold
                instance of the Threshold class
            order: int
                Evaluated order of the beta function
            method : {"analytic"}
                Applied method to solve the beta function
    """

    def __init__(
        self,
        constants,
        alpha_s_ref,
        scale_ref,
        threshold_holder,
        order=0,
        method="analytic",
    ):
        # Sanity checks
        if method not in ["analytic"]:
            raise ValueError(f"Unknown method {method}")
        self._method = method
        if order not in [0]:
            raise NotImplementedError("a_s beyond LO is not implemented")
        self._order = order

        self._constants = constants
        # Move alpha_s from qref to q0
        area_path = threshold_holder.get_path_from_q0(scale_ref)
        # Now run through the list in reverse to set the alpha at q0
        input_as_ref = alpha_s_ref / 4.0 / np.pi  # convert to a_s
        for area in reversed(area_path):
            scale_to = area.qref
            area_nf = area.nf
            new_alpha_s = self._compute(input_as_ref, area_nf, scale_ref, scale_to)
            scale_ref = scale_to
            input_as_ref = new_alpha_s

        # At this point we moved the value of alpha_s down to q0, store
        self._ref_alpha = new_alpha_s
        self._threshold_holder = threshold_holder

    # Hidden computation functions
    def _compute_analytic(self, as_ref, nf, scale_from, scale_to):
        """
            Compute via analytic expression.

            Parameters
            ----------
                as_ref: t_float
                    reference alpha_s
                nf: int
                    value of nf for computing alpha_s
                scale_from: t_float
                    reference scale
                scale_to : t_float
                    target scale

            Returns
            -------
                a_s : t_float
                    coupling at target scale
        """
        beta0 = beta_0(nf, self._constants.CA, self._constants.CF, self._constants.TF)
        L = np.log(scale_to / scale_from)
        a_s = as_ref / (1.0 + beta0 * as_ref * L)
        # add higher orders ...
        return a_s

    def _compute(self, *args):
        """
            Wrapper in order to pass the computation to the corresponding
            method (depending on the calculation method).
            This function has no knowledge of the incoming parameters
            as they are defined in the respective computation methods

            Parameters
            ----------
                `*args`: tuple
                    List of arguments accepted by the computational
                    method defined by self._method
                

            Returns
            -------
                a_s : t_float
                    strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`
        """
        if self._method == "analytic":
            return self._compute_analytic(*args)
        raise ValueError(f"Unknown method {self._method}")

    def __call__(self, scale_to):
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
        # Set up the path to follow in order to go from q0 to qref
        final_alpha = self._ref_alpha
        area_path = self._threshold_holder.get_path_from_q0(scale_to)
        # TODO set up a cache system here
        for area in area_path:
            q_from = area.qref
            q_to = area.q_towards(scale_to)
            if np.isclose(q_from, q_to):
                continue
            area_nf = area.nf
            final_alpha = self._compute(final_alpha, area_nf, q_from, q_to)
        return final_alpha

    def a_s(self, *args):
        return self(*args)

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
