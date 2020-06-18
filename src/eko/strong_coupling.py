# -*- coding: utf-8 -*-
r"""
    This file contains the QCD beta function coefficients and the handling of the running
    coupling :math:`\alpha_s`.

    See :doc:`pQCD ingredients </Theory/pQCD>`.
"""

import numpy as np
import numba as nb

from eko import t_float
from eko import constants
from eko import thresholds


@nb.njit
def beta_0(
    nf: int, CA: t_float, CF: t_float, TF: t_float
):  # pylint: disable=unused-argument
    """
        Computes the first coefficient of the QCD beta function.

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
        Computes the second coefficient of the QCD beta function.

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

        Note that

        - all scale parameters, `scale_ref`, `scale_to`, `thresholds`,
          have to be given as squared values
        - although, we only provide methods for
          :math:`a_s = \frac{\alpha_s(\mu^2)}{4\pi}` the reference value has to be
          given in terms of :math:`\alpha_s(\mu_0^2)` due to legacy reasons

        Normalization is given by :cite:`Herzog:2017ohr`:

        .. math::
            \frac{da_s}{d\ln\mu^2} = \beta(a_s) \
            = - \sum\limits_{n=0} \beta_n a_s^{n+2} \quad
            \text{with}~ a_s = \frac{\alpha_s(\mu^2)}{4\pi}

        See :doc:`pQCD ingredients </Theory/pQCD>`.

        Parameters
        ----------
            constants: eko.constants.Constants
                An instance of :class:`~eko.constants.Constants`
            alpha_s_ref : t_float
                alpha_s(!) at the reference scale :math:`\alpha_s(\mu_0^2)`
            scale_ref : t_float
                reference scale :math:`\mu_0^2`
            threshold_holder : eko.thresholds.Threshold
                An instance of :class:`~eko.thresholds.Threshold`
            order: int
                Evaluated order of the beta function: ``0`` = LO, ...
            method : ["analytic"]
                Applied method to solve the beta function

        Examples
        --------
            >>> c = Constants()
            >>> alpha_ref = 0.35
            >>> scale_ref = 2
            >>> threshold_holder = Threshold( ... )
            >>> alpha_s = StrongCoupling(c, alpha_ref, scale_ref, threshold_holder)
            >>> q2 = 91.1
            >>> alpha_s(q2)
            0.118
            >>> q02 = 50.0
            >>> alpha_s.delta_t(q02, q2)
            0.54
    """

    def __init__(
        self,
        consts,
        alpha_s_ref,
        scale_ref,
        threshold_holder,
        order=0,
        method="analytic",
    ):
        # Sanity checks
        if not isinstance(consts, constants.Constants):
            raise ValueError("Needs a Constants instance")
        if alpha_s_ref <= 0:
            raise ValueError(f"alpha_s_ref has to be positive - got {alpha_s_ref}")
        if scale_ref <= 0:
            raise ValueError(f"scale_ref has to be positive - got {scale_ref}")
        if not isinstance(threshold_holder, thresholds.ThresholdsConfig):
            raise ValueError("Needs a Threshold instance")
        if order not in [0, 1, 2]:
            raise NotImplementedError("a_s beyond NNLO is not implemented")
        self._order = order
        if method not in ["analytic"]:
            raise ValueError(f"Unknown method {method}")
        self._method = method

        self._constants = consts
        # Move alpha_s from q2_ref to scale_ref
        area_path = threshold_holder.get_path_from_q2_ref(scale_ref)
        # Now run through the list in reverse to set the alpha at q0
        input_as_ref = alpha_s_ref / 4.0 / np.pi  # convert to a_s
        for area in reversed(area_path):
            scale_to = area.q2_ref
            area_nf = area.nf
            new_as = self._compute(input_as_ref, area_nf, scale_ref, scale_to)
            scale_ref = scale_to
            input_as_ref = new_as

        # At this point we moved the value of alpha_s down to q0, store
        self._as_ref = new_as
        self._threshold_holder = threshold_holder

    @property
    def as_ref(self):
        """ reference value """
        return self._as_ref

    @property
    def q2_ref(self):
        """ reference scale """
        return self._threshold_holder.q2_ref

    @classmethod
    def from_dict(cls, setup, constants, thresholds):
        """
            Create object from theory dictionary.

            Read keys:

                - alphas : required, reference value for  alpha_s (!)
                - Qref : required, reference value in GeV (!)

            Parameters
            ----------
                setup : dict
                    theory dictionary
                constants : Constants
                    Color constants
                thresholds : Threshold
                    threshold configuration

            Returns
            -------
                cls : StrongCoupling
                    created object
        """
        alpha_ref = setup["alphas"]
        q2_alpha = pow(setup["Qref"], 2)
        return cls(constants, alpha_ref, q2_alpha, thresholds)

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
        # common vars
        beta0 = beta_0(nf, self._constants.CA, self._constants.CF, self._constants.TF)
        lmu = np.log(scale_to / scale_from)
        den = 1.0 + beta0 * as_ref * lmu
        # LO
        as_LO = as_ref / den
        res = as_LO
        # NLO
        if self._order >= 1:
            beta1 = beta_1(
                nf, self._constants.CA, self._constants.CF, self._constants.TF
            )
            b1 = beta1 / beta0
            as_NLO = as_LO * (1 - b1 * as_LO * np.log(den))
            res = as_NLO
            # NNLO
            if self._order == 2:
                beta2 = beta_2(
                    nf, self._constants.CA, self._constants.CF, self._constants.TF
                )
                b2 = beta2 / beta0
                res = as_LO * (
                    1
                    + as_LO * (as_LO - as_ref) * (b2 - b1 ** 2)
                    + as_NLO * b1 * np.log(as_NLO / as_ref)
                )

        return res

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
        # at the moment everything is analytic - and type has been checked in the constructor
        return self._compute_analytic(*args)

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
        # Set up the path to follow in order to go from q2_0 to q2_ref
        final_alpha = self._as_ref
        area_path = self._threshold_holder.get_path_from_q2_ref(scale_to)
        # TODO set up a cache system here
        for area in area_path:
            q2_from = area.q2_ref
            q2_to = area.q2_towards(scale_to)
            if np.isclose(q2_from, q2_to):
                continue
            area_nf = area.nf
            final_alpha = self._compute(final_alpha, area_nf, q2_from, q2_to)
        return final_alpha

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
        return self(scale_to)

    def _param_t(self, scale_to):
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
        return np.log(1.0 / self(scale_to))

    def delta_t(self, scale_from, scale_to):
        """
            Compute evolution parameter :math:`\\Delta t(Q_0^2, Q_1^2) = t(Q_1^2)-t(Q_0^2)`
            with :math:`t(Q^2) = log(1/a_s(Q^2))`.

            Parameters
            ----------
                scale_from : t_float
                    scale to evolve from :math:`Q_0^2`
                scale_to : t_float
                    final scale to evolve to :math:`Q_1^2`

            Returns
            -------
                delta : t_float
                    evolution parameter :math:`\\Delta t(Q_0^2, Q_1^2)`
        """
        delta = self._param_t(scale_to) - self._param_t(scale_from)
        return delta
