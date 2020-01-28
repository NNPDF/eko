# -*- coding: utf-8 -*-
r"""
    This file contains the QCD beta function coefficients and the handling of the running
    coupling :math:`\alpha_s`.

    Normalization is given by :cite:`Herzog:2017ohr`

    .. math::
        \frac{da}{d\ln\mu^2} = \beta(a) \
        = - \sum\limits_{n=0} \beta_n a^{n+2} \quad \text{with}~ a = \frac{\alpha_s(\mu^2)}{4\pi}

    References
    ----------
        The 5-loop references are :cite:`Herzog:2017ohr`, :cite:`Luthe:2016ima`,
        :cite:`Baikov:2016tgj` which also include the lower order results.
        We use the Herzog paper :cite:`Herzog:2017ohr` as our main reference.
"""
import numpy as np
import numba as nb
import mpmath as mp
from eko import t_float
from eko.constants import Constants


#@nb.njit
def beta_0(
    nf: int, CA: t_float, CF: t_float, Tf: t_float
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
            Tf : t_float
                fundamental normalization factor

        Returns
        -------
            beta_0 : t_float
                first coefficient of the QCD beta function :math:`\\beta_0^{n_f}`
    """
    beta_0 = mp.mpf('11.0') / mp.mpf('3.0') * CA - mp.mpf('4.0') / mp.mpf('3.0') * Tf * nf
    return beta_0


@nb.njit
def beta_1(nf: int, CA: t_float, CF: t_float, Tf: t_float):
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
            Tf : t_float
                fundamental normalization factor

        Returns
        -------
        beta_1 : t_float
            second coefficient of the QCD beta function :math:`\\beta_1^{n_f}`
    """
    b_ca2 = 34.0 / 3.0 * CA * CA
    b_ca = -20.0 / 3.0 * CA * Tf * nf
    b_cf = -4.0 * CF * Tf * nf
    beta_1 = b_ca2 + b_ca + b_cf
    return beta_1


@nb.njit
def beta_2(nf: int, CA: t_float, CF: t_float, Tf: t_float):
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
        Tf : t_float
            fundamental normalization factor.

        Returns
        -------
        beta_2 : t_float
            third coefficient of the QCD beta function :math:`\\beta_2^{n_f}`
    """
    beta_2 = (
        2857.0 / 54.0 * CA * CA * CA
        - 1415.0 / 27.0 * CA * CA * Tf * nf
        - 205.0 / 9.0 * CF * CA * Tf * nf
        + 2.0 * CF * CF * Tf * nf
        + 44.0 / 9.0 * CF * Tf * Tf * nf * nf
        + 158.0 / 27.0 * CA * Tf * Tf * nf * nf
    )
    return beta_2


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

    #@nb.njit
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
        L = mp.log(mp.convert(scale_to) / mp.convert(scale_ref))
        if order == 0:
            return mp.convert(alpha_s_ref) / (4.0 * mp.pi + beta0 * alpha_s_ref * L)
        else:
            raise NotImplementedError("Alpha_s beyond LO not implemented")

    return a_s


def get_evolution_params(
    setup: dict, constants: Constants, nf: t_float, mu2init: t_float, mu2final: t_float
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
    a0 = a_s(pto, mu2init)
    a1 = a_s(pto, mu2final)
    # evolution parameters
    t0 = mp.log(1.0 / a0)
    t1 = mp.log(1.0 / a1)
    return t1 - t0
