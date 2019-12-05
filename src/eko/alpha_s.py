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
  The 5-loop references are :cite:`Herzog:2017ohr` :cite:`Luthe:2016ima` :cite:`Baikov:2016tgj`
  which also include the lower order results.
  We use the Herzog paper :cite:`Herzog:2017ohr` as our main reference.
"""
import numpy as np
import numba as nb
from eko import t_float
from eko.constants import Constants


@nb.njit
def beta_0(
    nf: int, CA: t_float, CF: t_float, Tf: t_float
):  # pylint: disable=unused-argument
    """Computes the first coefficient of the QCD beta function

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
    beta_0 = 11.0 / 3.0 * CA - 4.0 / 3.0 * Tf * nf
    return beta_0


@nb.njit
def beta_1(nf: int, CA: t_float, CF: t_float, Tf: t_float):
    """Computes the second coefficient of the QCD beta function

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
    """Computes the third coefficient of the QCD beta function

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


class Alphas_Dispatcher:
    """
    Generates the alpha_s functions for a given number of flavours for a given method
    for a given reference scale

    Note that all scale parameters, :math:`\\mu_0^2` and :math:`Q^2`,
    have to be given as squared values.

    This class is callable with a signature (q0: double, q2: array/double)


    Parameters
    ----------
        constants:
            An instance of the Constants class
        alpha_s_ref : t_float
            alpha_s at the reference scale :math:`\\alpha_s(\\mu_0^2)`
        scale_ref : t_float
            reference scale :math:`\\mu_0^2`
        nf : int
            Number of active flavours (is passed to the beta function)
        method : {"analytic"}
            Applied method to solve the beta function
        order: int (default = 0)
            Evaluated order of the beta function
    """

    def __init__(
        self, constants, alpha_s_ref, scale_ref, nf, method="analytic", order=0
    ):
        self.beta = None
        self.order = order
        self.set_order(constants, nf)
        self.scale_ref = scale_ref
        self.alpha_s_ref = alpha_s_ref

    def set_order(self, constants, nf):
        """
        Parameters
        ----------
            constants:
                An instance of the Constants class
            nf : int
                Number of active flavours (is passed to the beta function)
        """
        if self.order == 0:
            self.beta = beta_0(nf, constants.CA, constants.CF, constants.TF)
        else:
            raise NotImplementedError("Alpha_s beyond LO not implemented")

    def __call__(self, scale_to):
        """
        Parameters
        ----------
            scale_to : t_float (or array)
                final scale to evolve to :math:`Q^2`

        Returns
        -------
            a_s : t_float (or array)
                strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`
        """
        L = np.log(scale_to / self.scale_ref)
        fpi = 4.0 * np.pi
        result = self.alpha_s_ref / (fpi + self.beta * self.alpha_s_ref * L)
        return result
