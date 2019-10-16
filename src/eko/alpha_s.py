# -*- coding: utf-8 -*-
r"""
This file contains the QCD beta function coefficients and the handling of the running
coupling :math:`\alpha_s`.

Normalization is given by

.. math::
      \frac{da}{d\ln\mu^2} = \beta(a) \
      = - \sum\limits_{n=0} \beta_n a^{n+2} \quad \text{with}~ a = \frac{\alpha_s(\mu^2)}{4\pi}


References
----------
  The 5-loop references are [1]_ [2]_ [3]_ which also include the lower order results.
  We use the Herzog paper [1]_ as our main reference.

  .. [1] F. Herzog et al. "The five-loop beta function of Yang-Mills theory with fermions"
     In: JHEP 02 (2017), p. 090. doi: 10.1007/JHEP02(2017)090. arXiv: 1701.01404 [hep-ph].

  .. [2] Thomas Luthe et al. "Towards the five-loop Beta function for a general gauge group"
     In: JHEP 07 (2016), p. 127. doi: 10.1007/JHEP07(2016)127. arXiv: 1606.08662 [hep-ph].

  .. [3] P. A. Baikov et al. "Five-Loop Running of the QCD coupling constant"
         In: Phys. Rev. Lett. 118.8 (2017), p. 082002. doi: 10.1103/PhysRevLett.118.082002.
         arXiv: 1606.08659 [hep-ph].
"""
import numpy as np
from eko import t_float
from eko.constants import Constants

def beta_0(nf : int, CA : t_float, CF : t_float, Tf : t_float): # pylint: disable=unused-argument
    """Computes the first coefficient of the QCD beta function

    Implements Eq. (3.1) of [1]_. For the conventions on normalization see header comment.
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
    beta_0 = 11./3. * CA - 4./3. * Tf * nf
    return beta_0

def beta_1(nf : int, CA : t_float, CF : t_float, Tf : t_float):
    """Computes the second coefficient of the QCD beta function

    Implements Eq. (3.2) of [1]_. For the conventions on normalization see header comment.

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
    return 34./3. * CA*CA \
         - 20./3. * CA * Tf * nf \
         - 4.     * CF * Tf * nf

def beta_2(nf : int, CA : t_float, CF : t_float, Tf : t_float):
    """Computes the third coefficient of the QCD beta function

    Implements Eq. (3.3) of [1]_. For the conventions on normalization see header comment.

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
    return 2857./54. * CA*CA*CA \
         - 1415./27. * CA*CA * Tf * nf \
         - 205./9.   * CF * CA * Tf * nf \
         + 2.        * CF*CF * Tf * nf \
         + 44./9.    * CF * Tf*Tf * nf*nf \
         + 158./27.  * CA * Tf*Tf * nf*nf

def a_s(order : int, alpha_s_ref : t_float, scale_ref : t_float, scale_to : t_float, nf : int, \
            method : str): # pylint: disable=unused-argument
    """Evolves the running coupling of QCD.

    For the conventions on normalization see header comment. Note that both scale parameters, \
    :math:`\\mu_0^2` and :math:`Q^2`, have to be given as squared values.

    Parameters
    ----------
      order : int
         evaluated order of beta function
      alpha_s_ref : t_float
         alpha_s at the reference scale :math:`\\alpha_s(\\mu_0^2)`
      scale_ref : t_float
         reference scale :math:`\\mu_0^2`
      scale_to : t_float
         final scale to evolve to :math:`Q^2`
      nf : int
         Number of active flavours (is passed to the beta function)
      method : {"analytic"}
         Applied method to solve the beta function

    Returns
    -------
    a_s : t_float
      strong coupling :math:`a_s(Q^2) = \\frac{\\alpha_s(Q^2)}{4\\pi}`
    """
    # TODO implement more complex runnings (we may take a glimpse into LHAPDF)
    # TODO change reference arguments of a_s also to a_s (instead of alpha_s_ref)?
    # for now: LO analytic
    c = Constants()
    beta0 = beta_0(nf, c.CA, c.CF, c.TF)
    L = np.log(scale_to/scale_ref)
    return alpha_s_ref / (4.*np.pi + beta0 * alpha_s_ref * L)
