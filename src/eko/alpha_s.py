# -*- coding: utf-8 -*-
r"""
This file contains the QCD beta function coefficients and the handling of the running
coupling alpha_s.

The 5-loop references are
 - F. Herzog et al. "The five-loop beta function of Yang-Mills theory with fermions"
   In: JHEP 02 (2017), p. 090. doi: 10.1007/JHEP02(2017)090. arXiv: 1701.01404 [hep-ph].
 - Thomas Luthe et al. "Towards the five-loop Beta function for a general gauge group"
   In: JHEP 07 (2016), p. 127. doi: 10.1007/JHEP07(2016)127. arXiv: 1606.08662 [hep-ph].
 - P. A. Baikov et al. "Five-Loop Running of the QCD coupling constant"
   In: Phys. Rev. Lett. 118.8 (2017), p. 082002. doi: 10.1103/PhysRevLett.118.082002.
   arXiv: 1606.08659 [hep-ph].
which also include the lower order results. We use the Herzog paper as reference.
Normalization is given by
$\frac{da}{d\ln\mu^2} = \beta(a) \
  = - \sum\limits_{n=0} \beta_n a^{n+2}$ with $a = \frac{\alpha(\mu^2)}{4\pi}$
"""

# TODO move global color constants + dtype outside
NC = 3.
CA = NC
CF = (NC*NC - 1.)/(2. * NC)
Tf = 1./2.
dtype = float

def beta_0(nf : int, CA : dtype, CF : dtype, Tf : dtype): # pylint: disable=unused-argument
    r"""Computes the first coefficient of the QCD beta function with nf active flavours.
    Implements 10.1007/JHEP02(2017)090 Eq. (3.1).
    Normalization is given by
    $\frac{da}{d\ln\mu^2} = \beta(a) \
         = - \sum\limits_{n=0} \beta_n a^{n+2}$ with $a = \frac{\alpha(\mu^2)}{4\pi}$
    For the sake of unification we keep a unique function signature.

    Args:
     - nf number of active flavours
     - CA Casimir constant of adjoint representation
     - CF Casimir constant of fundamental representation (which is actually not used in beta_0)
     - Tf fundamental normalization factor

    Returns
      first coefficient of the QCD beta function $\beta_0^{n_f}$
    """
    return 11./3. * CA - 4./3. * Tf * nf

def beta_1(nf : int, CA : dtype, CF : dtype, Tf : dtype):
    r"""Computes the second coefficient of the QCD beta function with nf active flavours.
    Implements 10.1007/JHEP02(2017)090 Eq. (3.2).
    Normalization is given by
    $\frac{da}{d\ln\mu^2} = \beta(a) \
         = - \sum\limits_{n=0} \beta_n a^{n+2}$ with $a = \frac{\alpha(\mu^2)}{4\pi}$

    Args:
     - nf number of active flavours
     - CA Casimir constant of adjoint representation
     - CF Casimir constant of fundamental representation
     - Tf fundamental normalization factor

    Returns
      second coefficient of the QCD beta function $\beta_1^{n_f}$
    """
    return 34./3. * CA*CA \
         - 20./3.  * CA * Tf * nf \
         - 4.      * CF * Tf * nf

def beta_2(nf : int):
    r"""Computes the third coefficient of the QCD beta function with nf active flavours.
    Implements 10.1007/JHEP02(2017)090 Eq. (3.3).
    Normalization is given by
    $\frac{da}{d\ln\mu^2} = \beta(a) \
         = - \sum\limits_{n=0} \beta_n a^{n+2}$ with $a = \frac{\alpha(\mu^2)}{4\pi}$

    Args:
     - nf number of active flavours
     - CA Casimir constant of adjoint representation
     - CF Casimir constant of fundamental representation
     - Tf fundamental normalization factor

    Returns
      third coefficient of the QCD beta function $\beta_2^{n_f}$
    """
    return 2857./54. * CA*CA*CA \
         - 1415./27. * CA*CA * Tf * nf \
         - 205./9.   * CF * CA * Tf * nf \
         + 2.        * CF*CF * Tf * nf \
         + 44./9.    * CF * Tf*Tf * nf*nf \
         + 158./27.  * CA * Tf*Tf * nf*nf

def alpha_s():
    """Compute strong coupling

    Args:

    Returns:
        strong coupling alpha_s
    """
    # TODO implement actual running (we may take a glimpse into LHAPDF)
    # TODO determine actual arguments; they should include
    #   - perturbation order
    #   - renormalization scale
    #   - number of active flavours
    #   - method? such as diff_eq, analytic
    # return fixed value for now
    return 0.118
