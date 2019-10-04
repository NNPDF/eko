# -*- coding: utf-8 -*-
"""
This file contains the QCD beta function coefficients and the handling of the running
coupling alpha_s
"""

# the 5-Loop references are
# - F. Herzog et al. “The five-loop beta function of Yang-Mills theory with fermions”
#   In: JHEP 02 (2017), p. 090. doi: 10.1007/JHEP02(2017)090. arXiv: 1701.01404 [hep-ph]
# - Thomas Luthe et al. “Towards the five-loop Beta function for a general gauge group”
#   In: JHEP 07 (2016), p. 127. doi: 10.1007/JHEP07(2016)127. arXiv: 1606.08662 [hep-ph]
# - P. A. Baikov, K. G. Chetyrkin, and J. H. Kühn. “Five-Loop Running of the QCD coupling constant”
#   In: Phys. Rev. Lett. 118.8 (2017), p. 082002. doi: 10.1103/PhysRevLett.118.082002.
#   arXiv: 1606.08659 [hep-ph]
# which also include the lower order results

# global color constants 
# TODO hide them? hard code them?
NC = 3.
CA = NC
CF = (NC*NC - 1.)/(2. * NC)
Tf = 1./2.

def beta_0(nf : int):
    """returns the first coefficient of the QCD beta function with nf active flavours"""
    return 11./3. * CA - 4./3. * Tf * nf

def beta_1(nf : int):
    """returns the second coefficient of the QCD beta function with nf active flavours"""
    return 34./3. * CA*CA - 20./3. * CA * Tf * nf - 4. * CF * Tf * nf

def beta_2(nf : int):
    """returns the third coefficient of the QCD beta function with nf active flavours"""
    return 2857./54. * CA*CA*CA - 1415./27. * CA*CA * Tf * nf - 205./9. * CF * CA * Tf * nf\
            + 2. * CF*CF * Tf * nf + 44./9. * CF * Tf*Tf * nf*nf + 158./27. * CA * Tf*Tf * nf*nf

def alpha_s():
    """Compute strong coupling

    Args:

    Returns:
        strong coupling alpha_s
    """
    # TODO implement actual running (we may take a glimpse into LHAPDF)
    # TODO determine actual arguments; they should include
    #   - perturbation order
    #   - renormalasation scale
    #   - number of active flavours
    #   - method? such as diff_eq, analytic
    # return fixed value for now
    return 0.118
