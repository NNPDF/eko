"""Sets the physical constants."""

import numba as nb
import numpy as np
from scipy.special import zeta

NC = 3
"""The number of colors."""

TR = float(1.0 / 2.0)
"""The normalization of fundamental generators.

Defaults to :math:`T_R = 1/2`.
"""

CA = float(NC)
"""Second Casimir constant in the adjoint representation.

Defaults to :math:`N_C = 3`.
"""

CF = float((NC * NC - 1.0) / (2.0 * NC))
r"""Second Casimir constant in the fundamental representation.

Defaults to :math:`\frac{N_C^2-1}{2N_C} = 4/3`.
"""

MTAU = 1.777
"""Mass of the tau."""

eu2 = 4.0 / 9
"""Up quarks charge squared."""

ed2 = 1.0 / 9
"""Down quarks charge squared."""


zeta2 = zeta(2)
r""":math:`\zeta(2)`"""

zeta3 = zeta(3)
r""":math:`\zeta(3)`"""

zeta4 = zeta(4)
r""":math:`\zeta(4)`"""

zeta5 = zeta(5)
r""":math:`\zeta(5)`"""

log2 = np.log(2)
r""":math:`\ln(2)`"""

li4half = 0.517479
""":math:`Li_{4}(1/2)`"""


def update_colors(nc):
    """Update the number of colors to :math:`NC = nc`.

    The Casimirs for a generic value of :math:`NC` are consistenly updated as
    well.

    Parameters
    ----------
    nc : int
      Number of colors
    """
    global NC, CA, CF  # pylint: disable=global-statement

    NC = int(nc)
    CA = float(NC)
    CF = float((NC * NC - 1.0) / (2.0 * NC))


@nb.njit(cache=True)
def uplike_flavors(nf):
    """Compute the number of up flavors.

    Parameters
    ----------
    nf : int
        Number of active flavors

    Returns
    -------
    nu : int
    """
    if nf > 6:
        raise NotImplementedError("Selected nf is not implemented")
    nu = nf // 2
    return nu


@nb.njit(cache=True)
def charge_combinations(nf):
    """Compute the combination of charges.

    Parameters
    ----------
    nf : int
        Number of active flavors

    Returns
    -------
    e2avg : float
    vue2m : float
    vde2m : float
    """
    nu = uplike_flavors(nf)
    nd = nf - nu
    e2avg = (nu * eu2 + nd * ed2) / nf
    vue2m = nu / nf * (eu2 - ed2)
    vde2m = nd / nf * (eu2 - ed2)
    e2delta = vde2m - vue2m + e2avg
    return e2avg, vue2m, vde2m, e2delta
