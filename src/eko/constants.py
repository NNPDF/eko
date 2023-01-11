"""Sets the physical constants."""

import numba as nb

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

eu2 = 4.0 / 9
"""Up quarks charge squared."""

ed2 = 1.0 / 9
"""Down quarks charge squared."""


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
    if nf not in range(2, 6 + 1):
        raise NotImplementedError("Selected nf is not implemented")
    nu = nf // 2
    return nu


@nb.njit(cache=True)
def charge_combinations(nf):
    """
    Compute the combination of charges.

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
