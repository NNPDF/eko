# -*- coding: utf-8 -*-
r"""
This files sets the physical constants.

"""

NC = 3
"""the number of colors"""

TR = float(1.0 / 2.0)
"""the normalization of fundamental generators - defaults to :math:`T_R = 1/2`"""

CA = float(NC)
"""second Casimir constant in the adjoint representation - defaults to :math:`N_C = 3`"""

CF = float((NC * NC - 1.0) / (2.0 * NC))
"""second Casimir constant in the fundamental representation - defaults to :math:`\frac{N_C^2-1}{2N_C} = 4/3`"""

eu2 = 4.0 / 9
"""up quarks charge squared"""

ed2 = 1.0 / 9
"""down quarks charge squared"""


def update_colors(nc):
    """
    Updates the number of colors to :math:`NC = nc` and the Casimirs for a generic value of :math:`NC`

    Parameters
    ----------
      nc : int
        Number of colors
    """
    global NC, CA, CF  # pylint: disable=global-statement

    NC = int(nc)
    CA = float(NC)
    CF = float((NC * NC - 1.0) / (2.0 * NC))


def number_of_flavors(nf):
    """
    Computes the number of up (nu) and down (nd) flavors

    Parameters
    ----------
        nf : int
            Number of active flavors

    Returns
    -------
        nu : int
        nd : int
    """
    if nf == 2:
        nu = 1
        nd = 1
    elif nf == 3:
        nu = 1
        nd = 2
    elif nf == 4:
        nu = 2
        nd = 2
    elif nf == 5:
        nu = 2
        nd = 3
    elif nf == 6:
        nu = 3
        nd = 3
    else:
        raise NotImplementedError("Selected nf is not implemented")
    return nu, nd
