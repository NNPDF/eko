# -*- coding: utf-8 -*-
r"""
This files sets the physical constants.


The constants are:

    - :math:`N_C` the number of colors - defaults to :math:`3`
    - :math:`T_R` the normalization of fundamental generators - defaults to :math:`1/2`
    - :math:`C_A` second Casimir constant in the adjoint representation - defaults to
      :math:`N_C = 3`
    - :math:`C_F` second Casimir constant in the fundamental representation - defaults to
      :math:`\frac{N_C^2-1}{2N_C} = 4/3`
"""

NC = 3
"""the number of colors"""

TR = float(1.0 / 2.0)
"""the normalization of fundamental generators"""

CA = float(NC)
"""second Casimir constant in the adjoint representation"""

CF = float((NC * NC - 1.0) / (2.0 * NC))
"""second Casimir constant in the fundamental representation"""
