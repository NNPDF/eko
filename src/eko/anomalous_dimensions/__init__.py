# -*- coding: utf-8 -*-
r"""
  This module contains the Altarelli-Parisi splitting kernels.

  Normalization is given by

  .. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

  with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
  The 3-loop references are given for the non-singlet :cite:`Moch:2004pa`
  and singlet :cite:`Vogt:2004mw` case, which contain also the lower
  order results. The results are also determined in Mellin space in
  terms of the anomalous dimensions (note the additional sign!)

  .. math::
    \gamma(N) = - \mathcal{M}[\mathbf{P}(x)](N)
"""
