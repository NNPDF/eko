r"""The unpolarized, time-like Altarelli-Parisi splitting kernels.

Normalization is given by

.. math::
    \mathbf{P}(x) = \sum\limits_{j=0} a_s^{j+1} \mathbf P^{(j)}(x)

with :math:`a_s = \frac{\alpha_S(\mu^2)}{4\pi}`.
"""

import numba as nb


@nb.njit(cache=True)
def gamma_ns(_order, _mode, _n, _nf):
    raise NotImplementedError("Polarised is not yet implemented")


@nb.njit(cache=True)
def gamma_singlet(_order, _n, _nf):
    raise NotImplementedError("Polarised is not yet implemented")
