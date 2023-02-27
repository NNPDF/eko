r"""The polarized, space-like |OME|."""

import numba as nb
import numpy as np

from . import as1


@nb.njit(cache=True)
def A_singlet(matching_order, n, sx, nf, L, is_msbar, sx_ns=None):
    r"""
    Computes the tower of the singlet |OME|.

    Parameters
    ----------
        matching_order : tuple(int,int)
            perturbative matching_order
        n : complex
            Mellin variable
        sx : list
            singlet like harmonic sums cache
        nf: int
            number of active flavor below threshold
        L : float
            :math:``\ln(\mu_F^2 / m_h^2)``
        is_msbar: bool
            add the |MSbar| contribution
        sx_ns : list
            non-singlet like harmonic sums cache

    Returns
    -------
        A_singlet : numpy.ndarray
            singlet |OME|

    See Also
    --------
        ekore.matching_conditions.nlo.A_singlet_1 : :math:`A^{S,(1)}(N)`
        ekore.matching_conditions.nlo.A_hh_1 : :math:`A_{HH}^{(1)}(N)`
        ekore.matching_conditions.nlo.A_gh_1 : :math:`A_{gH}^{(1)}(N)`
        ekore.matching_conditions.nnlo.A_singlet_2 : :math:`A_{S,(2)}(N)`
    """
    A_s = np.zeros((matching_order[0], 3, 3), np.complex_)
    if matching_order[0] == 1:
        A_s[0] = as1.A_singlet(n, L)
    else:
        raise NotImplementedError(
            "Polarised, space-like is not yet implemented at this order"
        )
    return A_s


@nb.njit(cache=True)
def A_non_singlet(_matching_order, _n, _sx, _nf, _L, _is_msbar, _sx_ns=None):
    raise NotImplementedError("Polarised, space-like is not yet implemented")
