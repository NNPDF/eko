r"""The polarized, time-like |OME|."""

import numba as nb


@nb.njit(cache=True)
def A_non_singlet(_matching_order, _n, _sx, _nf, _L):
    """Compute the non-singlet |OME|."""
    raise NotImplementedError("Time-like is not yet implemented")


@nb.njit(cache=True)
def A_singlet(_matching_order, _n, _sx, _nf, _L, _is_msbar, _sx_ns=None):
    """Compute the singlet |OME|."""
    raise NotImplementedError("Time-like is not yet implemented")
